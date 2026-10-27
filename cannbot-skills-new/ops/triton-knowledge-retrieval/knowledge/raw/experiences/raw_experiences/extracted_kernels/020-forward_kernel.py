@triton.jit
def forward_kernel(
    q_ptr,  # Q: n x h x d
    k_ptr,  # K: n x kh x d
    v_ptr,  # V: n x kh x d
    t_ptr,  # topk_idx: kh x n x k
    o_ptr,  # O: n x h x d
    lse_ptr,  # LSE: h x n
    # seqlens
    cu_seqlens_q,
    cu_seqlens_k,
    # shape
    NUM_KV_HEADS,
    NUM_SHARE_Q_HEADS,
    HEAD_DIM,
    TOPK: tl.constexpr,
    # q loop num
    num_q_loop,
    # sm_scale
    sm_scale,
    # stride
    stride_qn,
    stride_qh,
    stride_qd,
    stride_kn,
    stride_kh,
    stride_kd,
    stride_vn,
    stride_vh,
    stride_vd,
    stride_th,
    stride_tn,
    stride_tk,
    stride_on,
    stride_oh,
    stride_od,
    stride_lh,
    stride_ln,
    # q block start offset for current batch
    q_block_offset,
    # META parameters
    BLOCK_SIZE_K: tl.constexpr,  # k block size
    BLOCK_SIZE_D: tl.constexpr,
    BLOCK_SIZE_H: tl.constexpr,
    BLOCK_SIZE_T: tl.constexpr,
):
    qk_scale = sm_scale * 1.44269504
    # get batch id and head id
    pid_b = tl.program_id(0)
    pid_kh = tl.program_id(1)
    pid_h = pid_kh * NUM_SHARE_Q_HEADS
    # local q block ID within current batch -> convert to global q block ID (add offset)
    pid_q_local = tl.program_id(2)
    pid_q = pid_q_local + q_block_offset  # global q block ID

    # get q k start and len after rmpad
    q_start = tl.load(cu_seqlens_q + pid_b)
    q_len = tl.load(cu_seqlens_q + pid_b + 1) - q_start
    k_start = tl.load(cu_seqlens_k + pid_b)
    k_len = tl.load(cu_seqlens_k + pid_b + 1) - k_start

    # skip blocks beyond actual sequence length (using global q block ID)
    if pid_q * num_q_loop >= q_len:
        return

    # calculate actual number of queries to process in current batch
    real_q_loop = min(num_q_loop, q_len - pid_q * num_q_loop)

    for j in range(real_q_loop):
        # calculate global query index (based on global q block ID)
        pid_q_j = pid_q * num_q_loop + j

        # init topk idx pointer
        off_t = tl.arange(0, BLOCK_SIZE_T)
        t_ptr_j = t_ptr + (q_start + pid_q_j) * stride_tn + pid_kh * stride_th
        topk_idx = tl.load(t_ptr_j + off_t * stride_tk, mask=off_t < TOPK, other=-1)
        real_topk = tl.sum(
            tl.where((topk_idx >= 0) & (topk_idx <= pid_q_j // BLOCK_SIZE_K), 1, 0),
            axis=0,
        )

        # init qkv pointer
        q_ptrs = tl.make_block_ptr(
            base=q_ptr + (q_start + pid_q_j) * stride_qn + pid_h * stride_qh,
            shape=(NUM_SHARE_Q_HEADS, HEAD_DIM),
            strides=(stride_qh, stride_qd),
            offsets=(0, 0),
            block_shape=(BLOCK_SIZE_H, BLOCK_SIZE_D),
            order=(1, 0),
        )
        k_ptrs = tl.make_block_ptr(
            base=k_ptr + k_start * stride_kn + pid_kh * stride_kh,
            shape=(HEAD_DIM, k_len),
            strides=(stride_kd, stride_kn),
            offsets=(0, 0),
            block_shape=(BLOCK_SIZE_D, BLOCK_SIZE_K),
            order=(0, 1),
        )
        v_ptrs = tl.make_block_ptr(
            base=v_ptr + k_start * stride_vn + pid_kh * stride_vh,
            shape=(k_len, HEAD_DIM),
            strides=(stride_vn, stride_vd),
            offsets=(0, 0),
            block_shape=(BLOCK_SIZE_K, BLOCK_SIZE_D),
            order=(1, 0),
        )

        # load q
        q = tl.load(q_ptrs, boundary_check=(0, 1), padding_option="zero")

        # init statistics
        off_h = tl.arange(0, BLOCK_SIZE_H)
        off_k = tl.arange(0, BLOCK_SIZE_K)
        m_i = tl.full((BLOCK_SIZE_H,), float("-inf"), dtype=tl.float32)
        lse_i = tl.full((BLOCK_SIZE_H,), float("-inf"), dtype=tl.float32)
        acc_o = tl.full((BLOCK_SIZE_H, BLOCK_SIZE_D), 0, dtype=tl.float32)

        # sparse attention
        for i in range(real_topk):
            # get current block start index
            t_ptr_j2 = (
                t_ptr
                + (q_start + pid_q_j) * stride_tn
                + pid_kh * stride_th
                + i * stride_tk
            )
            c = tl.load(t_ptr_j2).to(tl.int32) * BLOCK_SIZE_K
            # load k
            k = tl.load(
                tl.advance(k_ptrs, (0, c)), boundary_check=(1, 0), padding_option="zero"
            )
            # compute qk
            qk = tl.zeros((BLOCK_SIZE_H, BLOCK_SIZE_K), dtype=tl.float32)
            qk += tl.where((pid_q_j >= c + off_k)[None, :], 0, float("-inf"))
            qk += tl.dot(q, k) * qk_scale
            # compute m_ij and l_ij
            m_ij = tl.maximum(m_i, tl.max(qk, axis=1))
            p = tl.exp2(qk - m_ij[:, None])
            l_ij = tl.sum(p, axis=1)
            # scale acc_o
            acc_o_scale = tl.exp2(m_i - m_ij)
            acc_o = acc_o * acc_o_scale[:, None]
            # load v and update acc_o
            v = tl.load(
                tl.advance(v_ptrs, (c, 0)), boundary_check=(0, 1), padding_option="zero"
            )
            p = p.to(v.dtype)
            acc_o += tl.dot(p, v)
            # update statistics
            m_i = m_ij
            lse_i = m_ij + tl.math.log2(tl.exp2(lse_i - m_ij) + l_ij)

        # final scale
        acc_o = acc_o * tl.exp2(m_i - lse_i)[:, None]

        # save output
        o_ptrs = tl.make_block_ptr(
            base=o_ptr + (q_start + pid_q_j) * stride_on + pid_h * stride_oh,
            shape=(NUM_SHARE_Q_HEADS, HEAD_DIM),
            strides=(stride_oh, stride_od),
            offsets=(0, 0),
            block_shape=(BLOCK_SIZE_H, BLOCK_SIZE_D),
            order=(1, 0),
        )
        tl.store(o_ptrs, acc_o.to(o_ptr.dtype.element_ty), boundary_check=(0, 1))

        # save lse
        lse_ptrs = (
            lse_ptr + (q_start + pid_q_j) * stride_ln + (pid_h + off_h) * stride_lh
        )
        tl.store(lse_ptrs, lse_i, mask=off_h < NUM_SHARE_Q_HEADS)