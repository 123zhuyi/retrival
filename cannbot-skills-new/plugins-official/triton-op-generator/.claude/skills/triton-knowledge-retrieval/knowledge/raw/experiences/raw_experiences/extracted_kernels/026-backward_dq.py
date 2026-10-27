@triton.jit
def backward_dq(
    q_ptr,  # Q: n x qh x d
    k_ptr,  # K: n x kh x d
    v_ptr,  # V: n x kh x d
    t_ptr,  # topk_idx: kh x n x k
    lse_ptr,  # LSE: qh x n
    d_ptr,  # Delta: qh x n
    do_ptr,
    dq_ptr,
    # seqlens
    cu_seqlens_q,
    cu_seqlens_k,
    # shape
    NUM_KV_HEADS,
    NUM_SHARE_Q_HEADS,
    HEAD_DIM,
    TOPK,
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
    stride_lh,
    stride_ln,
    stride_dh,
    stride_dn,
    stride_don,
    stride_doh,
    stride_dod,
    stride_dqn,
    stride_dqh,
    stride_dqd,
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
    pid_q = tl.program_id(2) + q_block_offset
    pid_h = pid_kh * NUM_SHARE_Q_HEADS
    # get q k start and len after rmpad
    q_start = tl.load(cu_seqlens_q + pid_b)
    q_len = tl.load(cu_seqlens_q + pid_b + 1) - q_start
    k_start = tl.load(cu_seqlens_k + pid_b)
    k_len = tl.load(cu_seqlens_k + pid_b + 1) - k_start
    if pid_q * num_q_loop >= q_len:
        return
    real_q_loop = min(num_q_loop, q_len - pid_q * num_q_loop)
    for j in range(real_q_loop):
        pid_q_j = pid_q * num_q_loop + j
        # init topk idx pointer
        off_t = tl.arange(0, BLOCK_SIZE_T)
        t_ptr_j = t_ptr + (q_start + pid_q_j) * stride_tn + pid_kh * stride_th
        topk_idx = tl.load(t_ptr_j + off_t * stride_tk, mask=off_t < TOPK, other=-1)
        real_topk = tl.sum(
            tl.where((topk_idx >= 0) & (topk_idx <= pid_q_j // BLOCK_SIZE_K), 1, 0),
            axis=0,
        )
        # init pointers
        q_ptrs = tl.make_block_ptr(
            base=q_ptr + (q_start + pid_q_j) * stride_qn + pid_h * stride_qh,
            shape=(NUM_SHARE_Q_HEADS, HEAD_DIM),
            strides=(stride_qh, stride_qd),
            offsets=(0, 0),
            block_shape=(BLOCK_SIZE_H, BLOCK_SIZE_D),
            order=(1, 0),
        )
        dq_ptrs = tl.make_block_ptr(
            base=dq_ptr + (q_start + pid_q_j) * stride_dqn + pid_h * stride_dqh,
            shape=(NUM_SHARE_Q_HEADS, HEAD_DIM),
            strides=(stride_dqh, stride_dqd),
            offsets=(0, 0),
            block_shape=(BLOCK_SIZE_H, BLOCK_SIZE_D),
            order=(1, 0),
        )
        k_ptrs = tl.make_block_ptr(
            base=k_ptr + k_start * stride_kn + pid_kh * stride_kh,
            shape=(k_len, HEAD_DIM),
            strides=(stride_kn, stride_kd),
            offsets=(0, 0),
            block_shape=(BLOCK_SIZE_K, BLOCK_SIZE_D),
            order=(1, 0),
        )
        v_ptrs = tl.make_block_ptr(
            base=v_ptr + k_start * stride_vn + pid_kh * stride_vh,
            shape=(HEAD_DIM, k_len),
            strides=(stride_vd, stride_vn),
            offsets=(0, 0),
            block_shape=(BLOCK_SIZE_D, BLOCK_SIZE_K),
            order=(0, 1),
        )
        do_ptrs = tl.make_block_ptr(
            base=do_ptr + (q_start + pid_q_j) * stride_don + pid_h * stride_doh,
            shape=(NUM_SHARE_Q_HEADS, HEAD_DIM),
            strides=(stride_doh, stride_dod),
            offsets=(0, 0),
            block_shape=(BLOCK_SIZE_H, BLOCK_SIZE_D),
            order=(1, 0),
        )
        d_ptrs = tl.make_block_ptr(
            base=d_ptr + (q_start + pid_q_j) * stride_dn + pid_h * stride_dh,
            shape=(NUM_SHARE_Q_HEADS, 1),
            strides=(stride_dh, stride_dn),
            offsets=(0, 0),
            block_shape=(BLOCK_SIZE_H, 1),
            order=(1, 0),
        )
        lse_ptrs = tl.make_block_ptr(
            base=lse_ptr + (q_start + pid_q_j) * stride_ln + pid_h * stride_lh,
            shape=(NUM_SHARE_Q_HEADS, 1),
            strides=(stride_lh, stride_ln),
            offsets=(0, 0),
            block_shape=(BLOCK_SIZE_H, 1),
            order=(1, 0),
        )
        # offsets
        off_k = tl.arange(0, BLOCK_SIZE_K)
        # load q, do, lse, delta, and keep in SRAM
        q = tl.load(q_ptrs, boundary_check=(1, 0), padding_option="zero")
        do = tl.load(do_ptrs, boundary_check=(0, 1), padding_option="zero")
        lse = tl.load(lse_ptrs, boundary_check=(0, 1), padding_option="zero")
        d = tl.load(d_ptrs, boundary_check=(0, 1), padding_option="zero")
        # init dq
        dq = tl.zeros((BLOCK_SIZE_H, BLOCK_SIZE_D), dtype=tl.float32)
        # sparse
        for i in range(real_topk):
            # get current block start index
            t_ptr_j2 = t_ptr_j + stride_tk * i
            c = tl.load(t_ptr_j2).to(tl.int32) * BLOCK_SIZE_K
            # load
            k = tl.load(
                tl.advance(k_ptrs, (c, 0)), boundary_check=(1, 0), padding_option="zero"
            )
            v = tl.load(
                tl.advance(v_ptrs, (0, c)), boundary_check=(0, 1), padding_option="zero"
            )
            # compute qk
            qk = tl.zeros((BLOCK_SIZE_H, BLOCK_SIZE_K), dtype=tl.float32)
            qk += tl.where((pid_q_j >= c + off_k)[None, :], 0, float("-inf"))
            # [BLOCK_SIZE_H, HEAD_DIM] @ [BLOCK_SIZE_K, HEAD_DIM].T -> [BLOCK_SIZE_H, BLOCK_SIZE_K]
            qk += tl.dot(q, tl.trans(k)) * qk_scale
            # compute p, ds
            p = tl.exp2(qk - lse)
            dp = tl.dot(do, v)
            ds = sm_scale * p * (dp - d)
            # cast dtype
            ds = ds.to(q.dtype)
            # update dq
            dq += tl.dot(ds, k)
        # save dq
        tl.store(dq_ptrs, dq.to(dq_ptr.dtype.element_ty), boundary_check=(0, 1))