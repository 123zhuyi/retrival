@triton.jit
def paged_decode_kernel(
    q_ptr,
    k_cache_ptr,
    v_cache_ptr,
    o_ptr,
    seqlens_ptr,
    block_tables_ptr,
    BATCH_SIZE,
    NUM_Q_HEADS,
    NUM_KV_HEADS,
    GQA_INTERLEAVE,
    HEAD_DIM,
    NUM_TOTAL_BLOCKS,
    MAX_NUM_BLOCKS_PER_SEQ,
    stride_qb,
    stride_qh,
    stride_qd,
    stride_k_block,
    stride_k_head,
    stride_k_blksz,
    stride_k_dim,
    stride_v_block,
    stride_v_head,
    stride_v_blksz,
    stride_v_dim,
    stride_ob,
    stride_oh,
    stride_od,
    stride_bt_batch,
    stride_bt_block,
    sm_scale,
    BLOCK_SIZE_D: tl.constexpr,
    BLOCK_SIZE_N: tl.constexpr,
):
    pid_b = tl.program_id(0)
    pid_h = tl.program_id(1)

    NUM_SHARE_Q_HEADS = NUM_Q_HEADS // NUM_KV_HEADS
    if GQA_INTERLEAVE:
        pid_kh = pid_h % NUM_KV_HEADS
    else:
        pid_kh = pid_h // NUM_SHARE_Q_HEADS

    kv_len = tl.load(seqlens_ptr + pid_b)

    num_logical_blocks = (kv_len + BLOCK_SIZE_N - 1) // BLOCK_SIZE_N

    q_offset = pid_b * stride_qb + pid_h * stride_qh

    offs_d = tl.arange(0, BLOCK_SIZE_D)
    q_ptrs = q_ptr + q_offset + offs_d * stride_qd
    q = tl.load(q_ptrs)

    m_i = -float("inf")
    l_i = 0.0
    acc_o = tl.zeros((BLOCK_SIZE_D,), dtype=tl.float32)

    for logical_block_idx in range(0, num_logical_blocks):
        bt_offset = pid_b * stride_bt_batch + logical_block_idx * stride_bt_block
        physical_block_id = tl.load(block_tables_ptr + bt_offset)

        k_block_ptr = tl.make_block_ptr(
            base=k_cache_ptr + pid_kh * stride_k_head,
            shape=(NUM_TOTAL_BLOCKS, BLOCK_SIZE_N, HEAD_DIM),
            strides=(stride_k_block, stride_k_blksz, stride_k_dim),
            offsets=(physical_block_id, 0, 0),
            block_shape=(1, BLOCK_SIZE_N, BLOCK_SIZE_D),
            order=(2, 1, 0),
        )
        v_block_ptr = tl.make_block_ptr(
            base=v_cache_ptr + pid_kh * stride_v_head,
            shape=(NUM_TOTAL_BLOCKS, BLOCK_SIZE_N, HEAD_DIM),
            strides=(stride_v_block, stride_v_blksz, stride_v_dim),
            offsets=(physical_block_id, 0, 0),
            block_shape=(1, BLOCK_SIZE_N, BLOCK_SIZE_D),
            order=(2, 1, 0),
        )

        k = tl.load(k_block_ptr)
        v = tl.load(v_block_ptr)

        k = tl.reshape(k, (BLOCK_SIZE_N, BLOCK_SIZE_D))
        v = tl.reshape(v, (BLOCK_SIZE_N, BLOCK_SIZE_D))

        qk = tl.sum(q[None, :] * k, axis=1)

        current_logical_offset = logical_block_idx * BLOCK_SIZE_N + tl.arange(0, BLOCK_SIZE_N)
        mask = current_logical_offset < kv_len

        qk = tl.where(mask, qk, -float("inf"))
        qk *= sm_scale

        m_j = tl.max(qk, axis=0)
        m_new = tl.maximum(m_i, m_j)

        p = tl.exp(qk - m_new)
        l_j = tl.sum(p, axis=0)

        alpha = tl.exp(m_i - m_new)
        beta = tl.exp(m_j - m_new)

        l_new = alpha * l_i + l_j

        acc_o = acc_o * alpha

        p = p.to(v.dtype)

        acc_o += tl.sum(p[:, None] * v, axis=0)

        l_i = l_new
        m_i = m_new

    acc_o = acc_o / l_i

    o_offset = pid_b * stride_ob + pid_h * stride_oh
    o_ptrs = o_ptr + o_offset + offs_d * stride_od
    tl.store(o_ptrs, acc_o.to(o_ptr.dtype.element_ty))