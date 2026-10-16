@triton.jit
def _sdpa_infer_inner(
    acc_ptr,
    l_i,
    m_i,
    q,  # Accumulator, local l, local m, query vector
    K_block_ptr,
    V_block_ptr,  # Key and value block pointers for current stage
    mask_base_ptr,
    start_m,
    qk_scale,  # Starting position of current query block, qk scale factor
    BLOCK_M: tl.constexpr,
    HEAD_DIM: tl.constexpr,
    BLOCK_N: tl.constexpr,  # Block size constants
    offs_m: tl.constexpr,
    offs_n: tl.constexpr,  # Current stage flag, m and n offset indices
    SEQ: tl.constexpr,
    fp8_v: tl.constexpr,
):
    # Iterate over all k, v blocks in the current stage and accumulate the output
    for start_n in range(0, SEQ, BLOCK_N):  # Process BLOCK_N columns at a time
        start_n = tl.multiple_of(start_n, BLOCK_N)  # Align column start position
        mask_ptr = (
            mask_base_ptr
            + start_m * BLOCK_M * SEQ
            + start_n
            + tl.arange(0, BLOCK_M)[:, None] * SEQ
            + tl.arange(0, BLOCK_N)[None, :]
        )
        # -- Compute qk ----
        k = tl.load(K_block_ptr)
        # Modify K
        trans_k = tl.trans(k)
        qk = tl.dot(q, trans_k)

        # NOTE(zhangjihang): tl.where will introduce ub overflow
        qk = qk * qk_scale
        mask = tl.load(mask_ptr)

        # qk += (1 - mask.to(tl.float32)) * (-1e6)
        # qk = tl.where(mask, qk, float("-inf"))
        qk = tl.where(mask, qk, -1e6)

        m_ij = tl.maximum(m_i, tl.max(qk, 1))  # Scaled max
        qk = qk - m_ij[:, None]  # Stabilize

        # Softmax weights p = exp(qk)
        p = tl.math.exp(qk)

        p_cast = p.to(k.dtype)

        v = tl.load(V_block_ptr)  # Load corresponding V block
        l_ij = tl.sum(p, 1)  # Softmax denominator (sum of each row)
        # -- Update m_i and l_i
        alpha = tl.math.exp(m_i - m_ij)  # Update factor: exp difference between old and new max
        l_i = l_i * alpha + l_ij  # Update softmax denominator
        # -- Update output accumulator --
        acc_ptr = acc_ptr * alpha[:, None]
        acc_ptr = tl.dot(p_cast, v, acc_ptr)
        tl.compile_hint(acc_ptr, "tile_cube_loop", 2)

        m_i = m_ij  # Update current block max
        # Advance V and K block pointers to next BLOCK_N range
        V_block_ptr = tl.advance(V_block_ptr, (BLOCK_N, 0))
        K_block_ptr = tl.advance(K_block_ptr, (BLOCK_N, 0))

    # NOTE(zhangjihang): for training
    # Return accumulated output acc_ptr, softmax denominator l_i, and max value m_i
    return acc_ptr, l_i, m_i