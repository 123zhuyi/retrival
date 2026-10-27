@triton.jit
def _fwd_kernel(
    Q,
    K,
    V,
    sm_scale,
    Q_start_loc,
    Q_seqlens,
    KV_seqlens,
    Block_offsets,
    Out,
    stride_qbs: tl.constexpr,
    stride_qh: tl.constexpr,
    stride_qd: tl.constexpr,
    stride_kp: tl.constexpr,
    stride_kbs: tl.constexpr,
    stride_kh: tl.constexpr,
    stride_kd: tl.constexpr,
    stride_vp: tl.constexpr,
    stride_vbs: tl.constexpr,
    stride_vh: tl.constexpr,
    stride_vd: tl.constexpr,
    stride_obs: tl.constexpr,
    stride_oh: tl.constexpr,
    stride_od: tl.constexpr,
    stride_boffb,
    kv_group_num: tl.constexpr,
    window_size: tl.constexpr,
    head_size: tl.constexpr,
    head_size_v: tl.constexpr,
    logit_softcapping: tl.constexpr,
    BLOCK_M: tl.constexpr,
    BLOCK_DMODEL: tl.constexpr,
    BLOCK_DV: tl.constexpr,
    BLOCK_N: tl.constexpr,
    BLOCK_DMODEL1: tl.constexpr,
):
    """paged attention kernel."""
    cur_batch = tl.program_id(2)
    cur_head = tl.program_id(1)
    start_m = tl.program_id(0)

    cur_kv_head = cur_head // kv_group_num

    q_seqlen = tl.load(Q_seqlens + cur_batch)
    kv_seqlen = tl.load(KV_seqlens + cur_batch)
    q_start_loc = tl.load(Q_start_loc + cur_batch)
    history_len = kv_seqlen - q_seqlen

    block_start_loc = BLOCK_M * start_m
    if block_start_loc >= q_seqlen:
        return

    # initialize offsets
    offs_n = tl.arange(0, BLOCK_N)
    offs_d = tl.arange(0, BLOCK_DMODEL)
    offs_dv = tl.arange(0, BLOCK_DV)
    mask_d = offs_d < head_size
    mask_dv = offs_dv < head_size_v
    offs_m = start_m * BLOCK_M + tl.arange(0, BLOCK_M)
    off_q = (
        (q_start_loc + offs_m[:, None]) * stride_qbs
        + cur_head * stride_qh
        + offs_d[None, :] * stride_qd
    )
    off_k = (
        cur_kv_head * stride_kh
        + offs_n[:, None] * stride_kbs
        + offs_d[None, :] * stride_kd
    )
    off_v = (
        cur_kv_head * stride_vh
        + offs_n[:, None] * stride_vbs
        + offs_dv[None, :] * stride_vd
    )

    q = tl.load(
        Q + off_q, mask=(offs_m[:, None] < q_seqlen) & mask_d[None, :], other=0.0
    )

    k_ptrs = K + off_k
    v_ptrs = V + off_v

    if BLOCK_DMODEL1 != 0:
        offs_d1 = BLOCK_DMODEL + tl.arange(0, BLOCK_DMODEL1)
        mask_d1 = offs_d1 < head_size
        off_q1 = (
            (q_start_loc + offs_m[:, None]) * stride_qbs
            + cur_head * stride_qh
            + offs_d1[None, :] * stride_qd
        )
        q1 = tl.load(Q + off_q1, mask=(offs_m[:, None] < q_seqlen) & mask_d1, other=0.0)
        off_k1 = (
            cur_kv_head * stride_kh
            + offs_n[:, None] * stride_kbs
            + offs_d1[None, :] * stride_kd
        )
        k1_ptrs = K + off_k1

    block_offset_ptrs = Block_offsets + cur_batch * stride_boffb

    # initialize pointer to m and l
    m_i = tl.zeros([BLOCK_M], dtype=tl.float32) - float("inf")
    l_i = tl.zeros([BLOCK_M], dtype=tl.float32)
    acc = tl.zeros([BLOCK_M, BLOCK_DV], dtype=tl.float32)

    kv_start_loc = 0
    if window_size > 0:
        start_block_id = tl.maximum(history_len - window_size, 0) // BLOCK_N
        kv_min_loc = tl.maximum(history_len + offs_m - window_size, 0)
        kv_start_loc = start_block_id * BLOCK_N
        block_offset_ptrs += start_block_id

    for start_n in range(kv_start_loc, kv_seqlen, BLOCK_N):
        b_offset = tl.load(block_offset_ptrs)
        block_offset_ptrs += 1

        # -- compute qk ----
        k = tl.load(k_ptrs + b_offset * stride_kp).trans()
        if BLOCK_DMODEL1 != 0:
            k1 = tl.load(k1_ptrs + b_offset * stride_kp).trans()

        v = tl.load(v_ptrs + b_offset * stride_vp)

        qk = tl.zeros([BLOCK_M, BLOCK_N], dtype=tl.float32)
        qk += tl.dot(q, k)
        if BLOCK_DMODEL1 != 0:
            qk += tl.dot(q1, k1)
        qk *= sm_scale
        if logit_softcapping > 0.0:
            qk = qk / logit_softcapping
            qk = tanh(qk)
            qk = qk * logit_softcapping
        # NOTE: inf - inf = nan, and nan will leads to error
        history_len_fp = history_len.to(tl.float32)
        offs_m_fp = offs_m.to(tl.float32)
        offs_n_fp = offs_n.to(tl.float32)
        start_n_fp = start_n.to(tl.float32)
        qk_mask = (history_len_fp + offs_m_fp[:, None]) >= (
            start_n_fp + offs_n_fp[None, :]
        )
        if window_size > 0:
            kv_min_loc_fp = kv_min_loc.to(tl.float32)
            qk_mask = qk_mask and (
                (start_n_fp + offs_n_fp[None, :]) >= kv_min_loc_fp[:, None]
            )
        qk = tl.where(
            qk_mask,
            qk,
            float(-1e30),
        )

        # -- compute p, m_i and l_i
        m_i_new = tl.maximum(m_i, tl.max(qk, 1))
        p = fast_expf(qk - m_i_new[:, None])
        alpha = fast_expf(m_i - m_i_new)
        l_i_new = alpha * l_i + tl.sum(p, 1)
        # -- update output accumulator --
        # scale acc
        acc = acc * alpha[:, None]

        # update acc
        p, v = _convert_pv(p, v)
        acc += tl.dot(p, v)
        # update m_i and l_i
        l_i = l_i_new
        m_i = m_i_new

    acc = fast_dividef(acc, l_i[:, None])
    # initialize pointers to output
    off_o = (
        (q_start_loc + offs_m[:, None]) * stride_obs
        + cur_head * stride_oh
        + offs_dv[None, :] * stride_od
    )
    out_ptrs = Out + off_o
    tl.store(out_ptrs, acc, mask=(offs_m[:, None] < q_seqlen) & mask_dv[None, :])