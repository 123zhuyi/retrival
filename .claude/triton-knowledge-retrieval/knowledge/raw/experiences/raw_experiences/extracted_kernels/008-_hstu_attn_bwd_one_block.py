@triton.jit
def _hstu_attn_bwd_one_block(  # noqa C901
    start_m,
    offs_n,
    offs_m,
    q_ptrs,
    dq_ptrs,
    mask_n,
    do_ptrs,
    dk,
    dv,
    k,
    v,
    pos_offs_n,
    seq_len,
    max_ids,
    stride_qm,
    stride_dom,
    stride_dqm,
    alpha,
    MAX_SEQ_LEN,
    CAUSAL: tl.constexpr,
    HAS_BIAS: tl.constexpr,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
    bias_block_ptr,
):
    pos_offs_m = offs_m + start_m
    mask_m = pos_offs_m < seq_len
    # recompute qk and silu
    q = tl.load(
        q_ptrs + start_m * stride_qm,
        mask=mask_m[:, None],
        other=0.0,
    )
    q_trans = tl.trans(q)
    qk_trans = tl.dot(k, q_trans) * alpha
    if HAS_BIAS:
        rel_attn_bias = tl.load(bias_block_ptr)
        qk_trans = qk_trans + tl.trans(rel_attn_bias)
    sig_trans = 1.0 / (1.0 + tl.exp(-qk_trans))
    silu_trans = qk_trans * sig_trans * (1.0 / MAX_SEQ_LEN)
    if CAUSAL:
        invalid_mask_trans = pos_offs_m[None, :] == offs_n[:, None]
        pos_offs_m_minus_n = pos_offs_m[None, :] - pos_offs_n[:, None]
        invalid_mask_trans = invalid_mask_trans | (pos_offs_m_minus_n > 0)
        silu_trans = tl.where(invalid_mask_trans, silu_trans, 0)
    silu_trans = silu_trans.to(k.dtype)
    # compute dv
    do = tl.load(
        do_ptrs + start_m * stride_dom,
        mask=mask_m[:, None],
        other=0.0,
    )
    dv += tl.dot(silu_trans, do)
    # compute dk and dq  (dqk = do * v^T  dk = dqk^T * q  dq = dqk * k)
    dqk_trans = tl.dot(v, tl.trans(do))
    dqk_trans = dqk_trans * sig_trans * (1 + qk_trans * (1 - sig_trans)) * (1.0 / MAX_SEQ_LEN)
    if CAUSAL:
        dqk_trans = tl.where(invalid_mask_trans, dqk_trans, 0)
    dqk_trans = dqk_trans.to(k.dtype)
    dq = tl.load(
        dq_ptrs + start_m * stride_dqm,
        mask=mask_m[:, None],
        other=0.0,
    )
    dq += tl.dot(tl.trans(dqk_trans), k) * alpha
    tl.store(
        dq_ptrs + start_m * stride_dqm,
        dq,
        mask=mask_m[:, None],
    )
    # Note: the factor `alpha` is delayed until the end of the function to reduce the cost
    dk += tl.dot(dqk_trans, q)
    return dk, dv