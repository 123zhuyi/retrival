@triton.jit
def _hstu_attn_fwd_one_block(
    q,
    k_block_ptr,
    v_block_ptr,
    bias_block_ptr,
    alpha,
    MAX_SEQ_LEN,
    CAUSAL: tl.constexpr,
    HAS_BIAS: tl.constexpr,
    mask_block,
):
    k = tl.load(k_block_ptr)
    qk = tl.dot(q, tl.trans(k)) * alpha
    if HAS_BIAS:
        rel_attn_bias = tl.load(bias_block_ptr)
        qk = qk + rel_attn_bias
    silu = qk / (1.0 + tl.exp(-qk)) * (1.0 / MAX_SEQ_LEN)
    if CAUSAL:
        silu = tl.where(mask_block, silu, 0)
    v = tl.load(v_block_ptr)
    silu = silu.to(v.dtype)
    return tl.dot(silu, v)