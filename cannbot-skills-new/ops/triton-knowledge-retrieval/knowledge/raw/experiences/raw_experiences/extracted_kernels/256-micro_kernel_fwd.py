@triton.jit
def micro_kernel_fwd(
    block_q,
    k,
    v,
    block_o,
    block_m,
    block_l,
    scale,
    offset_c,
    block_mask,
    idx_b,
    idx_n,
    idx_h,
    S: tl.constexpr,
    STRIDE_K_B: tl.constexpr,
    STRIDE_K_N: tl.constexpr,
    STRIDE_K_S: tl.constexpr,
    STRIDE_K_H: tl.constexpr,
    STRIDE_V_B: tl.constexpr,
    STRIDE_V_N: tl.constexpr,
    STRIDE_V_S: tl.constexpr,
    STRIDE_V_H: tl.constexpr,
    GROUP_SIZE: tl.constexpr,
    BLOCK_C: tl.constexpr,
    LOW_TYPE,
    HIGH_TYPE,
):
    ptr_k = (
        k
        + idx_b * STRIDE_K_B
        + (idx_n // GROUP_SIZE) * STRIDE_K_N
        + (offset_c + tl.arange(0, BLOCK_C))[:, None] * STRIDE_K_S
        + idx_h[None, :] * STRIDE_K_H
    )
    ptr_v = (
        v
        + idx_b * STRIDE_V_B
        + (idx_n // GROUP_SIZE) * STRIDE_V_N
        + (offset_c + tl.arange(0, BLOCK_C))[:, None] * STRIDE_V_S
        + idx_h[None, :] * STRIDE_V_H
    )

    mask_kv = (offset_c + tl.arange(0, BLOCK_C))[:, None] < S
    block_k = tl.load(ptr_k, mask=mask_kv, other=0.0)
    block_s = tl.dot(block_q, block_k.T) * scale
    if block_mask is not None:
        block_s += block_mask
    block_m_1 = tl.maximum(block_m, tl.max(block_s, axis=1))
    block_s = tl.exp(block_s - block_m_1[:, None])
    block_l_1 = tl.exp(block_m - block_m_1) * block_l + tl.sum(block_s, axis=1)

    block_v = tl.load(ptr_v, mask=mask_kv, other=0.0)
    block_o = tl.exp(block_m - block_m_1)[:, None].to(LOW_TYPE) * block_o + tl.dot(block_s.to(LOW_TYPE), block_v).to(
        LOW_TYPE
    )
    return block_o, block_m_1, block_l_1