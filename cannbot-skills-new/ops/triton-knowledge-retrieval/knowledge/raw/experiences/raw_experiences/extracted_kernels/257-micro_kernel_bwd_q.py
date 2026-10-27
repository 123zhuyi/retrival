@triton.jit
def micro_kernel_bwd_q(
    block_q,
    k,
    v,
    block_do,
    block_d,
    block_dq,
    block_lse,
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
    block_s = tl.dot(block_q, block_k.T).to(HIGH_TYPE) * scale
    if block_mask is not None:
        block_s += block_mask
    block_p = tl.exp(block_s - block_lse[:, None])
    block_v = tl.load(ptr_v, mask=mask_kv, other=0.0)
    block_dp = tl.dot(block_do, block_v.T).to(HIGH_TYPE)
    block_ds = block_p * (block_dp - block_d[:, None])
    block_dq += tl.dot(block_ds.to(LOW_TYPE), block_k).to(HIGH_TYPE) * scale
    return block_dq