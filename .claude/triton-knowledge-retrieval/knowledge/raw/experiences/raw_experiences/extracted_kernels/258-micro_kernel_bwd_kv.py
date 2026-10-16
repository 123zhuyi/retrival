@triton.jit
def micro_kernel_bwd_kv(
    q,
    block_k,
    block_v,
    do,
    d,
    block_dk,
    block_dv,
    lse,
    scale,
    offset_r,
    block_mask,
    idx_b,
    idx_n,
    idx_h,
    S: tl.constexpr,
    STRIDE_Q_B: tl.constexpr,
    STRIDE_Q_N: tl.constexpr,
    STRIDE_Q_S: tl.constexpr,
    STRIDE_Q_H: tl.constexpr,
    STRIDE_D_B: tl.constexpr,
    STRIDE_D_N: tl.constexpr,
    STRIDE_D_S: tl.constexpr,
    BLOCK_R: tl.constexpr,
    LOW_TYPE,
    HIGH_TYPE,
):
    ptr_q = (
        q
        + idx_b * STRIDE_Q_B
        + idx_n * STRIDE_Q_N
        + (offset_r + tl.arange(0, BLOCK_R))[:, None] * STRIDE_Q_S
        + idx_h[None, :] * STRIDE_Q_H
    )
    ptr_do = (
        do
        + idx_b * STRIDE_Q_B
        + idx_n * STRIDE_Q_N
        + (offset_r + tl.arange(0, BLOCK_R))[:, None] * STRIDE_Q_S
        + idx_h[None, :] * STRIDE_Q_H
    )
    ptr_d = d + idx_b * STRIDE_D_B + idx_n * STRIDE_D_N + (offset_r + tl.arange(0, BLOCK_R))[:] * STRIDE_D_S
    ptr_lse = lse + idx_b * STRIDE_D_B + idx_n * STRIDE_D_N + (offset_r + tl.arange(0, BLOCK_R))[:] * STRIDE_D_S

    mask_q = (offset_r + tl.arange(0, BLOCK_R))[:, None] < S
    mask_d = (offset_r + tl.arange(0, BLOCK_R))[:] < S

    block_q = tl.load(ptr_q, mask=mask_q, other=0.0)
    block_s = tl.dot(block_q, block_k.T).to(HIGH_TYPE) * scale
    if block_mask is not None:
        block_s += block_mask
    block_lse = tl.load(ptr_lse, mask=mask_d, other=0.0)
    block_p = tl.exp(block_s - block_lse[:, None])
    block_do = tl.load(ptr_do, mask=mask_q, other=0.0)
    block_dv += tl.dot(block_p.to(LOW_TYPE).T, block_do).to(HIGH_TYPE)
    block_dp = tl.dot(block_do, block_v.T).to(HIGH_TYPE)
    block_d = tl.load(ptr_d, mask=mask_d, other=0.0)
    block_ds = block_p * (block_dp - block_d[:, None])
    block_dk += tl.dot(block_ds.to(LOW_TYPE).T, block_q).to(HIGH_TYPE) * scale

    return block_dk, block_dv