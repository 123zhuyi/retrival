@triton.autotune(
    configs=[
        triton.Config({"multibuffer": True, "BLOCK_R": 128, "BLOCK_C": 64}),
        triton.Config({"multibuffer": True, "BLOCK_R": 64, "BLOCK_C": 128}),
        triton.Config({"multibuffer": True, "BLOCK_R": 64, "BLOCK_C": 64}),
        triton.Config({"multibuffer": False, "BLOCK_R": 128, "BLOCK_C": 128}),
        triton.Config({"multibuffer": False, "BLOCK_R": 128, "BLOCK_C": 64}),
        triton.Config({"multibuffer": False, "BLOCK_R": 64, "BLOCK_C": 128}),
    ],
    key=["N", "S", "H"],
)
@triton.jit
def kernel_sdpa_bwd_kv(
    q,
    k,
    v,
    do,
    d,
    dk,
    dv,
    lse,
    mask,
    scale: tl.constexpr,
    num_group: tl.constexpr,
    B: tl.constexpr,
    N: tl.constexpr,
    S: tl.constexpr,
    H: tl.constexpr,
    STRIDE_Q_B: tl.constexpr,
    STRIDE_Q_N: tl.constexpr,
    STRIDE_Q_S: tl.constexpr,
    STRIDE_Q_H: tl.constexpr,
    STRIDE_K_B: tl.constexpr,
    STRIDE_K_N: tl.constexpr,
    STRIDE_K_S: tl.constexpr,
    STRIDE_K_H: tl.constexpr,
    STRIDE_V_B: tl.constexpr,
    STRIDE_V_N: tl.constexpr,
    STRIDE_V_S: tl.constexpr,
    STRIDE_V_H: tl.constexpr,
    STRIDE_D_B: tl.constexpr,
    STRIDE_D_N: tl.constexpr,
    STRIDE_D_S: tl.constexpr,
    BLOCK_R: tl.constexpr,
    BLOCK_C: tl.constexpr,
    LOW_TYPE: tl.constexpr = tl.bfloat16,
    HIGH_TYPE: tl.constexpr = tl.float32,
):
    pid = tl.program_id(axis=0)
    num_r = tl.cdiv(S, BLOCK_R)
    num_c = tl.cdiv(S, BLOCK_C)
    group_size = N // num_group
    for task_id in range(pid, B * num_group * num_c, tl.num_programs(axis=0)):
        idx_b = task_id // (num_group * num_c)
        idx_group = task_id // num_c % num_group
        idx_c = task_id % num_c
        idx_h = tl.arange(0, H)

        ptr_k = (
            k
            + idx_b * STRIDE_K_B
            + idx_group * STRIDE_K_N
            + (idx_c * BLOCK_C + tl.arange(0, BLOCK_C))[:, None] * STRIDE_K_S
            + idx_h[None, :] * STRIDE_K_H
        )
        ptr_v = (
            v
            + idx_b * STRIDE_V_B
            + idx_group * STRIDE_V_N
            + (idx_c * BLOCK_C + tl.arange(0, BLOCK_C))[:, None] * STRIDE_V_S
            + idx_h[None, :] * STRIDE_V_H
        )
        ptr_dk = (
            dk
            + idx_b * STRIDE_K_B
            + idx_group * STRIDE_K_N
            + (idx_c * BLOCK_C + tl.arange(0, BLOCK_C))[:, None] * STRIDE_K_S
            + idx_h[None, :] * STRIDE_K_H
        )
        ptr_dv = (
            dv
            + idx_b * STRIDE_V_B
            + idx_group * STRIDE_V_N
            + (idx_c * BLOCK_C + tl.arange(0, BLOCK_C))[:, None] * STRIDE_V_S
            + idx_h[None, :] * STRIDE_V_H
        )

        mask_kv = (idx_c * BLOCK_C + tl.arange(0, BLOCK_C))[:, None] < S
        block_k = tl.load(ptr_k, mask=mask_kv, other=0.0)
        block_v = tl.load(ptr_v, mask=mask_kv, other=0.0)
        block_dk = tl.full([BLOCK_C, H], 0.0, dtype=HIGH_TYPE)
        block_dv = tl.full([BLOCK_C, H], 0.0, dtype=HIGH_TYPE)

        for idx_ingroup in range(group_size):
            idx_n = idx_group * group_size + idx_ingroup
            for idx_r in range(0, num_r):
                ptr_q = (
                    q
                    + idx_b * STRIDE_Q_B
                    + idx_n * STRIDE_Q_N
                    + (idx_r * BLOCK_R + tl.arange(0, BLOCK_R))[:, None] * STRIDE_Q_S
                    + idx_h[None, :] * STRIDE_Q_H
                )
                ptr_do = (
                    do
                    + idx_b * STRIDE_Q_B
                    + idx_n * STRIDE_Q_N
                    + (idx_r * BLOCK_R + tl.arange(0, BLOCK_R))[:, None] * STRIDE_Q_S
                    + idx_h[None, :] * STRIDE_Q_H
                )
                ptr_d = (
                    d
                    + idx_b * STRIDE_D_B
                    + idx_n * STRIDE_D_N
                    + (idx_r * BLOCK_R + tl.arange(0, BLOCK_R))[:] * STRIDE_D_S
                )
                ptr_lse = (
                    lse
                    + idx_b * STRIDE_D_B
                    + idx_n * STRIDE_D_N
                    + (idx_r * BLOCK_R + tl.arange(0, BLOCK_R))[:] * STRIDE_D_S
                )
                ptr_mask = (
                    mask
                    + (idx_r * BLOCK_R + tl.arange(0, BLOCK_R))[:, None] * S
                    + (idx_c * BLOCK_C + tl.arange(0, BLOCK_C))[None, :]
                )

                mask_q = (idx_r * BLOCK_R + tl.arange(0, BLOCK_R))[:, None] < S
                mask_d = (idx_r * BLOCK_R + tl.arange(0, BLOCK_R))[:] < S
                mask_mask = ((idx_r * BLOCK_R + tl.arange(0, BLOCK_R))[:, None] < S) & (
                    (idx_c * BLOCK_C + tl.arange(0, BLOCK_C))[None, :] < S
                )

                block_q = tl.load(ptr_q, mask=mask_q, other=0.0)
                block_do = tl.load(ptr_do, mask=mask_q, other=0.0)
                block_lse = tl.load(ptr_lse, mask=mask_d, other=0.0)
                block_d = tl.load(ptr_d, mask=mask_d, other=0.0)
                block_mask = tl.load(ptr_mask, mask=mask_mask, other=False)

                block_s = tl.dot(block_q, block_k.T) * scale
                block_s -= (1.0 - block_mask.to(HIGH_TYPE)) * 1e6
                block_p = tl.exp(block_s - block_lse[:, None])
                block_dv += tl.dot(block_p.to(LOW_TYPE).T, block_do).to(HIGH_TYPE)
                block_dp = tl.dot(block_do, block_v.T).to(HIGH_TYPE)
                block_ds = block_p * (block_dp - block_d[:, None])
                block_dk += tl.dot(block_ds.to(LOW_TYPE).T, block_q).to(HIGH_TYPE) * scale

        tl.store(ptr_dk, block_dk.to(LOW_TYPE), mask=mask_kv)
        tl.store(ptr_dv, block_dv.to(LOW_TYPE), mask=mask_kv)