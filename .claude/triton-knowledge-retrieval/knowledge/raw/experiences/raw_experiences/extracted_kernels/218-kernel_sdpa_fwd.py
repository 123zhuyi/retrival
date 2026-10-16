@triton.autotune(
    configs=[
        triton.Config({"multibuffer": True, "BLOCK_R": 128, "BLOCK_C": 128}),
        triton.Config({"multibuffer": True, "BLOCK_R": 64, "BLOCK_C": 128}),
        triton.Config({"multibuffer": True, "BLOCK_R": 128, "BLOCK_C": 64}),
        triton.Config({"multibuffer": False, "BLOCK_R": 128, "BLOCK_C": 256}),
        triton.Config({"multibuffer": False, "BLOCK_R": 256, "BLOCK_C": 128}),
        triton.Config({"multibuffer": False, "BLOCK_R": 128, "BLOCK_C": 128}),
    ],
    key=["N", "S", "H"],
)
@triton.jit
def kernel_sdpa_fwd(
    q,
    k,
    v,
    o,
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
    for task_id in range(pid, B * N * num_r, tl.num_programs(axis=0)):
        idx_b = task_id // (N * num_r)
        idx_n = task_id // num_r % N
        idx_r = task_id % num_r
        idx_h = tl.arange(0, H)

        ptr_q = (
            q
            + idx_b * STRIDE_Q_B
            + idx_n * STRIDE_Q_N
            + (idx_r * BLOCK_R + tl.arange(0, BLOCK_R))[:, None] * STRIDE_Q_S
            + idx_h[None, :] * STRIDE_Q_H
        )
        ptr_o = (
            o
            + idx_b * STRIDE_Q_B
            + idx_n * STRIDE_Q_N
            + (idx_r * BLOCK_R + tl.arange(0, BLOCK_R))[:, None] * STRIDE_Q_S
            + idx_h[None, :] * STRIDE_Q_H
        )
        ptr_lse = (
            lse + idx_b * STRIDE_D_B + idx_n * STRIDE_D_N + (idx_r * BLOCK_R + tl.arange(0, BLOCK_R))[:] * STRIDE_D_S
        )

        mask_q = (idx_r * BLOCK_R + tl.arange(0, BLOCK_R))[:, None] < S
        mask_lse = (idx_r * BLOCK_R + tl.arange(0, BLOCK_R))[:] < S

        block_q = tl.load(ptr_q, mask=mask_q, other=0.0)
        block_o = tl.full([BLOCK_R, H], 0.0, dtype=HIGH_TYPE)
        block_l = tl.full([BLOCK_R], 0.0, dtype=HIGH_TYPE)
        block_m = tl.full([BLOCK_R], -1e6, dtype=HIGH_TYPE)
        for idx_c in range(0, num_c):
            ptr_k = (
                k
                + idx_b * STRIDE_K_B
                + (idx_n // group_size) * STRIDE_K_N
                + (idx_c * BLOCK_C + tl.arange(0, BLOCK_C))[:, None] * STRIDE_K_S
                + idx_h[None, :] * STRIDE_K_H
            )
            ptr_v = (
                v
                + idx_b * STRIDE_V_B
                + (idx_n // group_size) * STRIDE_V_N
                + (idx_c * BLOCK_C + tl.arange(0, BLOCK_C))[:, None] * STRIDE_V_S
                + idx_h[None, :] * STRIDE_V_H
            )
            ptr_mask = (
                mask
                + (idx_r * BLOCK_R + tl.arange(0, BLOCK_R))[:, None] * S
                + (idx_c * BLOCK_C + tl.arange(0, BLOCK_C))[None, :]
            )

            mask_kv = (idx_c * BLOCK_C + tl.arange(0, BLOCK_C))[:, None] < S
            mask_mask = ((idx_r * BLOCK_R + tl.arange(0, BLOCK_R))[:, None] < S) & (
                (idx_c * BLOCK_C + tl.arange(0, BLOCK_C))[None, :] < S
            )

            block_mask = tl.load(ptr_mask, mask=mask_mask, other=False)
            block_k = tl.load(ptr_k, mask=mask_kv, other=0.0)
            block_v = tl.load(ptr_v, mask=mask_kv, other=0.0)

            block_s = tl.dot(block_q, tl.trans(block_k)) * scale
            block_s -= (1.0 - block_mask.to(HIGH_TYPE)) * 1e6
            block_m_1 = tl.maximum(block_m, tl.max(block_s, axis=1))
            block_s = tl.exp(block_s - block_m_1[:, None])
            block_l_1 = tl.exp(block_m - block_m_1) * block_l + tl.sum(block_s, axis=1)
            block_o = tl.exp(block_m - block_m_1)[:, None] * block_o + tl.dot(block_s.to(LOW_TYPE), block_v).to(
                HIGH_TYPE
            )
            block_m = block_m_1
            block_l = block_l_1

        block_o = block_o / block_l[:, None]
        block_lse = tl.log(block_l) + block_m
        tl.store(ptr_o, block_o.to(LOW_TYPE), mask=mask_q)
        tl.store(ptr_lse, block_lse, mask=mask_lse)