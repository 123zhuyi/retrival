@triton.autotune(
    configs=[
        triton.Config(
            {
                "BLOCK_R": 64,
                "BLOCK_C": 256,
                "multibuffer": True,
                "unit_flag": True,
                "set_workspace_multibuffer": 2,
                "enable_hivm_auto_cv_balance": True,
                "tile_mix_vector_loop": 2,
                "tile_mix_cube_loop": 2,
            }
        )
    ],
    key=["N", "S", "H"],
)
@triton.jit
def kernel_sda_fwd_up(
    q,
    k,
    v,
    o,
    fp32o,
    lse,
    scale: tl.constexpr,
    NUM_GROUP: tl.constexpr,
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
    BLOCK_SIZE: tl.constexpr,
    LOW_TYPE: tl.constexpr = tl.bfloat16,
    HIGH_TYPE: tl.constexpr = tl.float32,
):
    pid = tl.program_id(axis=0)
    SEQ_LEN = S // 2
    num_r = tl.cdiv(SEQ_LEN, BLOCK_R)
    GROUP_SIZE = N // NUM_GROUP
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
        ptr_fp32o = (
            fp32o
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

        offs_r_local = tl.arange(0, BLOCK_R)[:, None]
        offs_c_local = tl.arange(0, BLOCK_R)[None, :]
        chunk_idx_r = offs_r_local // BLOCK_SIZE
        chunk_idx_c = offs_c_local // BLOCK_SIZE
        block_mask_bool = chunk_idx_r == chunk_idx_c
        block_mask = (block_mask_bool.to(HIGH_TYPE) - 1.0) * 1e6
        block_o, block_m, block_l = micro_kernel_fwd(
            block_q,
            k,
            v,
            block_o,
            block_m,
            block_l,
            scale,
            idx_r * BLOCK_R,
            block_mask,
            idx_b,
            idx_n,
            idx_h,
            S,
            STRIDE_K_B,
            STRIDE_K_N,
            STRIDE_K_S,
            STRIDE_K_H,
            STRIDE_V_B,
            STRIDE_V_N,
            STRIDE_V_S,
            STRIDE_V_H,
            GROUP_SIZE,
            BLOCK_R,
            LOW_TYPE,
            HIGH_TYPE,
        )

        offs_r_local = tl.arange(0, BLOCK_R)[:, None]
        offs_c_local = tl.arange(0, BLOCK_R)[None, :]
        chunk_idx_r = offs_r_local // BLOCK_SIZE
        chunk_idx_c = offs_c_local // BLOCK_SIZE
        block_mask_bool = chunk_idx_r > chunk_idx_c
        block_mask = (block_mask_bool.to(LOW_TYPE) - 1.0) * 1e6
        block_o, block_m, block_l = micro_kernel_fwd(
            block_q,
            k,
            v,
            block_o,
            block_m,
            block_l,
            scale,
            SEQ_LEN + idx_r * BLOCK_R,
            block_mask,
            idx_b,
            idx_n,
            idx_h,
            S,
            STRIDE_K_B,
            STRIDE_K_N,
            STRIDE_K_S,
            STRIDE_K_H,
            STRIDE_V_B,
            STRIDE_V_N,
            STRIDE_V_S,
            STRIDE_V_H,
            GROUP_SIZE,
            BLOCK_R,
            LOW_TYPE,
            HIGH_TYPE,
        )

        for idx_tile_r in range(idx_r * BLOCK_R // BLOCK_C * BLOCK_C // BLOCK_R, idx_r):
            block_o, block_m, block_l = micro_kernel_fwd(
                block_q,
                k,
                v,
                block_o,
                block_m,
                block_l,
                scale,
                SEQ_LEN + idx_tile_r * BLOCK_R,
                None,
                idx_b,
                idx_n,
                idx_h,
                S,
                STRIDE_K_B,
                STRIDE_K_N,
                STRIDE_K_S,
                STRIDE_K_H,
                STRIDE_V_B,
                STRIDE_V_N,
                STRIDE_V_S,
                STRIDE_V_H,
                GROUP_SIZE,
                BLOCK_R,
                LOW_TYPE,
                HIGH_TYPE,
            )

        for idx_c in range(idx_r * BLOCK_R // BLOCK_C):
            block_o, block_m, block_l = micro_kernel_fwd(
                block_q,
                k,
                v,
                block_o,
                block_m,
                block_l,
                scale,
                SEQ_LEN + idx_c * BLOCK_C,
                None,
                idx_b,
                idx_n,
                idx_h,
                S,
                STRIDE_K_B,
                STRIDE_K_N,
                STRIDE_K_S,
                STRIDE_K_H,
                STRIDE_V_B,
                STRIDE_V_N,
                STRIDE_V_S,
                STRIDE_V_H,
                GROUP_SIZE,
                BLOCK_C,
                LOW_TYPE,
                HIGH_TYPE,
            )

        block_o = block_o / block_l[:, None]
        block_lse = tl.log(block_l) + block_m
        tl.store(ptr_o, block_o.to(LOW_TYPE), mask=mask_q)
        tl.store(ptr_fp32o, block_o, mask=mask_q)
        tl.store(ptr_lse, block_lse, mask=mask_lse)