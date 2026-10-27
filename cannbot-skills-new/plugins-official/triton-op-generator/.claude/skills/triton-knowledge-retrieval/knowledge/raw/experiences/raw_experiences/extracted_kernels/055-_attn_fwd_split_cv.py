@triton.jit
def _attn_fwd_split_cv(
    Q,
    K,
    V,
    Out,
    sm,
    WS1,
    WS2,
    WS3,
    WS4,
    stride_qz: tl.constexpr,
    stride_qh: tl.constexpr,
    stride_qm: tl.constexpr,
    stride_qk: tl.constexpr,
    stride_kz: tl.constexpr,
    stride_kh: tl.constexpr,
    stride_kn: tl.constexpr,
    stride_kk: tl.constexpr,
    stride_vz: tl.constexpr,
    stride_vh: tl.constexpr,
    stride_vn: tl.constexpr,
    stride_vk: tl.constexpr,
    stride_oz: tl.constexpr,
    stride_oh: tl.constexpr,
    stride_om: tl.constexpr,
    stride_on: tl.constexpr,
    w12_stride_nb: tl.constexpr,
    w12_stride_bm: tl.constexpr,
    w12_stride_bn: tl.constexpr,
    w34_stride_nb: tl.constexpr,
    w34_stride_bm: tl.constexpr,
    w34_stride_dm: tl.constexpr,
    Z: tl.constexpr,
    H: tl.constexpr,
    N_CTX: tl.constexpr,
    HEAD_DIM: tl.constexpr,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
    NUM_CORES: tl.constexpr,
):
    # Total number of blocks in sequence dimension (M)
    NUM_BLOCKS_M = N_CTX // BLOCK_M
    # Total tasks = number of sequence blocks × batch size (Z) × number of attention heads (H)
    NUM_BLOCKS = NUM_BLOCKS_M * Z * H
    BATCH_NUM_BLOCKS: tl.constexpr = NUM_BLOCKS // 2  # 2 tasks
    SUB_BLOCK_M: tl.constexpr = BLOCK_M // 4
    # Current M-dimension block index
    pid = tl.program_id(0)
    for block_idx in range(pid, BATCH_NUM_BLOCKS, NUM_CORES):
        block_idx_0 = block_idx * 2
        block_idx_1 = block_idx * 2 + 1
        # --- task 0
        (
            Q_ptr_0,
            K_ptr_0,
            V_ptr_0,
            O_ptr_0,
            WS1_0,
            WS1_L,
            WS2_0,
            WS2_L,
            WS3_0,
            WS3_L,
            WS4_L,
        ) = get_params(
            Q,
            K,
            V,
            Out,
            WS1,
            WS2,
            WS3,
            WS4,
            block_idx_0,
            NUM_BLOCKS_M,
            H,
            stride_qz,
            stride_qh,
            N_CTX,
            HEAD_DIM,
            stride_qm,
            stride_qk,
            stride_kn,
            stride_kk,
            stride_vn,
            stride_vk,
            stride_om,
            stride_on,
            w12_stride_nb,
            w12_stride_bm,
            w12_stride_bn,
            w34_stride_nb,
            w34_stride_bm,
            w34_stride_dm,
            BLOCK_M,
            SUB_BLOCK_M,
            BLOCK_N,
            pid,
        )
        (O_0_a, O_0_b, O_0_c, O_0_d) = O_ptr_0
        (WS1_0_a, WS1_0_b, WS1_0_c, WS1_0_d) = WS1_L
        (WS2_0_a, WS2_0_b, WS2_0_c, WS2_0_d) = WS2_L
        (WS3_0_a, WS3_0_b, WS3_0_c, WS3_0_d) = WS3_L
        (WS4_0_a, WS4_0_b, WS4_0_c, WS4_0_d) = WS4_L

        # -- task 1
        (
            Q_ptr_1,
            K_ptr_1,
            V_ptr_1,
            O_ptr_X_1,
            WS1_1,
            WS1_X,
            WS2_1,
            WS2_X,
            WS3_1,
            WS3_X,
            WS4_X,
        ) = get_params(
            Q,
            K,
            V,
            Out,
            WS1,
            WS2,
            WS3,
            WS4,
            block_idx_1,
            NUM_BLOCKS_M,
            H,
            stride_qz,
            stride_qh,
            N_CTX,
            HEAD_DIM,
            stride_qm,
            stride_qk,
            stride_kn,
            stride_kk,
            stride_vn,
            stride_vk,
            stride_om,
            stride_on,
            w12_stride_nb,
            w12_stride_bm,
            w12_stride_bn,
            w34_stride_nb,
            w34_stride_bm,
            w34_stride_dm,
            BLOCK_M,
            SUB_BLOCK_M,
            BLOCK_N,
            pid + NUM_CORES,
        )
        (O_1_a, O_1_b, O_1_c, O_1_d) = O_ptr_X_1
        (WS1_1_a, WS1_1_b, WS1_1_c, WS1_1_d) = WS1_X
        (WS2_1_a, WS2_1_b, WS2_1_c, WS2_1_d) = WS2_X
        (WS3_1_a, WS3_1_b, WS3_1_c, WS3_1_d) = WS3_X
        (WS4_1_a, WS4_1_b, WS4_1_c, WS4_1_d) = WS4_X

        with dl.async_task(scope=dl.async_task.cube):
            lo, hi = 0, N_CTX  # Process the entire context
            for start_n in range(lo, hi, BLOCK_N):  # Process BLOCK_N columns at a time
                start_n = tl.multiple_of(
                    start_n, BLOCK_N
                )  # Align column start position
                # task 0
                cube_part_a(K_ptr_0, Q_ptr_0, WS1_0, V_ptr_0, 0)
                # task 1
                cube_part_a(K_ptr_1, Q_ptr_1, WS1_1, V_ptr_1, 1)

                # task 0
                cube_part_b(WS2_0, WS3_0, V_ptr_0, 2, 4)
                V_ptr_0 = tl.advance(V_ptr_0, (BLOCK_N, 0))
                K_ptr_0 = tl.advance(K_ptr_0, (BLOCK_N, 0))

                # task 1
                cube_part_b(WS2_1, WS3_1, V_ptr_1, 3, 5)
                V_ptr_1 = tl.advance(V_ptr_1, (BLOCK_N, 0))
                K_ptr_1 = tl.advance(K_ptr_1, (BLOCK_N, 0))

        with dl.async_task(scope=dl.async_task.vector):
            # task 0
            mi0_a, li0_a = create_ub_tensor(SUB_BLOCK_M)
            mi0_b, li0_b = create_ub_tensor(SUB_BLOCK_M)
            mi0_c, li0_c = create_ub_tensor(SUB_BLOCK_M)
            mi0_d, li0_d = create_ub_tensor(SUB_BLOCK_M)

            # task 1
            mi1_a, li1_a = create_ub_tensor(SUB_BLOCK_M)
            mi1_b, li1_b = create_ub_tensor(SUB_BLOCK_M)
            mi1_c, li1_c = create_ub_tensor(SUB_BLOCK_M)
            mi1_d, li1_d = create_ub_tensor(SUB_BLOCK_M)

            lo, hi = 0, N_CTX  # Process the entire context
            for start_n in range(lo, hi, BLOCK_N):  # Process BLOCK_N columns at a time
                # task 0
                dl.wait_cross_flag(dl.SyncFlag.C2V, 0)
                mi0_a, alpha_0_a, li0_a = vec_a(Q, WS1_0_a, WS2_0_a, mi0_a, li0_a, sm)
                mi0_b, alpha_0_b, li0_b = vec_a(Q, WS1_0_b, WS2_0_b, mi0_b, li0_b, sm)
                mi0_c, alpha_0_c, li0_c = vec_a(Q, WS1_0_c, WS2_0_c, mi0_c, li0_c, sm)
                mi0_d, alpha_0_d, li0_d = vec_a(Q, WS1_0_d, WS2_0_d, mi0_d, li0_d, sm)

                dl.set_cross_flag(dl.SyncFlag.V2C, 2)

                # task 1
                dl.wait_cross_flag(dl.SyncFlag.C2V, 1)
                mi1_a, alpha_1_a, li1_a = vec_a(Q, WS1_1_a, WS2_1_a, mi1_a, li1_a, sm)
                mi1_b, alpha_1_b, li1_b = vec_a(Q, WS1_1_b, WS2_1_b, mi1_b, li1_b, sm)
                mi1_c, alpha_1_c, li1_c = vec_a(Q, WS1_1_c, WS2_1_c, mi1_c, li1_c, sm)
                mi1_d, alpha_1_d, li1_d = vec_a(Q, WS1_1_d, WS2_1_d, mi1_d, li1_d, sm)

                dl.set_cross_flag(dl.SyncFlag.V2C, 3)

                # task 0
                dl.wait_cross_flag(dl.SyncFlag.C2V, 4)
                vector_part_b(WS3_0_a, WS4_0_a, alpha_0_a)
                vector_part_b(WS3_0_b, WS4_0_b, alpha_0_b)
                vector_part_b(WS3_0_c, WS4_0_c, alpha_0_c)
                vector_part_b(WS3_0_d, WS4_0_d, alpha_0_d)

                # task 1
                dl.wait_cross_flag(dl.SyncFlag.C2V, 5)
                vector_part_b(WS3_1_a, WS4_1_a, alpha_1_a)
                vector_part_b(WS3_1_b, WS4_1_b, alpha_1_b)
                vector_part_b(WS3_1_c, WS4_1_c, alpha_1_c)
                vector_part_b(WS3_1_d, WS4_1_d, alpha_1_d)

            # task 0
            vector_last(Out, O_0_a, WS4_0_a, li0_a)
            vector_last(Out, O_0_b, WS4_0_b, li0_b)
            vector_last(Out, O_0_c, WS4_0_c, li0_c)
            vector_last(Out, O_0_d, WS4_0_d, li0_d)

            # task 1
            vector_last(Out, O_1_a, WS4_1_a, li1_a)
            vector_last(Out, O_1_b, WS4_1_b, li1_b)
            vector_last(Out, O_1_c, WS4_1_c, li1_c)
            vector_last(Out, O_1_d, WS4_1_d, li1_d)