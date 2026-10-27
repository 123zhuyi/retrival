@triton.jit(do_not_specialize=["T1", "T2", "S1", "S2"])
def sparse_flash_attention_prefill_kernel(
    # Input pointers
    Q_ptr,  # [B, S, H, D_qk] - queries (combined nope + rope)
    K_ptr,  # [B, T, H, D_qk] - keys (combined nope + rope)
    V_ptr,  # [B, T, H, D_v] - values 合并前的 K
    Indices_ptr,  # [B, S, K] - topk indices
    O_ptr,
    # Workspace pointers
    Wksp_K_ptr,
    Wksp_K2_ptr,
    Wksp_qk_ptr,
    Wksp_score_ptr,
    Wksp_sv_ptr,
    # Strides for Q
    stride_qt1,
    stride_qn2,
    stride_qn1,
    stride_qd,
    # Strides for indices
    stride_it1,
    stride_in2,
    stride_ik,
    # Strides for O
    stride_ot1,
    stride_on1,
    stride_wod,
    # Strides for aux K
    stride_wkt1,
    stride_wkn2,
    stride_wksbs,
    stride_wkd,
    # Strides for aux qk
    stride_wqkt1,
    stride_wqkn1,
    stride_wqks2,
    # Strides for aux score
    stride_wscoret1,
    stride_wscoren1,
    stride_wscores2,
    # Strides for aux sv
    stride_wsvt1,
    stride_wsvn1,
    stride_wsvd,
    # Param
    scale,
    ## Basic
    T1,
    N1: tl.constexpr,
    T2,
    G: tl.constexpr,
    D_qk: tl.constexpr,
    D_v: tl.constexpr,
    K: tl.constexpr,
    ## Extend
    B: tl.constexpr,
    S1,
    S2,
    # Proc per core
    bs1_per_core,
    # Block sizes
    BLOCK_G: tl.constexpr,
    BLOCK_SBS: tl.constexpr,
    BLOCK_K: tl.constexpr,
    BLOCK_V: tl.constexpr,
):
    # Grid: (T1(approx),)
    pid = tl.program_id(0)
    t1_start = pid * bs1_per_core

    t1_end = tl.minimum(T1, (pid + 1) * bs1_per_core)

    for idx_t1 in range(t1_start, t1_end):
        idx_b = idx_t1 // S1
        cur_s2 = S2
        beg_s2 = idx_b * S2

        # Compute base offsets
        topk_base = idx_t1 * stride_it1
        q_base = idx_t1 * stride_qt1

        # Cube WKSP
        wk_base = pid * stride_wkt1
        wdq_base = pid * stride_wqkt1
        wdqt_base = pid * stride_wscoret1

        wsv_base = pid * stride_wsvt1
        wsacc_base = idx_t1 * stride_ot1

        sub_vec_id = tle.sub_vec_id()
        sub_vec_num = tle.sub_vec_num()
        CACHE_BLOCK: tl.constexpr = (
            64  # 98304 // D_qk        # 一次满 ub 搬运的最大行数, TODO: CACHE_BLOCK > K 场景
        )
        offs_sub_Dv = tl.arange(0, D_qk)
        sliced_row_num = BLOCK_SBS // sub_vec_num
        row_offset = sliced_row_num * sub_vec_id

        HEAD_LOOP_TIMES: tl.constexpr = triton.cdiv(G, BLOCK_G)  # n1 轴切分
        BLOCK_SUB_G: tl.constexpr = BLOCK_G // 2
        sbs_loop_times = tl.cdiv(K, BLOCK_SBS)  # s2 轴切分
        for idx_n1_blk in range(0, HEAD_LOOP_TIMES):  # n1 block 循环
            offs_G = idx_n1_blk * BLOCK_G + tl.arange(0, BLOCK_G)
            arange_sub_G = sub_vec_id * BLOCK_SUB_G + tl.arange(0, BLOCK_SUB_G)
            offs_sub_G = idx_n1_blk * BLOCK_G + arange_sub_G

            lse = tl.zeros([BLOCK_SUB_G], dtype=tl.float32) - float("inf")

            for idx_topk_block in range(0, sliced_row_num, CACHE_BLOCK):
                idx_offset = row_offset + idx_topk_block + tl.arange(0, CACHE_BLOCK)
                sparse_ids = (
                    tl.load(Indices_ptr + topk_base + idx_offset * stride_ik) + beg_s2
                )
                result = tle.index_select_simd(
                    K_ptr,
                    dim=0,
                    index=sparse_ids,
                    src_shape=[cur_s2 * B, D_qk],
                    src_offset=[-1, 0],
                    read_shape=[-1, D_qk],
                )
                tl.store(
                    Wksp_K_ptr
                    + wk_base
                    + idx_offset[:, None] * stride_wksbs
                    + offs_sub_Dv[None, :] * stride_wkd,
                    result,
                )

            for idx_sub_sbs in range(0, sbs_loop_times):  # s2 循环
                offs_sbs = idx_sub_sbs * BLOCK_SBS + tl.arange(0, BLOCK_SBS)

                ## Calculate qk = q @ k^T
                qk_l1 = cube1(
                    Q_ptr,
                    Wksp_K_ptr,
                    # Strides for Q
                    stride_qt1,
                    stride_qn2,
                    stride_qn1,
                    stride_qd,
                    # Strides for aux K
                    stride_wkt1,
                    stride_wkn2,
                    stride_wksbs,
                    stride_wkd,
                    # Param
                    D_qk,
                    BLOCK_G,
                    BLOCK_SBS,
                    # Meta
                    offs_G,
                    offs_sbs,
                    q_base,
                    wk_base,
                    # Block size
                    BLOCK_K,
                )
                tl.store(
                    Wksp_qk_ptr
                    + wdq_base
                    + tl.arange(0, BLOCK_G)[:, None] * stride_wqkn1
                    + tl.arange(0, BLOCK_SBS)[None, :] * stride_wqks2,
                    qk_l1,
                )

                # qk shape: [G, BLOCK_SBS]
                qk = tl.load(
                    Wksp_qk_ptr
                    + wdq_base
                    + arange_sub_G[:, None] * stride_wqkn1
                    + tl.arange(0, BLOCK_SBS)[None, :] * stride_wqks2
                )
                qk *= scale  # [BLOCK_G, BLOCK_SBS]

                mask = offs_sbs < K
                qk = tl.where(mask[None, :], qk, -float("inf"))
                local_max = tl.max(qk, 1)  # [BLOCK_G]
                local_exp = tl.exp(qk - local_max[:, None])  # [BLOCK_G, BLOCK_SBS]
                local_sum = tl.sum(local_exp, 1)  # [BLOCK_G]
                local_lse = local_max + tl.log(local_sum)  # [BLOCK_G]
                mean_lse = (lse + local_lse) / 2  # [BLOCK_G]
                new_lse = mean_lse + tl.log(
                    tl.exp(lse - mean_lse) + tl.exp(local_lse - mean_lse)
                )  # [BLOCK_G]
                new_lse = tl.where(new_lse != new_lse, local_lse, new_lse)  # [BLOCK_G]
                current_weight = tl.exp(local_lse - new_lse)  # [BLOCK_G]
                score = local_exp / local_sum[:, None]  # [BLOCK_G, BLOCK_SBS]
                tl.store(
                    Wksp_score_ptr
                    + wdqt_base
                    + arange_sub_G[:, None] * stride_wscoren1
                    + tl.arange(0, BLOCK_SBS)[None, :] * stride_wscores2,
                    score.to(Q_ptr.dtype.element_ty),
                )

                blk_v_loop_time: tl.constexpr = triton.cdiv(D_v, BLOCK_V)

                start_row = (idx_sub_sbs + 1) * BLOCK_SBS
                end_row = tl.minimum(start_row + sliced_row_num, K)
                for idx_topk_block in range(start_row, end_row, CACHE_BLOCK):
                    idx_offset = row_offset + idx_topk_block + tl.arange(0, CACHE_BLOCK)
                    sparse_ids = (
                        tl.load(Indices_ptr + topk_base + idx_offset * stride_ik)
                        + beg_s2
                    )
                    result = tle.index_select_simd(
                        K_ptr,
                        dim=0,
                        index=sparse_ids,
                        src_shape=[cur_s2 * B, D_qk],
                        src_offset=[-1, 0],
                        read_shape=[-1, D_qk],
                    )
                    tl.store(
                        Wksp_K2_ptr
                        + wk_base
                        + idx_offset[:, None] * stride_wksbs
                        + offs_sub_Dv[None, :] * stride_wkd,
                        result,
                    )

                for idx_blk_v in range(0, blk_v_loop_time):
                    offset_V = idx_blk_v * BLOCK_V + tl.arange(0, BLOCK_V)
                    kv_cache = tl.load(
                        Wksp_K_ptr
                        + wk_base
                        + offs_sbs[:, None] * stride_wksbs
                        + offset_V[None, :] * stride_wkd
                    )
                    score_l1 = tl.load(
                        Wksp_score_ptr
                        + wdq_base
                        + tl.arange(0, BLOCK_G)[:, None] * stride_wqkn1
                        + tl.arange(0, BLOCK_SBS)[None, :] * stride_wqks2
                    )
                    sv = tl.dot(score_l1, kv_cache)
                    tl.store(
                        Wksp_sv_ptr
                        + wsv_base
                        + offs_G[:, None] * stride_wsvn1
                        + offset_V[None, :] * stride_wsvd,
                        sv,
                    )

                    acc = tl.load(
                        O_ptr
                        + wsacc_base
                        + offs_sub_G[:, None] * stride_on1
                        + offset_V[None, :] * stride_wod
                    )
                    sv_sub = tl.load(
                        Wksp_sv_ptr
                        + wsv_base
                        + offs_sub_G[:, None] * stride_wsvn1
                        + offset_V[None, :] * stride_wsvd
                    )
                    acc = acc * tl.exp(lse - new_lse)[:, None]  # [BLOCK_G, BLOCK_SBS]

                    acc += sv_sub * current_weight[:, None]

                    tl.store(
                        O_ptr
                        + wsacc_base
                        + offs_sub_G[:, None] * stride_on1
                        + offset_V[None, :] * stride_wod,
                        acc,
                    )

                lse = new_lse