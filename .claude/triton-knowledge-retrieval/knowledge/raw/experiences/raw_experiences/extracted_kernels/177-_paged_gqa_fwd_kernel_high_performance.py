@triton.jit
def _paged_gqa_fwd_kernel_high_performance(
    Q,
    K_Buffer,
    V_Buffer,
    sm_scale,
    kv_seq_lens,
    Att_Out,
    block_table,
    qk_out,
    p_ptr,
    pv_ptr,
    stride_block_table_batch: tl.constexpr,
    stride_qkbs: tl.constexpr,
    stride_qkh: tl.constexpr,
    stride_pvbs: tl.constexpr,
    stride_pvh: tl.constexpr,
    stride_qbs: tl.constexpr,
    stride_qh: tl.constexpr,
    stride_buf_kbs: tl.constexpr,
    stride_buf_kpage: tl.constexpr,
    stride_buf_kh: tl.constexpr,
    stride_buf_vbs: tl.constexpr,
    stride_buf_vpage: tl.constexpr,
    stride_buf_vh: tl.constexpr,
    stride_mid_ob: tl.constexpr,
    stride_mid_oh: tl.constexpr,
    q_heads_per_kv_head: tl.constexpr,
    q_head_num: tl.constexpr,
    BLOCK_DMODEL: tl.constexpr,
    BLOCK_DPE: tl.constexpr,
    BLOCK_DV: tl.constexpr,
    BLOCK_N: tl.constexpr,
    BLOCK_H: tl.constexpr,
    Lk: tl.constexpr,
    Lv: tl.constexpr,
):
    """
    High-performance forward kernel for Grouped-Query Attention with paged KV cache.

    It supports:
      - bfloat16 and float16
      - Paged KV Cache via block_table
      - GQA (Grouped Query Attention): multiple Q heads share one KV head
      - Multi-buffer strategy for large context handling
      - Split-Dimensions (e.g., Lk = Lk_nope + Lk_rope) via BLOCK_DMODEL/BLOCK_DPE

    The computation is split into "CUBE" and "VECTOR" operations:
      - CUBE: matrix-matrix ops (QK, PV)
      - VECTOR: row-wise reductions (softmax normalization)

    Grid structure:
        grid = (batch_size, num_head_groups)
    """
    cur_batch = tl.program_id(0)
    cur_head_group_id = tl.program_id(1)
    cur_kv_head = cur_head_group_id // tl.cdiv(q_heads_per_kv_head, BLOCK_H)

    # --- Determine effective number of heads per group ---
    if BLOCK_H < q_heads_per_kv_head:
        GROUP_HEAD_NUM: tl.constexpr = BLOCK_H
    else:
        GROUP_HEAD_NUM: tl.constexpr = q_heads_per_kv_head
    offset_h = cur_head_group_id * GROUP_HEAD_NUM + tl.arange(0, GROUP_HEAD_NUM)
    offset_d = tl.arange(0, BLOCK_DMODEL + BLOCK_DPE)
    mask_h = offset_h < q_head_num
    mask_d = offset_d < Lk
    offset_q = (
        cur_batch * stride_qbs + offset_h[:, None] * stride_qh + offset_d[None, :]
    )
    q = tl.load(Q + offset_q, mask=(mask_h[:, None]) & (mask_d[None, :]))

    # --- Get sequence length and page info ---
    cur_seq_len = tl.load(kv_seq_lens + cur_batch)
    page_num = tl.cdiv(cur_seq_len, BLOCK_N)
    cur_page_start = cur_batch * stride_block_table_batch

    # --- Main loop over all pages (physical blocks) ---
    # use `tl.parallel(0, 2)` to divide vector works within a head group into two sub-blocks
    # because npu has two vectors in one AICore
    for sub_block_id in tl.parallel(0, 2, bind_sub_block=True):
        offset_page = tl.arange(0, BLOCK_N)
        offset_dv = tl.arange(0, BLOCK_DV)
        mask_dv = offset_dv < Lv

        history_max = tl.zeros([GROUP_HEAD_NUM // 2], dtype=tl.float32) - float("inf")
        l = tl.zeros([GROUP_HEAD_NUM // 2], dtype=tl.float32)
        acc = tl.zeros([GROUP_HEAD_NUM // 2, BLOCK_DV], dtype=tl.float32)

        offset_sub_h = (
            cur_head_group_id * GROUP_HEAD_NUM
            + sub_block_id * (GROUP_HEAD_NUM // 2)
            + tl.arange(0, GROUP_HEAD_NUM // 2)
        )
        mask_h_sub = offset_sub_h < q_head_num

        for start_n in range(page_num):
            # CUBE1
            page_loc = tl.load(block_table + cur_page_start + start_n)
            mask_page = (start_n * BLOCK_N + offset_page) < cur_seq_len
            offset_k = (
                page_loc * stride_buf_kbs
                + offset_page[:, None] * stride_buf_kpage
                + cur_kv_head * stride_buf_kh
                + offset_d[None, :]
            )
            k = tl.load(
                K_Buffer + offset_k, mask=(mask_page[:, None] & mask_d[None, :])
            )
            k = tl.trans(k, (1, 0))
            qk = tl.dot(q, k)

            offset_qk = (
                cur_batch * stride_qkbs
                + offset_h[:, None] * stride_qkh
                + start_n * BLOCK_N
                + offset_page[None, :]
            )
            tl.store(
                qk_out + offset_qk, qk, mask=(mask_h[:, None] & mask_page[None, :])
            )

            # Prepare V for later use
            offset_v = (
                page_loc * stride_buf_vbs
                + offset_page[:, None] * stride_buf_vpage
                + cur_kv_head * stride_buf_vh
                + offset_dv[None, :]
            )
            v = tl.load(
                V_Buffer + offset_v, mask=(mask_page[:, None] & mask_dv[None, :])
            )

            # VECTOR1
            offset_qk_sub = (
                cur_batch * stride_qkbs
                + offset_sub_h[:, None] * stride_qkh
                + start_n * BLOCK_N
                + offset_page[None, :]
            )
            qk_sub_block = tl.load(
                qk_out + offset_qk_sub, mask=(mask_h_sub[:, None] & mask_page[None, :])
            )
            qk_sub_block = tl.where(
                (mask_h_sub[:, None] & mask_page[None, :]), qk_sub_block, float("-inf")
            )
            qk_sub_block = qk_sub_block * sm_scale
            new_e_max = tl.maximum(tl.max(qk_sub_block, 1), history_max)
            p_sub_exp = tl.exp(qk_sub_block - new_e_max[:, None])
            tl.store(
                p_ptr + offset_qk_sub,
                p_sub_exp,
                mask=(mask_h_sub[:, None] & mask_page[None, :]),
            )

            re_scale = tl.exp(history_max - new_e_max)
            l = l * re_scale + tl.sum(p_sub_exp, 1)
            history_max = new_e_max

            # CUBE2
            p = tl.load(p_ptr + offset_qk, mask=(mask_h[:, None] & mask_page[None, :]))
            pv = tl.dot(p.to(v.dtype), v)
            offset_pv = (
                cur_batch * stride_pvbs
                + offset_h[:, None] * stride_pvh
                + offset_dv[None, :]
            )
            tl.store(pv_ptr + offset_pv, pv, mask=(mask_h[:, None] & mask_dv[None, :]))

            # VECTOR2
            offset_sub_pv = (
                cur_batch * stride_pvbs
                + offset_sub_h[:, None] * stride_pvh
                + offset_dv[None, :]
            )
            sub_pv = tl.load(
                pv_ptr + offset_sub_pv, mask=(mask_h_sub[:, None] & mask_dv[None, :])
            )
            acc = acc * re_scale[:, None] + sub_pv

        # Final Vector
        offs_mid_o = (
            cur_batch * stride_mid_ob
            + offset_sub_h[:, None] * stride_mid_oh
            + offset_dv[None, :]
        )
        tl.store(
            Att_Out + offs_mid_o,
            acc / l[:, None],
            mask=(mask_h_sub[:, None] & mask_dv[None, :]),
        )