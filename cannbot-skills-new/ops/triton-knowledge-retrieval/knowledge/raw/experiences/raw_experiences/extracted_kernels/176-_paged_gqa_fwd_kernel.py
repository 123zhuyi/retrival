@triton.jit
def _paged_gqa_fwd_kernel(
    Q,
    K_Buffer,
    V_Buffer,
    sm_scale,
    kv_seq_lens,
    Att_Out,
    block_table,
    stride_block_table_batch: tl.constexpr,
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
    Forward kernel for Grouped-Query Attention (GQA) with paged KV cache.

    GQA allows multiple query heads to share the same key/value heads,
    reducing memory bandwidth while preserving expressiveness.

    Uses online softmax for numerical stability during incremental decoding.

    Args:
        Q (Tensor): Queries, shape [batch_size, q_head_num, Lk].
        K_Buffer (Tensor): Paged key cache, shape [num_blocks, page_size, kv_head_num, Lk].
        V_Buffer (Tensor): Paged value cache, shape [num_blocks, page_size, kv_head_num, Lv].
        sm_scale (float): Attention scaling factor.
        kv_seq_lens (Tensor): Sequence lengths, shape [batch_size].
        Att_Out (Tensor): Output buffer, shape [batch_size, q_head_num, Lv].
        block_table (Tensor): Logical-to-physical block mapping.
        ... (strides): Memory strides.
        q_heads_per_kv_head (int): Ratio of Q heads to KV heads.
        q_head_num (int): Total number of query heads.
        BLOCK_DMODEL (int): Tiled size for key dimension (padded).
        BLOCK_DPE (int): Optional extra dim for RoPE (can be zero).
        BLOCK_DV (int): Tiled size for value dimension (padded).
        BLOCK_N (int): Page size.
        BLOCK_H (int): Number of query heads processed per block.
        Lk (int): Actual key dimension.
        Lv (int): Actual value dimension.
    """
    cur_batch = tl.program_id(0)
    cur_kv_head = tl.program_id(1)

    if BLOCK_H < q_heads_per_kv_head:
        HEAD_NUM: tl.constexpr = BLOCK_H
    else:
        HEAD_NUM: tl.constexpr = q_heads_per_kv_head
    cur_q_head_start = cur_kv_head * q_heads_per_kv_head

    for head_block_start in range(0, q_heads_per_kv_head, HEAD_NUM):
        # Step 1: Load Q_nope + Q_rope
        offset_h = cur_q_head_start + head_block_start + tl.arange(0, HEAD_NUM)
        offset_d = tl.arange(0, BLOCK_DMODEL + BLOCK_DPE)
        mask_h = offset_h < q_head_num
        mask_d = offset_d < Lk
        offset_q = (
            cur_batch * stride_qbs + offset_h[:, None] * stride_qh + offset_d[None, :]
        )
        q = tl.load(Q + offset_q, mask=(mask_h[:, None]) & (mask_d[None, :]))

        # Step 2: Iterate over physical blocks using PagedAttention
        cur_seq_len = tl.load(kv_seq_lens + cur_batch)
        page_num = tl.cdiv(cur_seq_len, BLOCK_N)

        offset_page = tl.arange(0, BLOCK_N)
        offset_dv = tl.arange(0, BLOCK_DV)
        mask_dv = offset_dv < Lv

        history_max = tl.zeros([HEAD_NUM], dtype=tl.float32) - float("inf")
        l = tl.zeros([HEAD_NUM], dtype=tl.float32)
        acc = tl.zeros([HEAD_NUM, BLOCK_DV], dtype=tl.float32)
        for page_id in range(page_num):
            # Load K
            page_loc = tl.load(
                block_table + cur_batch * stride_block_table_batch + page_id
            )
            offset_k = (
                page_loc * stride_buf_kbs
                + offset_page[:, None] * stride_buf_kpage
                + cur_kv_head * stride_buf_kh
                + offset_d[None, :]
            )
            mask_page = (page_id * BLOCK_N + offset_page) < cur_seq_len
            k = tl.load(
                K_Buffer + offset_k, mask=(mask_page[:, None] & mask_d[None, :])
            )
            k = tl.trans(k, (1, 0))
            qk = tl.dot(q, k)

            # Load V early to overlap with computation
            offset_v = (
                page_loc * stride_buf_vbs
                + offset_page[:, None] * stride_buf_vpage
                + cur_kv_head * stride_buf_vh
                + offset_dv[None, :]
            )
            v = tl.load(
                V_Buffer + offset_v, mask=(mask_page[:, None] & mask_dv[None, :])
            )

            qk = qk * sm_scale
            qk = tl.where((mask_h[:, None] & mask_page[None, :]), qk, float("-inf"))
            new_e_max = tl.maximum(tl.max(qk, 1), history_max)
            re_scale = tl.exp(history_max - new_e_max)
            p_exp = tl.exp(qk - new_e_max[:, None])

            # Online softmax update
            l = l * re_scale + tl.sum(p_exp, 1)
            acc = acc * re_scale[:, None] + tl.dot(p_exp.to(v.dtype), v)
            history_max = new_e_max

        offs_mid_o = (
            cur_batch * stride_mid_ob
            + offset_h[:, None] * stride_mid_oh
            + offset_dv[None, :]
        )
        tl.store(
            Att_Out + offs_mid_o,
            acc / l[:, None],
            mask=(mask_h[:, None] & mask_dv[None, :]),
        )