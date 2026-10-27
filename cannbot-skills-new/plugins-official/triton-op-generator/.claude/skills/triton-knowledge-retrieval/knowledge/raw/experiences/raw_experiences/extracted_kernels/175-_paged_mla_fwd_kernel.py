@triton.jit
def _paged_mla_fwd_kernel(
    Q,
    K_NOPE_Buffer,
    K_ROPE_Buffer,
    sm_scale,
    kv_seq_lens,
    Att_Out,
    block_table,
    stride_block_table_batch: tl.constexpr,
    stride_qbs: tl.constexpr,
    stride_qh: tl.constexpr,
    stride_buf_knbs: tl.constexpr,
    stride_buf_knpage: tl.constexpr,
    stride_buf_knh: tl.constexpr,
    stride_buf_krbs: tl.constexpr,
    stride_buf_krpage: tl.constexpr,
    stride_buf_krh: tl.constexpr,
    stride_mid_ob: tl.constexpr,
    stride_mid_oh: tl.constexpr,
    q_heads_per_kv_head: tl.constexpr,
    q_head_num: tl.constexpr,
    BLOCK_DNOPE: tl.constexpr,
    BLOCK_DROPE: tl.constexpr,
    BLOCK_N: tl.constexpr,
    BLOCK_H: tl.constexpr,
    Lkv: tl.constexpr,
    Lrope: tl.constexpr,
):
    """
    Forward kernel for Multi-Latent Attention (MLA) with paged KV cache support.

    Support bfloat16 and float16

    MLA separates query/key into two components:
        - K/Q_nope: Content-aware features without rotation.
        - K/Q_rope: Positional features with RoPE applied.

    The attention score is computed as:
        score = (Q_nope @ K_nope.T) + (Q_rope @ K_rope.T)

    Memory layout follows paged attention:
        - KV caches are stored in blocks of shape [page_size, num_heads, head_dim].
        - `block_table` maps logical pages to physical block IDs in NPU memory.

    This kernel computes one group of query heads per launch.

    Args:
        Q (Tensor): Input queries of shape [batch_size, q_head_num, D], where D = Lkv + Lrope.
        K_NOPE_Buffer (Tensor): Paged key buffer (non-rotated) of shape [num_blocks, page_size, kv_head_num, Lkv].
        K_ROPE_Buffer (Tensor): Paged key buffer (RoPE-applied) of shape [num_blocks, page_size, kv_head_num, Lrope].
        sm_scale (float): Scaling factor for attention scores (usually 1/sqrt(head_dim)).
        kv_seq_lens (Tensor): Current sequence lengths for each batch item, shape [batch_size].
        Att_Out (Tensor): Output buffer of shape [batch_size, q_head_num, Lkv].
        block_table (Tensor): Mapping from logical page index to physical block ID, shape [batch_size, max_pages].
        ... (strides): Precomputed strides for efficient indexing.
        q_heads_per_kv_head (int): Number of query heads sharing a single KV head (q_head_num // kv_head_num).
        q_head_num (int): Total number of query heads.
        BLOCK_DNOPE (int): Tiling size for nope dimension (padded to power-of-2).
        BLOCK_DROPE (int): Tiling size for rope dimension (padded to power-of-2).
        BLOCK_N (int): Page size (number of tokens per kv cache block).
        BLOCK_H (int): Number of query heads processed per block.
        Lkv (int): Actual dimension of K/Q_nope.
        Lrope (int): Actual dimension of K/Q_rope.
    """

    cur_batch = tl.program_id(0)
    cur_head_group_id = tl.program_id(1)
    # Determine how many query heads this block processes
    cur_kv_head = cur_head_group_id // tl.cdiv(q_heads_per_kv_head, BLOCK_H)
    if BLOCK_H < q_heads_per_kv_head:
        GROUP_HEAD_NUM: tl.constexpr = BLOCK_H
    else:
        GROUP_HEAD_NUM: tl.constexpr = q_heads_per_kv_head

    # Step 1: Load Q_nope (content part)
    offset_h = cur_head_group_id * GROUP_HEAD_NUM + tl.arange(0, GROUP_HEAD_NUM)
    mask_h = offset_h < q_head_num
    offset_qk_nope = tl.arange(0, BLOCK_DNOPE)
    mask_qk_nope = offset_qk_nope < Lkv
    offset_q = (
        cur_batch * stride_qbs + offset_h[:, None] * stride_qh + offset_qk_nope[None, :]
    )
    q = tl.load(Q + offset_q, mask=(mask_h[:, None]) & (mask_qk_nope[None, :]))

    # Step 2: Load Q_rope (positional part)
    offset_qk_rope = tl.arange(0, BLOCK_DROPE)
    mask_qk_rope = offset_qk_rope < Lrope

    # RoPE part starts after nope part in Q
    offset_qrope = BLOCK_DNOPE + offset_qk_rope
    mask_qrope = offset_qrope < (Lkv + Lrope)
    offset_qpe = (
        cur_batch * stride_qbs + offset_h[:, None] * stride_qh + offset_qrope[None, :]
    )
    q_pe = tl.load(Q + offset_qpe, mask=(mask_h[:, None]) & (mask_qrope[None, :]))

    # Step 3: Iterate over physical blocks using PagedAttention
    cur_seq_len = tl.load(kv_seq_lens + cur_batch)
    page_num = tl.cdiv(cur_seq_len, BLOCK_N)
    offset_page = tl.arange(0, BLOCK_N)

    history_max = tl.zeros([GROUP_HEAD_NUM], dtype=tl.float32) - float("inf")
    l = tl.zeros([GROUP_HEAD_NUM], dtype=tl.float32)
    acc = tl.zeros([GROUP_HEAD_NUM, BLOCK_DNOPE], dtype=tl.float32)
    for page_id in range(page_num):
        # --- Load K_nope ---
        page_loc = tl.load(block_table + cur_batch * stride_block_table_batch + page_id)
        offset_k = (
            page_loc * stride_buf_knbs
            + offset_page[:, None] * stride_buf_knpage
            + cur_kv_head * stride_buf_knh
            + offset_qk_nope[None, :]
        )
        mask_page = (page_id * BLOCK_N + offset_page) < cur_seq_len
        k = tl.load(
            K_NOPE_Buffer + offset_k, mask=(mask_page[:, None] & mask_qk_nope[None, :])
        )
        v = k  # In MLA, V shares same buffer as K_nope

        k = tl.trans(k, (1, 0))
        qk = tl.dot(q, k)

        # --- Load K_rope ---
        offset_krope = (
            page_loc * stride_buf_krbs
            + offset_page[:, None] * stride_buf_krpage
            + cur_kv_head * stride_buf_krh
            + offset_qk_rope[None, :]
        )
        k_pe = tl.load(
            K_ROPE_Buffer + offset_krope,
            mask=(mask_page[:, None] & mask_qk_rope[None, :]),
        )
        k_pe = tl.trans(k_pe, (1, 0))
        # Combine both parts
        qk += tl.dot(q_pe, k_pe)

        qk = qk * sm_scale
        # Online softmax update
        qk = tl.where((mask_h[:, None] & mask_page[None, :]), qk, float("-inf"))
        new_e_max = tl.maximum(tl.max(qk, 1), history_max)
        re_scale = tl.exp(history_max - new_e_max)
        p_exp = tl.exp(qk - new_e_max[:, None])

        l = l * re_scale + tl.sum(p_exp, 1)
        acc = acc * re_scale[:, None] + tl.dot(p_exp.to(v.dtype), v)
        history_max = new_e_max

    offs_mid_o = (
        cur_batch * stride_mid_ob
        + offset_h[:, None] * stride_mid_oh
        + offset_qk_nope[None, :]
    )
    tl.store(
        Att_Out + offs_mid_o,
        acc / l[:, None],
        mask=(mask_h[:, None] & mask_qk_nope[None, :]),
    )