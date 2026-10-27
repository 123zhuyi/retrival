@triton.jit
def attention_sinks_kernel(
    query,
    k_cache,
    v_cache,
    sinks,
    attn_out,
    block_tables,
    kv_seq_lens,
    scale,
    sliding_window_size,
    q_head_num: tl.constexpr,
    k_head_num: tl.constexpr,
    block_group_size: tl.constexpr,
    D: tl.constexpr,
    PAGE_SIZE: tl.constexpr,
    MAX_BLOCKS: tl.constexpr,
):
    i_s, i_gh = tl.program_id(0), tl.program_id(1)
    i_kvh = i_gh * block_group_size // (q_head_num // k_head_num)

    kv_seq_len = tl.load(kv_seq_lens + i_s)
    page_num = tl.cdiv(kv_seq_len, PAGE_SIZE)
    page_num = min(page_num, MAX_BLOCKS)

    start_page_num = 0
    start_kv_len = 0
    if sliding_window_size != -1 and kv_seq_len > sliding_window_size:
        start_kv_len = (kv_seq_len - sliding_window_size).to(tl.int32)
        start_page_num = start_kv_len // PAGE_SIZE

    cur_page_start = i_s * MAX_BLOCKS
    offset_page = tl.arange(0, PAGE_SIZE)
    offset_d = tl.arange(0, D)
    Br: tl.constexpr = block_group_size

    sink = tl.load(sinks + i_gh * block_group_size + tl.arange(0, Br))
    history_max = tl.zeros([Br], dtype=tl.float32) + sink
    l = tl.zeros([Br], dtype=tl.float32)
    acc = tl.zeros([Br, D], dtype=tl.float32)

    offset_seq = (i_s * q_head_num + i_gh * block_group_size + tl.arange(0, Br)) * D
    q = tl.load(query + offset_seq[:, None] + offset_d[None, :]).to(tl.float32)

    for page_idx in range(start_page_num, page_num):
        block_idx = tl.load(block_tables + cur_page_start + page_idx)
        mask_page = ((page_idx * PAGE_SIZE + offset_page) < kv_seq_len) & (
            (page_idx * PAGE_SIZE + offset_page) >= start_kv_len
        )

        offset_k = (
            block_idx * PAGE_SIZE * k_head_num * D
            + offset_page[:, None] * k_head_num * D
            + i_kvh * D
            + offset_d[None, :]
        )
        k = tl.load(k_cache + offset_k, mask=mask_page[:, None]).to(tl.float32)
        v = tl.load(v_cache + offset_k, mask=mask_page[:, None]).to(tl.float32)

        k = tl.trans(k, (1, 0))
        qk = tl.dot(q, k)
        qk = qk * scale
        qk = tl.where(mask_page[None, :], qk, float("-inf"))

        new_e_max = tl.maximum(tl.max(qk, 1), history_max)
        re_scale = tl.exp(history_max - new_e_max)
        p_exp = tl.exp(qk - new_e_max[:, None])

        # Online softmax update
        l = l * re_scale + tl.sum(p_exp, 1)
        acc = acc * re_scale[:, None] + tl.dot(p_exp.to(v.dtype), v)

        history_max = new_e_max

    sink = tl.math.exp(sink - history_max)
    l = l + sink
    acc = acc / l[:, None]
    tl.store(
        attn_out + offset_seq[:, None] + offset_d[None, :],
        acc.to(attn_out.type.element_ty),
    )