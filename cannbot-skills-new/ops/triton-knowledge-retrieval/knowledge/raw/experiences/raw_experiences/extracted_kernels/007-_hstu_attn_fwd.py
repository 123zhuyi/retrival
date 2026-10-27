@triton.jit
def _hstu_attn_fwd(  # noqa C901
    Q,
    K,
    V,
    seq_offsets,
    Out,
    stride_qm: tl.constexpr,
    stride_qh: tl.constexpr,
    stride_kn: tl.constexpr,
    stride_kh: tl.constexpr,
    stride_vn: tl.constexpr,
    stride_vh: tl.constexpr,
    stride_om: tl.constexpr,
    stride_oh: tl.constexpr,
    alpha: tl.constexpr,
    batch: tl.constexpr,
    head_num: tl.constexpr,
    MAX_SEQ_LEN: tl.constexpr,
    head_dim: tl.constexpr,
    CAUSAL: tl.constexpr,
    HAS_BIAS: tl.constexpr,
    CORE_NUM: tl.constexpr,
    tasks: tl.constexpr,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
    mask,
    bias,
):
    core_id = tl.program_id(0)
    cur_batch = 0
    mask_block = None
    if CAUSAL and mask is not None:
        mask_ptr = tl.make_block_ptr(
            base=mask,
            shape=(MAX_SEQ_LEN, MAX_SEQ_LEN),
            strides=(MAX_SEQ_LEN, 1),
            offsets=(0, 0),
            block_shape=(BLOCK_M, BLOCK_M),
            order=(1, 0),
        )
        mask_block = tl.load(mask_ptr)
    for col in range(core_id, tasks, CORE_NUM):
        seq_end = tl.load(seq_offsets + cur_batch + 1)
        start_m = col * BLOCK_M
        while start_m >= seq_end * head_num // 2:
            cur_batch += 1
            seq_end = tl.load(seq_offsets + cur_batch + 1)
        seq_start = tl.load(seq_offsets + cur_batch)
        seq_len = seq_end - seq_start
        off_batch = cur_batch
        off_head = (start_m - seq_start * head_num // 2) // (seq_len // 2)
        start_m_1 = (start_m - seq_start * head_num // 2) % (seq_len // 2)
        start_m_2 = seq_len - start_m_1 - BLOCK_M
        _hstu_attn_fwd_compute(Q, K, V, seq_offsets, Out, stride_qm, stride_qh, stride_kn, stride_kh,
            stride_vn, stride_vh, stride_om, stride_oh, alpha, head_num, MAX_SEQ_LEN, off_batch, off_head,
            start_m_1, seq_start, seq_len, CAUSAL, HAS_BIAS, head_dim, head_dim, BLOCK_M, BLOCK_N,
            mask_block=mask_block,
            bias=bias,
        )
        _hstu_attn_fwd_compute(Q, K, V, seq_offsets, Out, stride_qm, stride_qh, stride_kn, stride_kh,
            stride_vn, stride_vh, stride_om, stride_oh, alpha, head_num, MAX_SEQ_LEN, off_batch, off_head,
            start_m_2, seq_start, seq_len, CAUSAL, HAS_BIAS, head_dim, head_dim, BLOCK_M, BLOCK_N,
            mask_block=mask_block,
            bias=bias,
        )