@triton.jit
def _hstu_attn_fwd_compute(  # noqa C901
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
    alpha,
    head_num,
    MAX_SEQ_LEN,
    off_batch,
    off_head,
    start_m,
    seq_start,
    seq_len,
    CAUSAL: tl.constexpr,
    HAS_BIAS: tl.constexpr,
    BLOCK_D_Q: tl.constexpr,
    BLOCK_D_V: tl.constexpr,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
    mask_block,
    bias,
):
    off_head = off_head.to(tl.int64)
    off_seq = seq_start.to(tl.int64)
    start_m = start_m.to(tl.int32)

    # initialize offsets
    q_offset = off_seq * stride_qm + off_head * stride_qh
    k_offset = off_seq * stride_kn + off_head * stride_kh
    v_offset = off_seq * stride_vn + off_head * stride_kh

    Q_block_ptr = tl.make_block_ptr(
        base=Q + q_offset,
        shape=(seq_len, BLOCK_D_Q),
        strides=(stride_qm, 1),
        offsets=(start_m, 0),
        block_shape=(BLOCK_M, BLOCK_D_Q),
        order=(1, 0),
    )
    k_block_ptr = tl.make_block_ptr(
        base=K + k_offset,
        shape=(seq_len, BLOCK_D_Q),
        strides=(stride_kn, 1),
        offsets=(0, 0),
        block_shape=(BLOCK_N, BLOCK_D_Q),
        order=(1, 0),
    )
    v_block_ptr = tl.make_block_ptr(
        base=V + v_offset,
        shape=(seq_len, BLOCK_D_V),
        strides=(stride_vn, 1),
        offsets=(0, 0),
        block_shape=(BLOCK_N, BLOCK_D_V),
        order=(1, 0),
    )
    q = tl.load(Q_block_ptr)

    acc = tl.zeros([BLOCK_M, BLOCK_D_V], dtype=tl.float32)
    if CAUSAL:
        low = 0
        high = start_m + BLOCK_M
    else:
        low = 0
        high = seq_len

    bias_block_ptr = None
    if HAS_BIAS:
        bias_block_ptr = tl.make_block_ptr(
            base=bias + off_batch * head_num * MAX_SEQ_LEN * MAX_SEQ_LEN + off_head * MAX_SEQ_LEN * MAX_SEQ_LEN,
            shape=(MAX_SEQ_LEN, MAX_SEQ_LEN),
            strides=(MAX_SEQ_LEN, 1),
            offsets=(start_m, 0),
            block_shape=(BLOCK_M, BLOCK_N),
            order=(1, 0),
        )

    for start_n in range(low, high, BLOCK_N):
        acc += _hstu_attn_fwd_one_block(
            q=q,
            k_block_ptr=k_block_ptr,
            v_block_ptr=v_block_ptr,
            bias_block_ptr=bias_block_ptr,
            alpha=alpha,
            MAX_SEQ_LEN=MAX_SEQ_LEN,
            CAUSAL=CAUSAL and start_m == start_n,
            HAS_BIAS=HAS_BIAS,
            mask_block=mask_block,
        )
        k_block_ptr = tl.advance(k_block_ptr, (BLOCK_N, 0))
        v_block_ptr = tl.advance(v_block_ptr, (BLOCK_N, 0))
        if HAS_BIAS:
            bias_block_ptr = tl.advance(bias_block_ptr, (0, BLOCK_N))

    # rematerialize offsets to save registers
    offs_v_d = tl.arange(0, BLOCK_D_V)
    off_o = Out + off_seq * stride_om + off_head * stride_oh
    offs_m = start_m + tl.arange(0, BLOCK_M)
    out_ptrs = off_o + offs_m[:, None] * stride_om + offs_v_d[None, :]
    tl.store(out_ptrs, acc, mask=(offs_m < seq_len)[:, None])