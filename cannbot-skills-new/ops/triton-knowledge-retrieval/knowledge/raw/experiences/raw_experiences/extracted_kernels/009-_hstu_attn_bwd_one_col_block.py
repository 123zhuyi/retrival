@triton.jit
def _hstu_attn_bwd_one_col_block(  # noqa C901
    start_n,
    seq_len,
    Q,
    K,
    V,
    DOut,
    DQ,
    DK,
    DV,
    stride_qm,
    stride_kn,
    stride_vn,
    stride_dom,
    stride_dqm,
    stride_dkn,
    stride_dvn,
    alpha,
    MAX_SEQ_LEN,
    CAUSAL: tl.constexpr,
    HAS_BIAS: tl.constexpr,
    BLOCK_D_Q: tl.constexpr,
    BLOCK_D_V: tl.constexpr,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
    bias,
):
    # Work on the subsequence dv[start_n, start_n + BLOCK_N, :]
    if CAUSAL:
        low = start_n
        high = seq_len
    else:
        low = 0
        high = seq_len

    # initialize row/col offsets
    offs_m = tl.arange(0, BLOCK_M)
    offs_qk_d = tl.arange(0, BLOCK_D_Q)
    offs_v_d = tl.arange(0, BLOCK_D_V)
    offs_n = start_n + tl.arange(0, BLOCK_N)

    dq_ptrs = DQ + (offs_m[:, None] * stride_dqm + offs_qk_d[None, :])
    dk = tl.zeros([BLOCK_N, BLOCK_D_Q], dtype=tl.float32)
    dv = tl.zeros([BLOCK_N, BLOCK_D_V], dtype=tl.float32)

    mask_n = offs_n < seq_len
    q_ptrs = Q + (offs_m[:, None] * stride_qm + offs_qk_d[None, :])
    do_ptrs = DOut + (offs_m[:, None] * stride_dom + offs_v_d[None, :])
    k_ptrs = K + (offs_n[:, None] * stride_kn + offs_qk_d[None, :])
    v_ptrs = V + (offs_n[:, None] * stride_vn + offs_v_d[None, :])
    k = tl.load(k_ptrs, mask=mask_n[:, None], other=0.0)
    v = tl.load(v_ptrs, mask=mask_n[:, None], other=0.0)

    max_ids = seq_len
    pos_offs_n = offs_n
    # loop over rows
    for start_m in tl.range(low, high, BLOCK_M):
        bias_block_ptr = None
        if HAS_BIAS:
            bias_block_ptr = tl.make_block_ptr(
                base=bias,
                shape=(MAX_SEQ_LEN, MAX_SEQ_LEN),
                strides=(MAX_SEQ_LEN, 1),
                offsets=(start_m, start_n),
                block_shape=(BLOCK_M, BLOCK_N),
                order=(1, 0),
            )
        start_m = tl.multiple_of(start_m, BLOCK_M)
        dk, dv = _hstu_attn_bwd_one_block(
            start_m=start_m,
            offs_n=offs_n,
            offs_m=offs_m,
            q_ptrs=q_ptrs,
            dq_ptrs=dq_ptrs,
            mask_n=mask_n,
            do_ptrs=do_ptrs,
            dk=dk,
            dv=dv,
            k=k,
            v=v,
            pos_offs_n=pos_offs_n,
            seq_len=seq_len,
            max_ids=max_ids,
            stride_qm=stride_qm,
            stride_dom=stride_dom,
            stride_dqm=stride_dqm,
            alpha=alpha,
            MAX_SEQ_LEN=MAX_SEQ_LEN,
            CAUSAL=CAUSAL,
            HAS_BIAS=HAS_BIAS,
            BLOCK_M=BLOCK_M,
            BLOCK_N=BLOCK_N,
            bias_block_ptr=bias_block_ptr,
        )
    # write-back
    dk = dk * alpha
    dv_ptrs = DV + (offs_n[:, None] * stride_dvn + offs_v_d[None, :])
    dk_ptrs = DK + (offs_n[:, None] * stride_dkn + offs_qk_d[None, :])
    tl.store(dv_ptrs, dv.to(k.dtype), mask=mask_n[:, None])
    tl.store(dk_ptrs, dk.to(k.dtype), mask=mask_n[:, None])