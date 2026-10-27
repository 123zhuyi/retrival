@triton.jit
def _hstu_attn_bwd(  # noqa C901
    Q, K, V, Grad, DQ, DK, DV,
    stride_qm: tl.constexpr,
    stride_qh: tl.constexpr,
    stride_kn: tl.constexpr,
    stride_kh: tl.constexpr,
    stride_vn: tl.constexpr,
    stride_vh: tl.constexpr,
    stride_dom: tl.constexpr,
    stride_doh: tl.constexpr,
    seq_offsets,
    alpha: tl.constexpr,
    batch: tl.constexpr,
    head_num: tl.constexpr,
    MAX_SEQ_LEN: tl.constexpr,
    head_dim: tl.constexpr,
    CAUSAL: tl.constexpr,
    HAS_BIAS: tl.constexpr,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
    bias,
):
    off = tl.program_id(0)
    off_batch = off // head_num
    off_head = off % head_num
    off_head = off_head.to(tl.int64)
    seq_start = tl.load(seq_offsets + off_batch).to(tl.int64)
    seq_end = tl.load(seq_offsets + off_batch + 1)
    seq_len = (seq_end - seq_start).to(tl.int32)
    # offset pointers for batch/head
    q_offset = seq_start * stride_qm + off_head * stride_qh
    k_offset = seq_start * stride_kn + off_head * stride_kh
    v_offset = seq_start * stride_vn + off_head * stride_vh
    grad_offset = seq_start * stride_dom + off_head * stride_doh
    bias_offset = off_batch * head_num * MAX_SEQ_LEN * MAX_SEQ_LEN + off_head * MAX_SEQ_LEN * MAX_SEQ_LEN
    for start_n in range(0, seq_len, BLOCK_N):
        _hstu_attn_bwd_one_col_block(
            start_n=start_n,
            seq_len=seq_len,
            Q=Q + q_offset,
            K=K + k_offset,
            V=V + v_offset,
            DOut=Grad + grad_offset,
            DQ=DQ + q_offset,
            DK=DK + k_offset,
            DV=DV + v_offset,
            stride_qm=stride_qm,
            stride_kn=stride_kn,
            stride_vn=stride_vn,
            stride_dom=stride_dom,
            stride_dqm=stride_qm,
            stride_dkn=stride_kn,
            stride_dvn=stride_vn,
            alpha=alpha,
            MAX_SEQ_LEN=MAX_SEQ_LEN,
            CAUSAL=CAUSAL,
            HAS_BIAS=HAS_BIAS,
            BLOCK_D_Q=head_dim,
            BLOCK_D_V=head_dim,
            BLOCK_M=BLOCK_M,
            BLOCK_N=BLOCK_N,
            bias=bias + bias_offset if HAS_BIAS else bias,
        )