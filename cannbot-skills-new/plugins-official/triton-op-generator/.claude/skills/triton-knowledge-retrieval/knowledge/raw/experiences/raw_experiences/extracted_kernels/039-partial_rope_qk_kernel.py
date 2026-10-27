@triton.jit
def partial_rope_qk_kernel(
    Q,
    K,
    COS,
    SIN,
    Q_EMBED,
    K_EMBED,
    total_seq_len,
    NOPE_DIM_Q: tl.constexpr,
    NOPE_DIM_K: tl.constexpr,
    ROPE_DIM: tl.constexpr,
    BLOCK: tl.constexpr,
    NUM_Q_HEADS: tl.constexpr,
    NUM_K_HEADS: tl.constexpr,
    NUM_CORES: tl.constexpr,
):
    pid = tl.program_id(0)
    NUM_BLOCKS = tl.cdiv(total_seq_len, BLOCK)
    Q_HEAD_DIM = NOPE_DIM_Q + ROPE_DIM
    K_HEAD_DIM = NOPE_DIM_K + ROPE_DIM
    for seq_block_id in range(pid, NUM_BLOCKS, NUM_CORES):
        pos_offset = seq_block_id * BLOCK + tl.arange(0, BLOCK)
        pos_mask = pos_offset < total_seq_len
        half_dim: tl.constexpr = ROPE_DIM // 2
        cso_dim_offset_l = tl.arange(0, half_dim)
        cso_dim_offset_h = cso_dim_offset_l + half_dim
        seq_mask = pos_mask[:, None]
        cs_offset_l = pos_offset[:, None] * ROPE_DIM + cso_dim_offset_l[None, :]
        cs_offset_h = pos_offset[:, None] * ROPE_DIM + cso_dim_offset_h[None, :]

        cos_l = tl.load(COS + cs_offset_l, mask=seq_mask)
        cos_h = tl.load(COS + cs_offset_h, mask=seq_mask)
        sin_l = tl.load(SIN + cs_offset_l, mask=seq_mask)
        sin_h = tl.load(SIN + cs_offset_h, mask=seq_mask)

        partial_rope_qk_compute(
            X=Q,
            O=Q_EMBED,
            cos_l=cos_l,
            cos_h=cos_h,
            sin_l=sin_l,
            sin_h=sin_h,
            pos_offset=pos_offset,
            cso_dim_offset_l=cso_dim_offset_l,
            cso_dim_offset_h=cso_dim_offset_h,
            seq_mask=seq_mask,
            NOPE_DIM=NOPE_DIM_Q,
            ROPE_DIM=ROPE_DIM,
            half_dim=half_dim,
            NUM_HEADS=NUM_Q_HEADS,
            HEAD_DIM=Q_HEAD_DIM,
        )
        partial_rope_qk_compute(
            X=K,
            O=K_EMBED,
            cos_l=cos_l,
            cos_h=cos_h,
            sin_l=sin_l,
            sin_h=sin_h,
            pos_offset=pos_offset,
            cso_dim_offset_l=cso_dim_offset_l,
            cso_dim_offset_h=cso_dim_offset_h,
            seq_mask=seq_mask,
            NOPE_DIM=NOPE_DIM_K,
            ROPE_DIM=ROPE_DIM,
            half_dim=half_dim,
            NUM_HEADS=NUM_K_HEADS,
            HEAD_DIM=K_HEAD_DIM,
        )