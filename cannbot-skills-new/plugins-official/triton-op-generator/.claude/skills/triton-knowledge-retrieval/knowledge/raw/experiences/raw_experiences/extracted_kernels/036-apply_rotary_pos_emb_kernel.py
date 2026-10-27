@triton.jit
def apply_rotary_pos_emb_kernel(
    X,
    COS,
    SIN,
    O,
    seq_len,
    stride_xs: tl.constexpr,
    stride_xh: tl.constexpr,
    stride_xd: tl.constexpr,
    DIM: tl.constexpr,
    BLOCK: tl.constexpr,
    NUM_HEADS: tl.constexpr,
    NUM_CORES: tl.constexpr,
):
    pid = tl.program_id(0)
    NUM_BLOCKS = tl.cdiv(seq_len, BLOCK)
    for seq_block_id in range(pid, NUM_BLOCKS, NUM_CORES):
        pos_offset = seq_block_id * BLOCK + tl.arange(0, BLOCK)
        pos_mask = pos_offset < seq_len
        half_dim: tl.constexpr = DIM // 2
        feat_offset_l = tl.arange(0, half_dim)
        feat_offset_h = half_dim + feat_offset_l
        seq_mask = pos_mask[:, None]
        cs_offset_l = pos_offset[:, None] * DIM + feat_offset_l[None, :]
        cs_offset_h = pos_offset[:, None] * DIM + feat_offset_h[None, :]

        cos_l = tl.load(COS + cs_offset_l, mask=seq_mask)
        cos_h = tl.load(COS + cs_offset_h, mask=seq_mask)
        sin_l = tl.load(SIN + cs_offset_l, mask=seq_mask)
        sin_h = tl.load(SIN + cs_offset_h, mask=seq_mask)

        for head_id in range(NUM_HEADS):
            base_offset = pos_offset[:, None] * stride_xs + head_id * stride_xh
            rotary_emb_compute(
                X,
                O,
                cos_h,
                cos_l,
                sin_h,
                sin_l,
                base_offset,
                feat_offset_l,
                feat_offset_h,
                seq_mask,
            )