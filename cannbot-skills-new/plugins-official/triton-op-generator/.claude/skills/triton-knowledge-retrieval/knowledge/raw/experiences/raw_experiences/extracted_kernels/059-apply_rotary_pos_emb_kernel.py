@triton.jit
def apply_rotary_pos_emb_kernel(
    X,
    COS,
    SIN,
    O,
    seq_len,
    DIM: tl.constexpr,
    CS_DIM: tl.constexpr,
    BLOCK: tl.constexpr,
    NUM_CORES: tl.constexpr,
):
    pid = tl.program_id(0)
    NUM_BLOCKS = tl.cdiv(seq_len, BLOCK)
    for seq_block_id in range(pid, NUM_BLOCKS, NUM_CORES):
        pos_offset = seq_block_id * BLOCK + tl.arange(0, BLOCK)
        pos_mask = pos_offset < seq_len
        half_dim: tl.constexpr = DIM // 2
        feat_offset_ll = tl.arange(0, CS_DIM)
        feat_offset_lh = tl.arange(0, half_dim - CS_DIM) + CS_DIM

        feat_offset_hl = half_dim + feat_offset_ll
        feat_offset_hh = half_dim + feat_offset_lh
        seq_mask = pos_mask[:, None]
        cs_range = tl.arange(0, CS_DIM)
        cs_offset_l = pos_offset[:, None] * CS_DIM + cs_range[None, :]
        cos_l = tl.load(COS + cs_offset_l, mask=seq_mask)
        sin_l = tl.load(SIN + cs_offset_l, mask=seq_mask)
        cos_h = cos_l
        sin_h = sin_l
        base_offset = pos_offset[:, None] * DIM
        x_ll = tl.load(X + base_offset + feat_offset_ll[None, :], mask=seq_mask)
        x_lh = tl.load(X + base_offset + feat_offset_lh[None, :], mask=seq_mask)
        x_hl = tl.load(X + base_offset + feat_offset_hl[None, :], mask=seq_mask)
        x_hh = tl.load(X + base_offset + feat_offset_hh[None, :], mask=seq_mask)
        o_ll = x_ll * cos_l - x_hl * sin_l
        o_lh = x_lh - x_hh

        o_hl = x_hl * cos_h + x_ll * sin_h
        o_hh = x_hh + x_lh

        tl.store(O + base_offset + feat_offset_ll[None, :], o_ll, mask=seq_mask)
        tl.store(O + base_offset + feat_offset_lh[None, :], o_lh, mask=seq_mask)
        tl.store(O + base_offset + feat_offset_hl[None, :], o_hl, mask=seq_mask)
        tl.store(O + base_offset + feat_offset_hh[None, :], o_hh, mask=seq_mask)