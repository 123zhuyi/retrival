@triton.jit
def rotary_pos_emb_qr_kernel(
    X,
    COS,
    SIN,
    O,
    total_seq_len,
    NOPE_DIM: tl.constexpr,
    ROPE_DIM: tl.constexpr,
    BLOCK: tl.constexpr,
    NUM_HEADS: tl.constexpr,
    NUM_CORES: tl.constexpr,
):
    # q = matmul_qc_qr_out.view(total_seq_len, num_heads, q_head_dim)
    # q_nope, q_pe = torch.split(q, [qk_nope_head_dim, qk_rope_head_dim], dim=-1)
    # X是q，对q_pe实现rotary_pos_emb
    pid = tl.program_id(0)
    NUM_BLOCKS = tl.cdiv(total_seq_len, BLOCK)
    Q_HEAD_DIM = NOPE_DIM + ROPE_DIM
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

        x_dim_offset_l = NOPE_DIM + tl.arange(0, half_dim)
        x_dim_offset_h = x_dim_offset_l + half_dim
        for head_id in range(NUM_HEADS):
            x_base_offset = (pos_offset[:, None] * NUM_HEADS * Q_HEAD_DIM) + (
                head_id * Q_HEAD_DIM
            )
            x_l = tl.load(X + x_base_offset + x_dim_offset_l[None, :], mask=seq_mask)
            x_h = tl.load(X + x_base_offset + x_dim_offset_h[None, :], mask=seq_mask)
            o_l = x_l * cos_l - x_h * sin_l
            o_h = x_h * cos_h + x_l * sin_h
            o_base_offset = (pos_offset[:, None] * NUM_HEADS * ROPE_DIM) + (
                head_id * ROPE_DIM
            )
            tl.store(O + o_base_offset + cso_dim_offset_l[None, :], o_l, mask=seq_mask)
            tl.store(O + o_base_offset + cso_dim_offset_h[None, :], o_h, mask=seq_mask)