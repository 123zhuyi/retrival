@triton.jit
def partial_rope_qk_compute(
    X,
    O,
    cos_l,
    cos_h,
    sin_l,
    sin_h,
    pos_offset,
    cso_dim_offset_l,
    cso_dim_offset_h,
    seq_mask,
    NOPE_DIM: tl.constexpr,
    ROPE_DIM: tl.constexpr,
    half_dim: tl.constexpr,
    NUM_HEADS: tl.constexpr,
    HEAD_DIM: tl.constexpr,
):
    x_dim_offset_l = NOPE_DIM + tl.arange(0, half_dim)
    x_dim_offset_h = x_dim_offset_l + half_dim
    for head_id in range(NUM_HEADS):
        x_base_offset = (pos_offset[:, None] * NUM_HEADS * HEAD_DIM) + (
            head_id * HEAD_DIM
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