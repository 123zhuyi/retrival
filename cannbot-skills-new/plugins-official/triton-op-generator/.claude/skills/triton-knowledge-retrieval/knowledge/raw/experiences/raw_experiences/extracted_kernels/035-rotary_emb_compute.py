@triton.jit
def rotary_emb_compute(
    X,
    X_EMBED,
    cos_h,
    cos_l,
    sin_h,
    sin_l,
    base_offset,
    feat_offset_l,
    feat_offset_h,
    seq_mask,
):
    x_l = tl.load(X + base_offset + feat_offset_l[None, :], mask=seq_mask)
    x_h = tl.load(X + base_offset + feat_offset_h[None, :], mask=seq_mask)
    o_l = x_l * cos_l - x_h * sin_l
    o_h = x_h * cos_h + x_l * sin_h
    tl.store(X_EMBED + base_offset + feat_offset_l[None, :], o_l, mask=seq_mask)
    tl.store(X_EMBED + base_offset + feat_offset_h[None, :], o_h, mask=seq_mask)