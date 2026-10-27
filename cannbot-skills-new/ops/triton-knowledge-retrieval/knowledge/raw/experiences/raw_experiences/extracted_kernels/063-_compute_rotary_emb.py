@triton.jit
def _compute_rotary_emb(
    x1,
    x2,
    cos,
    sin,
):
    cos = tl.expand_dims(cos, -2)
    sin = tl.expand_dims(sin, -2)
    o1 = x1 * cos - x2 * sin
    o2 = x2 * cos + x1 * sin
    return o1, o2