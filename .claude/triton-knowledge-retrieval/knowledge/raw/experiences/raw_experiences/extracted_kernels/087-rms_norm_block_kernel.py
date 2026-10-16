@triton.jit
def rms_norm_block_kernel(
    input,
    weight,
    output,
    n_rows,
    input_row_stride: tl.constexpr,
    eps: tl.constexpr,
    N_COLS: tl.constexpr,
    BLOCK: tl.constexpr,
    NUM_CORES: tl.constexpr,
):
    pid = tl.program_id(0)
    cols_offsets = tl.arange(0, N_COLS)
    w = tl.load(weight + cols_offsets)
    w = tl.expand_dims(w, 0)
    w = tl.broadcast_to(w, (BLOCK, N_COLS))
    NUM_BLOCKS = tl.cdiv(n_rows, BLOCK)
    for row_block_id in range(pid, NUM_BLOCKS, NUM_CORES):
        pos_offset = row_block_id * BLOCK + tl.arange(0, BLOCK)
        pos_mask = (pos_offset < n_rows)[:, None]
        base_offset = pos_offset[:, None] * input_row_stride + cols_offsets[None, :]
        x = tl.load(input + base_offset, mask=pos_mask)
        xf = x.to(tl.float32)
        var = tl.sum(xf * xf, 1) / N_COLS
        qrt = tl.expand_dims(tl.math.rsqrt(var + eps), 1)
        out = xf * tl.broadcast_to(qrt, (BLOCK, N_COLS))
        out = w * out.to(x.dtype)
        tl.store(output + base_offset, out, mask=pos_mask)