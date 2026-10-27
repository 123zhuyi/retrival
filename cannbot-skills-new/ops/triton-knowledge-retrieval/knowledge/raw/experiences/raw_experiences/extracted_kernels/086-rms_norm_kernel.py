@triton.jit
def rms_norm_kernel(
    input,
    weight,
    output,
    n_rows,
    input_row_stride: tl.constexpr,
    eps: tl.constexpr,
    N_COLS: tl.constexpr,
    NUM_CORES: tl.constexpr,
):
    pid = tl.program_id(0)
    offsets = tl.arange(0, N_COLS)
    w = tl.load(weight + offsets)
    for row_idx in range(pid, n_rows, NUM_CORES):
        x_ptr = input + row_idx * input_row_stride
        x = tl.load(x_ptr + offsets)
        xf = x.to(tl.float32)
        var = tl.sum(xf * xf, 0) / N_COLS
        out = xf * tl.math.rsqrt(var + eps)
        out = w * out.to(x.dtype)
        out_ptr = output + row_idx * input_row_stride
        tl.store(out_ptr + offsets, out)