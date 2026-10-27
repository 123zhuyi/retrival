@libentry()
@triton.jit
def sum_and_scale(
    inp_ptr,
    out_ptr,
    N,
    scalebyw: tl.constexpr,
    BLOCK_N: tl.constexpr = 128,
    scale=1.0,
    mean_num=None,
):
    mid_sum = tl.zeros(
        [
            BLOCK_N,
        ],
        dtype=tl.float32,
    )
    if scalebyw:
        mid_wgt = tl.zeros(
            [
                BLOCK_N,
            ],
            dtype=tl.float32,
        )
        for off in range(0, N, BLOCK_N):
            offset = off + tl.arange(0, BLOCK_N)
            inp_ptrs = inp_ptr + offset
            mask = offset < N
            inp_vals = tl.load(inp_ptrs, mask=mask, other=0.0)
            mid_sum += inp_vals
            wgt_ptrs = scale + offset
            wgt_vals = tl.load(wgt_ptrs, mask=mask, other=0.0)
            mid_wgt += wgt_vals
        out_val = tl.sum(mid_sum)
        scale_val = tl.sum(mid_wgt)
        tl.store(mean_num, scale_val)
    else:
        for off in range(0, N, BLOCK_N):
            offset = off + tl.arange(0, BLOCK_N)
            inp_ptrs = inp_ptr + offset
            mask = offset < N
            inp_vals = tl.load(inp_ptrs, mask=mask, other=0.0)
            mid_sum += inp_vals
        out_val = tl.sum(mid_sum)
        scale_val = scale
    out_val /= scale_val
    tl.store(out_ptr, out_val)