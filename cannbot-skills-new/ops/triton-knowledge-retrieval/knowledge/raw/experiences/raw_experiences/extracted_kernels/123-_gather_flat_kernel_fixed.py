@libentry()
@triton.heuristics({"BLOCK_SIZE": lambda args: 4096})
@triton.jit
def _gather_flat_kernel_fixed(
    inp,
    index,
    out,
    base_offset,
    inp_dim_stride,
    N,
    BLOCK_SIZE: tl.constexpr,
):
    pid = tl.program_id(0)
    offset = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    mask = offset < N

    cur_index = tl.load(index + offset, mask=mask, other=0)
    base = tl.load(base_offset + offset, mask=mask, other=0)

    inp_offset = base + cur_index * inp_dim_stride

    val = tl.load(inp + inp_offset, mask=mask, other=0)
    tl.store(out + offset, val, mask=mask)