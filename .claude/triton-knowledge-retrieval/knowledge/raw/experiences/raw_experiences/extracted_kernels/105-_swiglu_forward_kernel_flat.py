@triton.jit
def _swiglu_forward_kernel_flat(
    a_ptr, b_ptr, c_ptr, total_elements, BLOCK_SIZE: tl.constexpr, NUM_STAGES: tl.constexpr
):
    pid = tl.program_id(0)
    num_progs = tl.num_programs(0)

    # Grid-Stride Loop
    start_idx = pid * BLOCK_SIZE
    stride = num_progs * BLOCK_SIZE

    for idx in tl.range(start_idx, total_elements, stride, num_stages=NUM_STAGES):
        offsets = idx + tl.arange(0, BLOCK_SIZE)
        mask = offsets < total_elements

        a_val = tl.load(a_ptr + offsets, mask=mask, other=0.0).to(tl.float32)
        b_val = tl.load(b_ptr + offsets, mask=mask, other=0.0).to(tl.float32)
        res = (a_val * tl.sigmoid(a_val)) * b_val
        tl.store(c_ptr + offsets, res, mask=mask)