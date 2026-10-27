@triton.jit
def _swiglu_backward_kernel_flat(
    dc_ptr, a_ptr, b_ptr, da_ptr, db_ptr, total_elements, BLOCK_SIZE: tl.constexpr, NUM_STAGES: tl.constexpr
):
    pid = tl.program_id(0)
    num_progs = tl.num_programs(0)
    start_idx = pid * BLOCK_SIZE
    stride = num_progs * BLOCK_SIZE

    for idx in tl.range(start_idx, total_elements, stride, num_stages=NUM_STAGES):
        offsets = idx + tl.arange(0, BLOCK_SIZE)
        mask = offsets < total_elements

        dc = tl.load(dc_ptr + offsets, mask=mask, other=0.0).to(tl.float32)
        a = tl.load(a_ptr + offsets, mask=mask, other=0.0).to(tl.float32)
        b = tl.load(b_ptr + offsets, mask=mask, other=0.0).to(tl.float32)

        sig_a = tl.sigmoid(a)
        silu_a = a * sig_a
        term1 = silu_a * (1.0 - sig_a) + sig_a

        db = dc * silu_a
        da = dc * b * term1

        tl.store(da_ptr + offsets, da, mask=mask)
        tl.store(db_ptr + offsets, db, mask=mask)