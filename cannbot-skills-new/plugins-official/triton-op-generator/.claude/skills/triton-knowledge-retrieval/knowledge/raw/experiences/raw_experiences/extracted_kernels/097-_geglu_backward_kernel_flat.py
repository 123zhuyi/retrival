@triton.jit
def _geglu_backward_kernel_flat(
    dc_ptr, a_ptr, b_ptr, da_ptr, db_ptr, total_elements, BLOCK_SIZE: tl.constexpr, NUM_STAGES: tl.constexpr
):
    """
    High-performance GEGLU backward kernel using flatten 1D approach.

    Uses grid-stride loop pattern for optimal performance on NPU.
    """
    pid = tl.program_id(0)
    num_progs = tl.num_programs(0)
    start_idx = pid * BLOCK_SIZE
    stride = num_progs * BLOCK_SIZE

    # Constants for GELU tanh approximation
    sqrt_2_over_pi = 0.7978845608028654  # sqrt(2 / pi)
    gelu_coeff = 0.044715

    for idx in tl.range(start_idx, total_elements, stride, num_stages=NUM_STAGES):
        offsets = idx + tl.arange(0, BLOCK_SIZE)
        mask = offsets < total_elements

        dc = tl.load(dc_ptr + offsets, mask=mask, other=0.0)
        a = tl.load(a_ptr + offsets, mask=mask, other=0.0).to(tl.float32)
        b = tl.load(b_ptr + offsets, mask=mask, other=0.0)

        # recomputation to save memory
        a_cubed = a * a * a
        tanh_arg = sqrt_2_over_pi * (a + gelu_coeff * a_cubed)
        tanh_result = tanh(tanh_arg)
        geglu_a = 0.5 * a * (1 + tanh_result)
        geglu_a = geglu_a.to(dc.dtype).to(tl.float32)

        db = dc.cast(tl.float32) * geglu_a

        # Gradient w.r.t. a can be computed with:
        # b * (0.5 * (1 + tanh(z)) + 0.5 * a * (1 - tanh(z)^2) * (sqrt(2/pi) * (1 + 3 * 0.044715 * a^2)))
        # where z = sqrt(2/pi) * (a + 0.044715 * a^3)
        term1 = 0.5 * (1.0 + tanh_result)
        tanh_sq = tanh_result * tanh_result
        a_sq = a * a
        term2 = 0.5 * a * (1.0 - tanh_sq) * (sqrt_2_over_pi * (1.0 + 3.0 * gelu_coeff * a_sq))
        da = dc * b * (term1 + term2)

        tl.store(da_ptr + offsets, da, mask=mask)
        tl.store(db_ptr + offsets, db.to(dc.dtype), mask=mask)