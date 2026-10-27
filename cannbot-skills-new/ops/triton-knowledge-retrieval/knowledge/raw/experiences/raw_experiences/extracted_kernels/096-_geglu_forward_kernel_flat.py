@triton.jit
def _geglu_forward_kernel_flat(a_ptr, b_ptr, c_ptr, total_elements, BLOCK_SIZE: tl.constexpr, NUM_STAGES: tl.constexpr):
    """
    High-performance GEGLU forward kernel using flatten 1D approach.

    Uses grid-stride loop pattern for optimal performance on NPU.
    """
    pid = tl.program_id(0)
    num_progs = tl.num_programs(0)

    # Grid-Stride Loop
    start_idx = pid * BLOCK_SIZE
    stride = num_progs * BLOCK_SIZE

    # Constants for GELU tanh approximation
    sqrt_2_over_pi = 0.7978845608028654  # sqrt(2 / pi)
    gelu_coeff = 0.044715

    for idx in tl.range(start_idx, total_elements, stride, num_stages=NUM_STAGES):
        offsets = idx + tl.arange(0, BLOCK_SIZE)
        mask = offsets < total_elements

        a_val = tl.load(a_ptr + offsets, mask=mask, other=0.0).to(tl.float32)
        b_val = tl.load(b_ptr + offsets, mask=mask, other=0.0)

        # tanh approximation form of GELU is computed with:
        # 0.5 * a * (1 + tanh(sqrt(2 / pi) * (a + 0.044715 * a^3)))
        a_cubed = a_val * a_val * a_val
        tanh_arg = sqrt_2_over_pi * (a_val + gelu_coeff * a_cubed)
        tanh_result = tanh(tanh_arg)
        geglu_a = 0.5 * a_val * (1.0 + tanh_result)
        c_row = geglu_a.cast(b_val.dtype) * b_val
        tl.store(c_ptr + offsets, c_row, mask=mask)