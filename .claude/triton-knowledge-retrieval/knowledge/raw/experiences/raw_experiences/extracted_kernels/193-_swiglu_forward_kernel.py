@triton.jit
def _swiglu_forward_kernel(
    a_ptr, b_ptr, c_ptr, stride, n_cols: tl.constexpr, BLOCK_SIZE: tl.constexpr
):
    program_id = tl.program_id(0).to(tl.int64)

    # locate start index
    a_ptr += program_id * stride
    b_ptr += program_id * stride
    c_ptr += program_id * stride

    base_offsets = tl.arange(0, BLOCK_SIZE)
    for i in range(0, n_cols, BLOCK_SIZE):
        col_offsets = base_offsets + i
        mask = col_offsets < n_cols

        # sigmoid requires type float32
        a_row = tl.load(a_ptr + col_offsets, mask=mask, other=0).to(tl.float32)
        b_row = tl.load(b_ptr + col_offsets, mask=mask, other=0)
        c_row = silu(a_row).cast(b_row.dtype) * b_row
        tl.store(c_ptr + col_offsets, c_row, mask=mask)