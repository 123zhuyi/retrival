@triton.jit
def _swiglu_backward_kernel(
    dc_ptr, a_ptr, b_ptr, stride, n_cols: tl.constexpr, BLOCK_SIZE: tl.constexpr
):
    program_id = tl.program_id(0).to(tl.int64)

    # locate start index
    dc_ptr += program_id * stride
    a_ptr += program_id * stride
    b_ptr += program_id * stride

    base_offsets = tl.arange(0, BLOCK_SIZE)
    for i in range(0, n_cols, BLOCK_SIZE):
        col_offsets = base_offsets + i
        mask = col_offsets < n_cols

        dc_row = tl.load(dc_ptr + col_offsets, mask=mask, other=0)
        # sigmoid requires type float32
        a_row = tl.load(a_ptr + col_offsets, mask=mask, other=0).to(tl.float32)
        b_row = tl.load(b_ptr + col_offsets, mask=mask, other=0)

        # recomputation to save memory
        sig_a = tl.sigmoid(a_row)
        silu_a = a_row * sig_a
        db_row = dc_row * silu_a
        da_row = dc_row * (silu_a * (1 - sig_a) + sig_a) * b_row

        tl.store(a_ptr + col_offsets, da_row, mask=mask)
        tl.store(b_ptr + col_offsets, db_row, mask=mask)