@triton.autotune(
    configs=[
        triton.Config({"BLOCK_SIZE_M": 1}),
        triton.Config({"BLOCK_SIZE_M": 2}),
        triton.Config({"BLOCK_SIZE_M": 4}),
        triton.Config({"BLOCK_SIZE_M": 8}),
        triton.Config({"BLOCK_SIZE_M": 12}),
        triton.Config({"BLOCK_SIZE_M": 16}),
        triton.Config({"BLOCK_SIZE_M": 20}),
        triton.Config({"BLOCK_SIZE_M": 24}),
        triton.Config({"BLOCK_SIZE_M": 32}),
    ],
    key=["n_rows", "n_cols"],
    restore_value=["dc", "da", "db"],
)
@libentry()
@triton.jit
def _swiglu_bwd_kernel(
    dc,
    a,
    b,
    da,
    db,
    stride_row,
    n_rows,
    n_cols,
    BLOCK_SIZE_N: tl.constexpr,
    BLOCK_SIZE_M: tl.constexpr,
):
    pid = tl.program_id(axis=0)
    grid_size = tl.num_programs(axis=0)

    num_row_tasks = (n_rows + BLOCK_SIZE_M - 1) // BLOCK_SIZE_M

    for row_task_id in range(pid, num_row_tasks, grid_size):
        block_start_row = row_task_id * BLOCK_SIZE_M
        rows_off = block_start_row + tl.arange(0, BLOCK_SIZE_M)
        rows_mask = rows_off < n_rows

        for col_offset in range(0, n_cols, BLOCK_SIZE_N):
            cols_off = col_offset + tl.arange(0, BLOCK_SIZE_N)
            cols_mask = cols_off < n_cols
            block_mask = rows_mask[:, None] & cols_mask[None, :]

            dc_ptrs = dc + rows_off[:, None] * stride_row + cols_off[None, :]
            a_ptrs = a + rows_off[:, None] * stride_row + cols_off[None, :]
            b_ptrs = b + rows_off[:, None] * stride_row + cols_off[None, :]
            da_ptrs = da + rows_off[:, None] * stride_row + cols_off[None, :]
            db_ptrs = db + rows_off[:, None] * stride_row + cols_off[None, :]

            dc_chunk = tl.load(dc_ptrs, mask=block_mask, other=0.0)
            a_chunk = tl.load(a_ptrs, mask=block_mask, other=0.0)
            b_chunk = tl.load(b_ptrs, mask=block_mask, other=0.0)

            a_f32 = a_chunk.to(tl.float32)
            sigmoid_a = tl.sigmoid(a_f32)
            silu_a = a_f32 * sigmoid_a

            db_chunk = dc_chunk * silu_a.to(dc_chunk.dtype)

            da_factor = silu_a * (1 - sigmoid_a) + sigmoid_a
            da_chunk = dc_chunk * b_chunk * da_factor.to(dc_chunk.dtype)

            tl.store(da_ptrs, da_chunk, mask=block_mask)
            tl.store(db_ptrs, db_chunk, mask=block_mask)