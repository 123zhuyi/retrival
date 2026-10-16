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
    restore_value=["dy", "dx"],
)
@libentry()
@triton.jit
def _silu_bwd_kernel(
    dy,
    x,
    dx,
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

            dy_ptrs = dy + rows_off[:, None] * stride_row + cols_off[None, :]
            x_ptrs = x + rows_off[:, None] * stride_row + cols_off[None, :]
            dx_ptrs = dx + rows_off[:, None] * stride_row + cols_off[None, :]

            dy_chunk = tl.load(dy_ptrs, mask=block_mask, other=0.0)
            x_chunk = tl.load(x_ptrs, mask=block_mask, other=0.0)

            x_f32 = x_chunk.to(tl.float32)
            sigmoid_x = tl.sigmoid(x_f32)

            dsilu_dx = sigmoid_x * (1 + x_f32 * (1 - sigmoid_x))

            dx_chunk = dy_chunk * dsilu_dx.to(dy_chunk.dtype)

            tl.store(dx_ptrs, dx_chunk, mask=block_mask)