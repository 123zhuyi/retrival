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
)
@libentry()
@triton.jit
def _swiglu_fwd_kernel(
    a,
    b,
    c,
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

            a_ptrs = a + rows_off[:, None] * stride_row + cols_off[None, :]
            b_ptrs = b + rows_off[:, None] * stride_row + cols_off[None, :]
            c_ptrs = c + rows_off[:, None] * stride_row + cols_off[None, :]

            a_chunk = tl.load(a_ptrs, mask=block_mask, other=0.0)
            b_chunk = tl.load(b_ptrs, mask=block_mask, other=0.0)

            a_f32 = a_chunk.to(tl.float32)
            silu_a = silu(a_f32)

            c_chunk = silu_a.to(a_chunk.dtype) * b_chunk

            tl.store(c_ptrs, c_chunk, mask=block_mask)