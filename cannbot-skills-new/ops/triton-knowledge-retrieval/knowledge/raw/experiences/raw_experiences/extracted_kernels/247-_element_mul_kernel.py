@triton.autotune(
    configs=[
        triton.Config({"BLOCK_SIZE_M": 8, "BLOCK_SIZE_N": 1024}),
        triton.Config({"BLOCK_SIZE_M": 12, "BLOCK_SIZE_N": 1024}),
        triton.Config({"BLOCK_SIZE_M": 16, "BLOCK_SIZE_N": 1024}),
        triton.Config({"BLOCK_SIZE_M": 24, "BLOCK_SIZE_N": 1024}),
        triton.Config({"BLOCK_SIZE_M": 32, "BLOCK_SIZE_N": 1024}),
        triton.Config({"BLOCK_SIZE_M": 8, "BLOCK_SIZE_N": 2048}),
        triton.Config({"BLOCK_SIZE_M": 12, "BLOCK_SIZE_N": 2048}),
        triton.Config({"BLOCK_SIZE_M": 16, "BLOCK_SIZE_N": 2048}),
        triton.Config({"BLOCK_SIZE_M": 24, "BLOCK_SIZE_N": 2048}),
        triton.Config({"BLOCK_SIZE_M": 32, "BLOCK_SIZE_N": 2048}),
        triton.Config({"BLOCK_SIZE_M": 8, "BLOCK_SIZE_N": 4096}),
        triton.Config({"BLOCK_SIZE_M": 12, "BLOCK_SIZE_N": 4096}),
        triton.Config({"BLOCK_SIZE_M": 16, "BLOCK_SIZE_N": 4096}),
        triton.Config({"BLOCK_SIZE_M": 24, "BLOCK_SIZE_N": 4096}),
        triton.Config({"BLOCK_SIZE_M": 32, "BLOCK_SIZE_N": 4096}),
    ],
    key=["n_cols"],
    restore_value=["X_ptr"],
)
@libentry()
@triton.jit
def _element_mul_kernel(
    X_ptr,
    grad_output_ptr,
    n_rows,
    n_cols,
    X_stride_row,  # Stride for moving between rows
    # X_stride_col is assumed to be 1 for contiguous rows
    BLOCK_SIZE_M: tl.constexpr,
    BLOCK_SIZE_N: tl.constexpr,
):
    """
    Multiplies each element of a 2D tensor X with a scalar grad_output.
    This kernel is adapted for NPU-style parallelism with a fixed grid size.

    Parameters:
    X_ptr: Pointer to the input 2D tensor.
    grad_output_ptr: Pointer to the scalar gradient output value.
    n_rows (int): The number of rows in the input tensor.
    n_cols (int): The number of columns in the input tensor.
    X_stride_row (int): The stride to move from one row to the next.
    BLOCK_SIZE_M (int): The number of rows to process in one tile.
    BLOCK_SIZE_N (int): The number of columns to process in one tile.
    """
    pid = tl.program_id(0)
    num_programs = tl.num_programs(0)

    grad_output = tl.load(grad_output_ptr)

    for row_block_start in range(pid * BLOCK_SIZE_M, n_rows, num_programs * BLOCK_SIZE_M):
        row_offsets = row_block_start + tl.arange(0, BLOCK_SIZE_M)
        row_mask = row_offsets < n_rows

        for col_block_start in range(0, n_cols, BLOCK_SIZE_N):
            col_offsets = col_block_start + tl.arange(0, BLOCK_SIZE_N)
            col_mask = col_offsets < n_cols

            X_ptr_tile = X_ptr + row_offsets[:, None] * X_stride_row + col_offsets[None, :]

            mask_2d = row_mask[:, None] & col_mask[None, :]

            X_tile = tl.load(X_ptr_tile, mask=mask_2d, other=0.0)

            result_tile = X_tile * grad_output

            tl.store(X_ptr_tile, result_tile, mask=mask_2d)