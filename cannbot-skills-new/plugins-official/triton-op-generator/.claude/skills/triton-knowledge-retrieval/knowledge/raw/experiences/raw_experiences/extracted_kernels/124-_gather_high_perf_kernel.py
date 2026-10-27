@triton.jit
def _gather_high_perf_kernel(
    # Pointers
    x_ptr,
    idx_ptr,
    out_ptr,
    stride_x_rows,
    stride_x_feats,
    stride_idx_rows,
    stride_idx_cols,
    stride_out_rows,
    stride_out_cols,
    num_indices: tl.constexpr,
    x_size: tl.constexpr,
):
    row_id = tl.program_id(0)

    offs_idx = tl.arange(0, num_indices)
    offs_x = tl.arange(0, x_size)

    # Load indices for this row
    idx_ptrs = idx_ptr + row_id * stride_idx_rows + offs_idx * stride_idx_cols
    indices = tl.load(idx_ptrs)

    # Load feature vector
    x_ptrs = x_ptr + row_id * stride_x_rows + offs_x * stride_x_feats
    x_vals = tl.load(x_ptrs)

    # Perform gather
    out_vals = tl.gather(x_vals, indices, 0)

    # Store result
    out_ptrs = out_ptr + row_id * stride_out_rows + offs_idx * stride_out_cols
    tl.store(out_ptrs, out_vals)