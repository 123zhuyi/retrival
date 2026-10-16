@triton.jit
def _fused_add_rms_norm_forward_kernel_no_tiling(
    Y_ptr,
    Y_row_stride,
    S_ptr,  # output residual
    S_row_stride,
    X_ptr,
    X_row_stride,
    R_ptr,  # input residual
    R_row_stride,
    W_ptr,
    RSTD_ptr,
    RSTD_row_stride,
    n_rows,
    n_cols,
    eps,
    offset,
    casting_mode: tl.constexpr,
    X_DTYPE: tl.constexpr,
    BLOCK_SIZE_M: tl.constexpr,
    BLOCK_SIZE_N: tl.constexpr,
    NUM_STAGES: tl.constexpr,
):
    """
    NPU-optimized fused_add_rms_norm forward kernel for small n_cols (< 2048).

    Performance optimizations:
    1. Keep S_row in registers, avoid reload from memory
    2. Minimize type conversions by careful ordering
    3. Use optimal cache policies
    4. Preload W_row while computing rstd (instruction-level parallelism)
    5. Use 2D vector loading to maximize UB utilization (e.g., (1,2048), (2,1024), (4,512))

    Used when n_cols < 2048 to avoid the overhead of column blocking.
    """
    pid = tl.program_id(0)
    num_progs = tl.num_programs(0)

    if casting_mode == _CASTING_MODE_NONE:
        eps = eps.to(X_DTYPE)
        offset = offset.to(X_DTYPE)

    # Grid-stride loop setup for 2D blocks
    grid_stride = num_progs * BLOCK_SIZE_M
    num_iterations = tl.cdiv(n_rows, grid_stride)

    col_offsets = tl.arange(0, BLOCK_SIZE_N)
    col_mask = col_offsets < n_cols
    row_offsets = tl.arange(0, BLOCK_SIZE_M)

    W_row = tl.load(W_ptr + col_offsets, mask=col_mask, other=0.0)

    # Grid-stride loop over row blocks
    for i in tl.range(num_iterations, num_stages=NUM_STAGES):
        row_idx = i * grid_stride + pid * BLOCK_SIZE_M + row_offsets
        row_mask = row_idx < n_rows
        block_mask = row_mask[:, None] & col_mask[None, :]

        # Load multiple rows at once using 2D indexing
        X_rows = tl.load(
            X_ptr + row_idx[:, None] * X_row_stride + col_offsets[None, :],
            mask=block_mask,
            other=0.0,
            eviction_policy="evict_first",
        )
        R_rows = tl.load(
            R_ptr + row_idx[:, None] * R_row_stride + col_offsets[None, :],
            mask=block_mask,
            other=0.0,
            eviction_policy="evict_first",
        )
        S_rows = X_rows + R_rows

        # Compute sum_square for all rows
        if casting_mode == _CASTING_MODE_LLAMA or casting_mode == _CASTING_MODE_GEMMA:
            S_rows = S_rows.to(tl.float32)

        sum_squares = tl.sum(tl.where(block_mask, S_rows * S_rows, 0.0), axis=1)

        # Compute rstd for all rows
        mean_squares = sum_squares / n_cols
        rstd_rows = rsqrt(mean_squares + eps)

        # Store S_rows and rstd_rows
        tl.store(
            S_ptr + row_idx[:, None] * S_row_stride + col_offsets[None, :],
            S_rows,
            mask=block_mask,
            cache_modifier=".cg",
        )
        tl.store(RSTD_ptr + row_idx * RSTD_row_stride, rstd_rows, mask=row_mask)

        # Normalize and apply weight - optimized for each casting mode
        if casting_mode == _CASTING_MODE_GEMMA:
            Y_rows = ((S_rows * rstd_rows[:, None]) * (offset + W_row[None, :])).to(X_DTYPE)
        elif casting_mode == _CASTING_MODE_LLAMA:
            S_normalized = (S_rows * rstd_rows[:, None]).to(X_DTYPE)
            Y_rows = S_normalized * (offset + W_row[None, :])
        else:
            Y_rows = (S_rows * rstd_rows[:, None]) * (offset + W_row[None, :])

        # Store results
        tl.store(Y_ptr + row_idx[:, None] * Y_row_stride + col_offsets[None, :], Y_rows, mask=block_mask)