@triton.jit
def _rms_norm_forward_kernel_no_tiling(
    Y_ptr,
    Y_row_stride,
    X_ptr,
    X_row_stride,
    W_ptr,
    RSTD_ptr,
    RSTD_row_stride,
    n_rows,
    n_cols,
    eps,
    offset,
    casting_mode: tl.constexpr,
    elementwise_affine: tl.constexpr,
    X_DTYPE: tl.constexpr,
    BLOCK_SIZE: tl.constexpr,
    NUM_STAGES: tl.constexpr,
):
    """
    NPU-optimized rms_norm forward kernel for small n_cols (< 2048).

    This kernel loads entire rows without column tiling:
    1. Each program handles multiple rows using grid-stride loop
    2. For each row, load the entire row at once (no column chunking)
    3. Grid size is limited to NPU core count to avoid resource overflow

    Used when n_cols < 2048 to avoid the overhead of column blocking.
    """
    pid = tl.program_id(0)
    num_progs = tl.num_programs(0)

    if casting_mode == _CASTING_MODE_NONE:
        eps = eps.to(X_DTYPE)
        offset = offset.to(X_DTYPE)

    # Load entire row at once
    col_offsets = tl.arange(0, BLOCK_SIZE)
    mask = col_offsets < n_cols

    # Grid-stride loop over rows
    for row_idx in tl.range(pid, n_rows, num_progs, num_stages=NUM_STAGES):
        Y_row_ptr = Y_ptr + row_idx * Y_row_stride
        X_row_ptr = X_ptr + row_idx * X_row_stride
        RSTD_row_ptr = RSTD_ptr + row_idx * RSTD_row_stride

        # Load entire row
        X_row = tl.load(X_row_ptr + col_offsets, mask=mask, other=0.0, eviction_policy="evict_first")

        if casting_mode == _CASTING_MODE_LLAMA or casting_mode == _CASTING_MODE_GEMMA:
            X_row = X_row.to(tl.float32)

        # Compute sum of squares for the entire row
        sum_square = tl.sum(tl.where(mask, X_row * X_row, 0.0))

        # Compute rstd for this row
        mean_square = sum_square / n_cols
        rstd = rsqrt(mean_square + eps)

        # Store rstd
        tl.store(RSTD_row_ptr, rstd)

        # Load row again for normalization
        X_row = tl.load(X_row_ptr + col_offsets, mask=mask, other=0.0, cache_modifier=".ca")

        if elementwise_affine:
            W_row = tl.load(W_ptr + col_offsets, mask=mask, other=0.0)

        # Apply casting based on mode
        if casting_mode == _CASTING_MODE_GEMMA:
            X_row = X_row.to(tl.float32)
            if elementwise_affine:
                W_row = W_row.to(tl.float32)
        elif casting_mode == _CASTING_MODE_LLAMA:
            X_row = X_row.to(tl.float32)

        # Normalize
        X_row = X_row * rstd

        # Cast back for Llama mode before weight multiplication
        if casting_mode == _CASTING_MODE_LLAMA:
            X_row = X_row.to(X_DTYPE)

        # Apply weight
        if elementwise_affine:
            Y_row = X_row * (offset + W_row)
        else:
            Y_row = X_row

        # Cast back for Gemma mode
        if casting_mode == _CASTING_MODE_GEMMA:
            Y_row = Y_row.to(X_DTYPE)

        # Store result
        tl.store(Y_row_ptr + col_offsets, Y_row, mask=mask)