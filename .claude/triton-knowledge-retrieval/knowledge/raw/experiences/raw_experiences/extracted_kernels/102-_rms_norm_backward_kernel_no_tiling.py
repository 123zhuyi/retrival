@triton.jit
def _rms_norm_backward_kernel_no_tiling(
    dY_ptr,
    dY_row_stride,
    dX_ptr,
    dX_row_stride,
    X_ptr,
    X_row_stride,
    X_dtype: tl.constexpr,
    W_ptr,
    RSTD_ptr,
    RSTD_row_stride,
    dW_ptr,
    dW_row_stride,
    n_rows,
    n_cols,
    offset,
    casting_mode: tl.constexpr,
    elementwise_affine: tl.constexpr,
    BLOCK_SIZE: tl.constexpr,
    NUM_STAGES: tl.constexpr,
):
    """
    NPU-optimized rms_norm backward kernel for small n_cols (< 2048).

    Each program processes multiple rows using grid-stride loop.
    For each row, load the entire row at once without column tiling.
    """
    pid = tl.program_id(0)
    num_progs = tl.num_programs(0)

    # Load entire row at once
    col_offsets = tl.arange(0, BLOCK_SIZE)
    mask = col_offsets < n_cols

    # Grid-stride loop over rows
    for row_idx in tl.range(pid, n_rows, num_progs, num_stages=NUM_STAGES):
        # Base pointers for this row
        dY_row_ptr = dY_ptr + row_idx * dY_row_stride
        dX_row_ptr = dX_ptr + row_idx * dX_row_stride
        X_row_ptr = X_ptr + row_idx * X_row_stride
        RSTD_row_ptr = RSTD_ptr + row_idx * RSTD_row_stride

        # Load rstd for this row
        rstd = tl.load(RSTD_row_ptr)

        # Load entire row
        dY_row = tl.load(dY_row_ptr + col_offsets, mask=mask, other=0.0, eviction_policy="evict_first")
        X_row = tl.load(X_row_ptr + col_offsets, mask=mask, other=0.0, eviction_policy="evict_first")

        # Convert to fp32 for computation
        X_row = X_row.to(tl.float32)

        if elementwise_affine:
            W_row = tl.load(W_ptr + col_offsets, mask=mask, other=0.0, eviction_policy="evict_first")
            W_offset = W_row + offset

            # Compute m based on casting mode
            if casting_mode == _CASTING_MODE_LLAMA:
                m = (dY_row * W_offset).to(tl.float32)
            elif casting_mode == _CASTING_MODE_GEMMA:
                dY_row_fp32 = dY_row.to(tl.float32)
                m = dY_row_fp32 * W_offset
            else:
                m = dY_row * W_offset
        else:
            # Compute m based on casting mode
            if casting_mode == _CASTING_MODE_LLAMA:
                m = dY_row.to(tl.float32)
            elif casting_mode == _CASTING_MODE_GEMMA:
                m = dY_row.to(tl.float32)
            else:
                m = dY_row

        # Compute sum(m * X) for the correction term
        sum_m_X = tl.sum(tl.where(mask, m * X_row, 0.0))

        # Compute the correction factor
        correction_factor = -(1.0 / n_cols) * rstd * rstd * sum_m_X

        # Compute dX
        dX_row = rstd * m + rstd * correction_factor * X_row

        # Store dX
        tl.store(dX_row_ptr + col_offsets, dX_row.to(X_dtype), mask=mask)

        if elementwise_affine:
            # Compute dW contribution (accumulate per program)
            if casting_mode == _CASTING_MODE_LLAMA:
                dW_row = dY_row * (X_row * rstd).to(X_dtype)
            else:
                dW_row = dY_row * (X_row * rstd)

            # Atomic add to dW_ptr (each program writes to its own row)
            dW_row_ptr = dW_ptr + pid * dW_row_stride

            # Load existing dW, add contribution, store back
            existing_dW = tl.load(dW_row_ptr + col_offsets, mask=mask, other=0.0)
            new_dW = existing_dW + dW_row.to(tl.float32)
            tl.store(dW_row_ptr + col_offsets, new_dW, mask=mask)