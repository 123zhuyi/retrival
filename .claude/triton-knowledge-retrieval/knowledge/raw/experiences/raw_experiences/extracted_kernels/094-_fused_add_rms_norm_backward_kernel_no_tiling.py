@triton.jit
def _fused_add_rms_norm_backward_kernel_no_tiling(
    dY_ptr,
    dY_row_stride,
    dS_out_ptr,
    dS_out_row_stride,
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
    BLOCK_SIZE_M: tl.constexpr,
    BLOCK_SIZE_N: tl.constexpr,
    NUM_STAGES: tl.constexpr,
    has_dS_out: tl.constexpr,
):
    """
    NPU-optimized fused_add_rms_norm backward kernel for small n_cols (< 2048).

    Performance optimizations:
    1. Keep all data in registers, minimize conversions
    2. Reuse X_normalized (X * rstd) for both dX and dW
    3. Optimize computation order to reduce register pressure
    4. Combine operations where possible
    5. Use 2D vector loading to maximize UB utilization (e.g., (1,2048), (2,1024), (4,512))
    """
    pid = tl.program_id(0)
    num_progs = tl.num_programs(0)

    # Grid-stride loop setup for 2D blocks
    grid_stride = num_progs * BLOCK_SIZE_M
    num_iterations = tl.cdiv(n_rows, grid_stride)

    col_offsets = tl.arange(0, BLOCK_SIZE_N)
    col_mask = col_offsets < n_cols
    row_offsets = tl.arange(0, BLOCK_SIZE_M)

    # Load W once for all iterations
    W_row = tl.load(W_ptr + col_offsets, mask=col_mask, other=0.0)
    W_offset = W_row + offset

    # Grid-stride loop over row blocks
    for i in tl.range(num_iterations, num_stages=NUM_STAGES):
        row_idx = i * grid_stride + pid * BLOCK_SIZE_M + row_offsets
        row_mask = row_idx < n_rows
        block_mask = row_mask[:, None] & col_mask[None, :]

        dY_rows = tl.load(
            dY_ptr + row_idx[:, None] * dY_row_stride + col_offsets[None, :],
            mask=block_mask,
            other=0.0,
            eviction_policy="evict_first",
        )
        X_rows = tl.load(
            X_ptr + row_idx[:, None] * X_row_stride + col_offsets[None, :],
            mask=block_mask,
            other=0.0,
            eviction_policy="evict_first",
        )

        # Load rstd for all rows in the block
        rstd_rows = tl.load(RSTD_ptr + row_idx * RSTD_row_stride, mask=row_mask, other=0.0)

        # Convert X to fp32 once
        X_rows = X_rows.to(tl.float32)

        # Compute X_normalized (reused in dX and dW)
        X_normalized = X_rows * rstd_rows[:, None]

        # Compute m based on casting mode (optimized for each mode)
        if casting_mode == _CASTING_MODE_LLAMA:
            m_rows = (dY_rows * W_offset[None, :]).to(tl.float32)
            # For dW in Llama mode, we need X_normalized in original dtype
            X_normalized_for_dW = X_normalized.to(X_dtype)
        elif casting_mode == _CASTING_MODE_GEMMA:
            m_rows = dY_rows.to(tl.float32) * W_offset[None, :]
            X_normalized_for_dW = X_normalized
        else:
            m_rows = dY_rows * W_offset[None, :]
            X_normalized_for_dW = X_normalized

        # Compute sum(m * X) for correction factor
        sum_m_X = tl.sum(tl.where(block_mask, m_rows * X_rows, 0.0), axis=1)

        # Compute correction factor
        correction_factors = -(1.0 / n_cols) * rstd_rows * rstd_rows * sum_m_X

        # Compute dX = rstd * m + rstd * correction_factor * X
        dX_rows = rstd_rows[:, None] * m_rows + rstd_rows[:, None] * correction_factors[:, None] * X_rows

        # Add dS_out gradient if present
        if has_dS_out:
            dS_out_rows = tl.load(
                dS_out_ptr + row_idx[:, None] * dS_out_row_stride + col_offsets[None, :], mask=block_mask, other=0.0
            )
            dX_rows += dS_out_rows

        # Store dX
        tl.store(dX_ptr + row_idx[:, None] * dX_row_stride + col_offsets[None, :], dX_rows.to(X_dtype), mask=block_mask)

        # Compute dW contribution: dY * X_normalized
        dW_rows = (dY_rows * X_normalized_for_dW).to(tl.float32)

        # Accumulate to per-program dW buffer
        dW_row_ptr = dW_ptr + pid * dW_row_stride
        existing_dW = tl.load(dW_row_ptr + col_offsets, mask=col_mask, other=0.0)
        new_dW = existing_dW + tl.sum(tl.where(block_mask, dW_rows, 0.0), axis=0)
        tl.store(dW_row_ptr + col_offsets, new_dW, mask=col_mask)