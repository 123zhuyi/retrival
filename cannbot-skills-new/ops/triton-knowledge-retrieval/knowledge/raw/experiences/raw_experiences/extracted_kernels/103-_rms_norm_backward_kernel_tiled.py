@triton.jit
def _rms_norm_backward_kernel_tiled(
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
    NPU-optimized rms_norm backward kernel for large n_cols (>= 2048).

    Each program processes multiple rows using grid-stride loop.
    For each row, we process columns in blocks to avoid UB overflow.
    """
    pid = tl.program_id(0)
    num_progs = tl.num_programs(0)

    # Initialize dW accumulator (per-program, will be reduced later)
    num_col_blocks = tl.cdiv(n_cols, BLOCK_SIZE)
    offsets = tl.arange(0, BLOCK_SIZE)

    # Grid-stride loop over rows
    for row_idx in tl.range(pid, n_rows, num_progs, num_stages=NUM_STAGES):
        # Base pointers for this row
        dY_row_ptr = dY_ptr + row_idx * dY_row_stride
        dX_row_ptr = dX_ptr + row_idx * dX_row_stride
        X_row_ptr = X_ptr + row_idx * X_row_stride
        RSTD_row_ptr = RSTD_ptr + row_idx * RSTD_row_stride

        # Load rstd for this row
        rstd = tl.load(RSTD_row_ptr)

        # First pass: compute sum(m * X) for the correction term
        sum_m_X = 0.0

        for col_block_idx in range(num_col_blocks):
            col_start = col_block_idx * BLOCK_SIZE
            col_offsets = col_start + offsets
            mask = col_offsets < n_cols

            dY_block = tl.load(dY_row_ptr + col_offsets, mask=mask, other=0.0, eviction_policy="evict_first")
            X_block = tl.load(X_row_ptr + col_offsets, mask=mask, other=0.0, eviction_policy="evict_first")

            # Convert to fp32 for computation
            X_block = X_block.to(tl.float32)

            if elementwise_affine:
                W_block = tl.load(W_ptr + col_offsets, mask=mask, other=0.0, eviction_policy="evict_first")
                W_offset = W_block + offset

                # Compute m based on casting mode
                if casting_mode == _CASTING_MODE_LLAMA:
                    m = (dY_block * W_offset).to(tl.float32)
                elif casting_mode == _CASTING_MODE_GEMMA:
                    dY_block = dY_block.to(tl.float32)
                    m = dY_block * W_offset
                else:
                    m = dY_block * W_offset
            else:
                # Compute m based on casting mode
                if casting_mode == _CASTING_MODE_LLAMA:
                    m = dY_block.to(tl.float32)
                elif casting_mode == _CASTING_MODE_GEMMA:
                    m = dY_block.to(tl.float32)
                else:
                    m = dY_block

            # Accumulate sum(m * X)
            sum_m_X += tl.sum(tl.where(mask, m * X_block, 0.0))

        # Compute the correction factor
        correction_factor = -(1.0 / n_cols) * rstd * rstd * sum_m_X

        # Second pass: compute gradients
        for col_block_idx in range(num_col_blocks):
            col_start = col_block_idx * BLOCK_SIZE
            col_offsets = col_start + offsets
            mask = col_offsets < n_cols

            dY_block = tl.load(dY_row_ptr + col_offsets, mask=mask, other=0.0)
            X_block = tl.load(X_row_ptr + col_offsets, mask=mask, other=0.0)

            X_block = X_block.to(tl.float32)

            if elementwise_affine:
                W_block = tl.load(W_ptr + col_offsets, mask=mask, other=0.0)
                W_offset = W_block + offset

                # Compute m based on casting mode
                if casting_mode == _CASTING_MODE_LLAMA:
                    m = (dY_block * W_offset).to(tl.float32)
                elif casting_mode == _CASTING_MODE_GEMMA:
                    dY_block = dY_block.to(tl.float32)
                    m = dY_block * W_offset
                else:
                    m = dY_block * W_offset
            else:
                # Compute m based on casting mode
                if casting_mode == _CASTING_MODE_LLAMA:
                    m = dY_block.to(tl.float32)
                elif casting_mode == _CASTING_MODE_GEMMA:
                    m = dY_block.to(tl.float32)
                else:
                    m = dY_block

            # Compute dX
            dX_block = rstd * m + rstd * correction_factor * X_block

            # Store dX
            tl.store(dX_row_ptr + col_offsets, dX_block.to(X_dtype), mask=mask)

            if elementwise_affine:
                # Compute dW contribution (accumulate per program)
                if casting_mode == _CASTING_MODE_LLAMA:
                    dW_block = dY_block * (X_block * rstd).to(X_dtype)
                else:
                    dW_block = dY_block * (X_block * rstd)

                # Atomic add to dW_ptr (each program writes to its own row)
                dW_row_ptr = dW_ptr + pid * dW_row_stride

                # Load existing dW, add contribution, store back
                existing_dW = tl.load(dW_row_ptr + col_offsets, mask=mask, other=0.0)
                new_dW = existing_dW + dW_block.to(tl.float32)
                tl.store(dW_row_ptr + col_offsets, new_dW, mask=mask)