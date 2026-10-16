@triton.jit
def _fused_add_rms_norm_forward_kernel_npu(
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
    BLOCK_SIZE: tl.constexpr,
    NUM_STAGES: tl.constexpr,
):
    """
    NPU-optimized fused_add_rms_norm forward kernel.

    This kernel processes rows using a grid-stride loop pattern:
    1. Each program handles multiple rows
    2. For each row, we process it in column chunks of BLOCK_SIZE_N
    3. Grid size is limited to NPU core count to avoid resource overflow

    This solves two problems:
    1. UB overflow when n_cols is too large (original kernel used n_cols as BLOCK_SIZE_N)
    2. Efficient multi-row processing within a single kernel launch
    """
    pid = tl.program_id(0)
    num_progs = tl.num_programs(0)
    num_col_blocks = tl.cdiv(n_cols, BLOCK_SIZE)

    if casting_mode == _CASTING_MODE_NONE:
        eps = eps.to(X_DTYPE)
        offset = offset.to(X_DTYPE)

    offsets = tl.arange(0, BLOCK_SIZE)
    # Grid-stride loop over rows
    for row_idx in tl.range(pid, n_rows, num_progs, num_stages=NUM_STAGES):
        Y_row_ptr = Y_ptr + row_idx * Y_row_stride
        S_row_ptr = S_ptr + row_idx * S_row_stride
        X_row_ptr = X_ptr + row_idx * X_row_stride
        R_row_ptr = R_ptr + row_idx * R_row_stride
        RSTD_row_ptr = RSTD_ptr + row_idx * RSTD_row_stride

        # Accumulator for mean_square computation across all column blocks
        sum_square = 0.0

        # First pass: compute S_row = X_row + R_row and accumulate sum of squares
        for col_block_idx in range(num_col_blocks):
            col_start = col_block_idx * BLOCK_SIZE
            col_offsets = col_start + offsets
            mask = col_offsets < n_cols

            X_block = tl.load(X_row_ptr + col_offsets, mask=mask, other=0.0, eviction_policy="evict_first")
            R_block = tl.load(R_row_ptr + col_offsets, mask=mask, other=0.0, eviction_policy="evict_first")
            S_block = X_block + R_block

            # Store S_row
            tl.store(S_row_ptr + col_offsets, S_block, mask=mask, cache_modifier=".cg")

            if casting_mode == _CASTING_MODE_LLAMA or casting_mode == _CASTING_MODE_GEMMA:
                S_block = S_block.to(tl.float32)

            # Accumulate sum of squares (only for valid elements)
            sum_square += tl.sum(tl.where(mask, S_block * S_block, 0.0))

        # Compute rstd for this row
        mean_square = sum_square / n_cols

        rstd = rsqrt(mean_square + eps)

        # Store rstd
        tl.store(RSTD_row_ptr, rstd)

        # Second pass: normalize and multiply by weight
        for col_block_idx in range(num_col_blocks):
            col_start = col_block_idx * BLOCK_SIZE
            col_offsets = col_start + offsets
            mask = col_offsets < n_cols

            # Load S_block (already computed in first pass)
            S_block = tl.load(S_row_ptr + col_offsets, mask=mask, other=0.0, cache_modifier=".ca")
            W_block = tl.load(W_ptr + col_offsets, mask=mask, other=0.0)

            # Apply casting based on mode
            if casting_mode == _CASTING_MODE_GEMMA:
                S_block = S_block.to(tl.float32)
                W_block = W_block.to(tl.float32)
            elif casting_mode == _CASTING_MODE_LLAMA:
                S_block = S_block.to(tl.float32)

            # Normalize
            S_block = S_block * rstd

            # Cast back for Llama mode before weight multiplication
            if casting_mode == _CASTING_MODE_LLAMA:
                S_block = S_block.to(X_DTYPE)
            # Apply weight
            Y_block = S_block * (offset + W_block)

            # Cast back for Gemma mode
            if casting_mode == _CASTING_MODE_GEMMA:
                Y_block = Y_block.to(X_DTYPE)

            # Store result
            tl.store(Y_row_ptr + col_offsets, Y_block, mask=mask)