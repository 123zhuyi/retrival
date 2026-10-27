@triton.autotune(
    configs=[
        triton.Config({"BLOCK_SIZE": 4096}),
        triton.Config({"BLOCK_SIZE": 8192}),
    ],
    key=["n_cols"],
    restore_value=["X_ptr"],
)
@libentry()
@triton.jit
def _cross_entropy_prime_kernel(
    X_ptr,
    Y_ptr,
    loss_ptr,
    n_rows,  # Now we need n_rows to know the loop boundary
    n_cols,
    X_stride_row,  # Stride to jump between rows
    Y_stride_row,  # Stride to jump between rows for Y
    loss_stride_row,  # Stride to jump between rows for loss
    n_non_ignore,
    ignore_index,
    lse_square_scale: tl.constexpr,
    label_smoothing: tl.constexpr,
    reduction: tl.constexpr,
    OVERWRITE_GRAD_LOGITS: tl.constexpr,
    IS_BACKWARD: tl.constexpr,
    BLOCK_SIZE: tl.constexpr,
):
    """
    This is the modified version of the cross-entropy kernel that uses a fixed grid size.
    Each program_id corresponds to a "worker" that processes multiple rows in a strided loop.
    """
    # program_id is the worker ID, from 0 to grid_size-1
    pid = tl.program_id(0)
    # num_programs is the total number of workers (e.g., 48)
    num_programs = tl.num_programs(0)

    # The outer loop iterates through all rows assigned to this worker
    for row_idx in range(pid, n_rows, num_programs):
        # --- Pointer calculations for the current row ---
        # Note: We use row_idx for offsets now, not program_id
        current_X_ptr = X_ptr + row_idx * X_stride_row
        current_Y_ptr = Y_ptr + row_idx * Y_stride_row

        y = tl.load(current_Y_ptr)

        m = float("-inf")
        d = 0.0
        ori_X_y = 0.0

        if y != ignore_index:
            ori_X_y = tl.load(current_X_ptr + y).cast(tl.float32)

        scaled_x_sum = 0.0
        eps = label_smoothing / n_cols

        # First pass: calculate lse (log-sum-exp)
        for i in range(0, n_cols, BLOCK_SIZE):
            X_offsets = i + tl.arange(0, BLOCK_SIZE)
            X_mask = X_offsets < n_cols
            X_block = tl.load(
                current_X_ptr + X_offsets,
                mask=X_mask,
                other=float("-inf"),
            ).cast(tl.float32)

            block_max = tl.max(X_block, axis=0)  # Use axis=0 for clarity, it's a 1D reduction
            if label_smoothing > 0:
                X_block2 = tl.load(
                    current_X_ptr + X_offsets,
                    mask=X_mask,
                    other=0.0,
                ).cast(tl.float32)
                scaled_x_sum += tl.sum(-eps * X_block2, axis=0).to(tl.float32)
            m_new = tl.maximum(m, block_max)
            d = d * tl.exp(m - m_new) + tl.sum(tl.exp(X_block - m_new), axis=0)
            m = m_new

        lse = m + tl.log(d)

        if OVERWRITE_GRAD_LOGITS:
            # Second pass: calculate gradient and overwrite X in-place
            for i in range(0, n_cols, BLOCK_SIZE):
                X_offsets = i + tl.arange(0, BLOCK_SIZE)
                X_mask = X_offsets < n_cols

                X_block = tl.load(
                    current_X_ptr + X_offsets,
                    mask=X_mask,
                    other=float("-inf"),
                ).cast(tl.float32)

                X_block = tl.exp(X_block - m) / d
                X_block += 2 * lse_square_scale * lse * X_block
                X_block += -eps
                if reduction == "mean":
                    X_block = X_block / n_non_ignore

                tl.store(current_X_ptr + X_offsets, X_block, mask=X_mask)

        if not IS_BACKWARD:
            current_loss_ptr = loss_ptr + row_idx * loss_stride_row
            # Calculate and store the loss for the current row
            loss = lse - ori_X_y

            if label_smoothing > 0:
                smooth_loss = scaled_x_sum + label_smoothing * lse
                loss = loss * (1 - label_smoothing) + smooth_loss

            z_loss = lse_square_scale * lse * lse
            if reduction == "mean":
                loss = loss / n_non_ignore
                z_loss = z_loss / n_non_ignore
            loss += z_loss

            tl.store(current_loss_ptr, loss)