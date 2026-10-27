@triton.heuristics({"BLOCK_SIZE_M": rms_norm_fwd_heuristics})
@libentry()
@triton.jit
def _rmsnorm_fwd_kernel(
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
    casting_mode_int: tl.constexpr,
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

        X_ptr_row_block = X_ptr + rows_off[:, None] * X_row_stride
        X_dtype = X_ptr.dtype.element_ty

        var_acc = tl.zeros((BLOCK_SIZE_M,), dtype=tl.float32)
        for col_offset in range(0, n_cols, BLOCK_SIZE_N):
            cols_off = col_offset + tl.arange(0, BLOCK_SIZE_N)
            cols_mask = cols_off < n_cols
            block_mask = rows_mask[:, None] & cols_mask[None, :]

            X_chunk = tl.load(X_ptr_row_block + cols_off[None, :], mask=block_mask, other=0.0).to(tl.float32)
            var_acc += tl.sum(X_chunk * X_chunk, axis=1)

        var = var_acc / n_cols
        rstd_vec = tl.rsqrt(var + eps)
        tl.store(RSTD_ptr + rows_off * RSTD_row_stride, rstd_vec, mask=rows_mask)

        Y_ptr_row_block = Y_ptr + rows_off[:, None] * Y_row_stride
        for col_offset in range(0, n_cols, BLOCK_SIZE_N):
            cols_off = col_offset + tl.arange(0, BLOCK_SIZE_N)
            cols_mask = cols_off < n_cols
            block_mask = rows_mask[:, None] & cols_mask[None, :]

            X_chunk = tl.load(X_ptr_row_block + cols_off[None, :], mask=block_mask, other=0.0)
            W_chunk = tl.load(W_ptr + cols_off, mask=cols_mask, other=0.0)

            if casting_mode_int == _CASTING_MODE_GEMMA:
                X_chunk = X_chunk.to(tl.float32)
                W_chunk = W_chunk.to(tl.float32)
            elif casting_mode_int == _CASTING_MODE_LLAMA:
                X_chunk = X_chunk.to(tl.float32)

            if casting_mode_int == _CASTING_MODE_LLAMA:
                normed_X_chunk = (X_chunk * rstd_vec[:, None]).to(X_dtype)
            else:
                normed_X_chunk = X_chunk * rstd_vec[:, None]

            Y_chunk = normed_X_chunk * (W_chunk[None, :] + offset)
            if casting_mode_int == _CASTING_MODE_GEMMA:
                Y_chunk = Y_chunk.to(X_dtype)

            tl.store(Y_ptr_row_block + cols_off[None, :], Y_chunk, mask=block_mask)