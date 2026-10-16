@libentry()
@triton.jit
def _rmsnorm_bwd_large_cols_kernel(
    dY_ptr,
    dY_row_stride,
    dX_ptr,
    dX_row_stride,
    X_ptr,
    X_row_stride,
    W_ptr,
    RSTD_ptr,
    RSTD_row_stride,
    dW_ptr,
    dW_row_stride,
    n_rows,
    n_cols,
    offset,
    casting_mode_int: tl.constexpr,
    X_dtype: tl.constexpr,
    BLOCK_SIZE_N: tl.constexpr,
    BLOCK_SIZE_M: tl.constexpr,
    IS_DETERMINISTIC: tl.constexpr,
):
    pid = tl.program_id(axis=0)
    grid_size = tl.num_programs(axis=0)

    num_row_tasks = (n_rows + BLOCK_SIZE_M - 1) // BLOCK_SIZE_M

    for row_task_id in range(pid, num_row_tasks, grid_size):
        block_start_row = row_task_id * BLOCK_SIZE_M
        rows_off = block_start_row + tl.arange(0, BLOCK_SIZE_M)
        rows_mask = rows_off < n_rows

        rstd_vec = tl.load(RSTD_ptr + rows_off * RSTD_row_stride, mask=rows_mask, other=0.0)

        dot_product_acc = tl.zeros((BLOCK_SIZE_M,), dtype=tl.float32)
        for col_offset in range(0, n_cols, BLOCK_SIZE_N):
            cols_off = col_offset + tl.arange(0, BLOCK_SIZE_N)
            cols_mask = cols_off < n_cols
            block_mask = rows_mask[:, None] & cols_mask[None, :]

            dY_chunk = tl.load(
                dY_ptr + rows_off[:, None] * dY_row_stride + cols_off[None, :], mask=block_mask, other=0.0
            )
            X_chunk = tl.load(
                X_ptr + rows_off[:, None] * X_row_stride + cols_off[None, :], mask=block_mask, other=0.0
            ).to(tl.float32)
            W_chunk = tl.load(W_ptr + cols_off, mask=cols_mask, other=0.0)

            W_chunk_offset = W_chunk + offset
            m_chunk = dY_chunk * W_chunk_offset[None, :]
            if casting_mode_int != _CASTING_MODE_NONE:
                m_chunk = m_chunk.to(tl.float32)

            dot_product_acc += tl.sum(m_chunk * X_chunk, axis=1)

        rstd_vec_sq = rstd_vec * rstd_vec
        for col_offset in range(0, n_cols, BLOCK_SIZE_N):
            cols_off = col_offset + tl.arange(0, BLOCK_SIZE_N)
            cols_mask = cols_off < n_cols
            block_mask = rows_mask[:, None] & cols_mask[None, :]

            dY_chunk = tl.load(
                dY_ptr + rows_off[:, None] * dY_row_stride + cols_off[None, :], mask=block_mask, other=0.0
            )
            X_chunk = tl.load(X_ptr + rows_off[:, None] * X_row_stride + cols_off[None, :], mask=block_mask, other=0.0)
            W_chunk = tl.load(W_ptr + cols_off, mask=cols_mask, other=0.0)

            W_chunk_offset = W_chunk + offset
            X_chunk_f32 = X_chunk.to(tl.float32)
            normed_X_chunk = X_chunk_f32 * rstd_vec[:, None]

            if casting_mode_int == _CASTING_MODE_LLAMA:
                m_chunk = (dY_chunk * W_chunk_offset[None, :]).to(tl.float32)
                dW_chunk_sum = tl.sum(dY_chunk * normed_X_chunk.to(X_dtype), axis=0)
            elif casting_mode_int == _CASTING_MODE_GEMMA:
                dY_chunk_f32 = dY_chunk.to(tl.float32)
                W_chunk_offset = W_chunk_offset.to(tl.float32)
                m_chunk = dY_chunk_f32 * W_chunk_offset[None, :]
                dW_chunk_sum = tl.sum(dY_chunk_f32 * normed_X_chunk, axis=0)
            else:
                m_chunk = dY_chunk * W_chunk_offset[None, :]
                dW_chunk_sum = tl.sum(dY_chunk * normed_X_chunk, axis=0)

            term1 = rstd_vec[:, None] * m_chunk
            term2 = -(1 / n_cols) * rstd_vec_sq[:, None] * rstd_vec[:, None] * dot_product_acc[:, None] * X_chunk_f32
            dX_chunk = term1 + term2

            tl.store(
                dX_ptr + rows_off[:, None] * dX_row_stride + cols_off[None, :], dX_chunk.to(X_dtype), mask=block_mask
            )

            if IS_DETERMINISTIC:
                dW_existing = tl.load(dW_ptr + pid * dW_row_stride + cols_off, mask=cols_mask, other=0.0)
                tl.store(dW_ptr + pid * dW_row_stride + cols_off, dW_existing + dW_chunk_sum, mask=cols_mask)
            else:
                tl.atomic_add(dW_ptr + cols_off, dW_chunk_sum, mask=cols_mask)