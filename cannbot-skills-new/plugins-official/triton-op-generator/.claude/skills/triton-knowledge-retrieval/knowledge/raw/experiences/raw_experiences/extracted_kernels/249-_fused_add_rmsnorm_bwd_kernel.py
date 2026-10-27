@triton.heuristics({"BLOCK_SIZE_M": lambda args: ceil_div(4096, args["n_cols"])})
@libentry()
@triton.jit
def _fused_add_rmsnorm_bwd_kernel(
    dY_ptr,
    dY_row_stride,
    dS_out_ptr,
    dS_out_row_stride,
    dX_ptr,
    dX_row_stride,
    S_ptr,
    S_row_stride,
    W_ptr,
    RSTD_ptr,
    RSTD_row_stride,
    dW_ptr,
    dW_row_stride,
    n_rows,
    n_cols,
    offset,
    casting_mode: tl.constexpr,
    S_dtype: tl.constexpr,
    has_dS_out: tl.constexpr,
    BLOCK_SIZE_N: tl.constexpr,
    BLOCK_SIZE_M: tl.constexpr,
):
    pid = tl.program_id(axis=0)
    grid_size = tl.num_programs(axis=0)
    num_row_tasks = (n_rows + BLOCK_SIZE_M - 1) // BLOCK_SIZE_M

    dW_acc = tl.zeros((BLOCK_SIZE_N,), dtype=tl.float32)

    cols_off = tl.arange(0, BLOCK_SIZE_N)
    cols_mask = cols_off < n_cols
    W_row = tl.load(W_ptr + cols_off, mask=cols_mask, other=0.0)
    W_row_offset = W_row + offset

    for row_task_id in range(pid, num_row_tasks, grid_size):
        block_start_row = row_task_id * BLOCK_SIZE_M
        rows_off = block_start_row + tl.arange(0, BLOCK_SIZE_M)
        rows_mask = rows_off < n_rows
        block_mask = rows_mask[:, None] & cols_mask[None, :]

        dY_block = tl.load(dY_ptr + rows_off[:, None] * dY_row_stride + cols_off[None, :], mask=block_mask, other=0.0)
        S_block = tl.load(S_ptr + rows_off[:, None] * S_row_stride + cols_off[None, :], mask=block_mask, other=0.0)
        rstd_vec = tl.load(RSTD_ptr + rows_off * RSTD_row_stride, mask=rows_mask, other=0.0)

        S_block_f32 = S_block.to(tl.float32)
        normed_S_block = S_block_f32 * rstd_vec[:, None]

        if casting_mode == _CASTING_MODE_LLAMA:
            m_block = (dY_block * W_row_offset[None, :]).to(tl.float32)
            dW_acc += tl.sum(dY_block * normed_S_block.to(S_dtype), axis=0)
        elif casting_mode == _CASTING_MODE_GEMMA:
            dY_block_f32 = dY_block.to(tl.float32)
            W_row_offset_f32 = W_row_offset.to(tl.float32)
            m_block = dY_block_f32 * W_row_offset_f32[None, :]
            dW_acc += tl.sum(dY_block_f32 * normed_S_block, axis=0)
        else:
            m_block = dY_block * W_row_offset[None, :]
            dW_acc += tl.sum(dY_block * normed_S_block, axis=0)

        dot_product_vec = tl.sum(m_block * S_block_f32, axis=1)
        rstd_vec_sq = rstd_vec * rstd_vec

        term1 = rstd_vec[:, None] * m_block
        term2 = -(1 / n_cols) * rstd_vec_sq[:, None] * rstd_vec[:, None] * dot_product_vec[:, None] * S_block_f32

        grad_after_norm = term1 + term2

        dS_block = grad_after_norm
        if has_dS_out:
            dS_out_block = tl.load(
                dS_out_ptr + rows_off[:, None] * dS_out_row_stride + cols_off[None, :], mask=block_mask, other=0.0
            )
            dS_block += dS_out_block.to(dS_block.dtype)

        tl.store(dX_ptr + rows_off[:, None] * dX_row_stride + cols_off[None, :], dS_block.to(S_dtype), mask=block_mask)

    dW_ptr_prog = dW_ptr + pid * dW_row_stride + cols_off
    tl.store(dW_ptr_prog, dW_acc, mask=cols_mask)