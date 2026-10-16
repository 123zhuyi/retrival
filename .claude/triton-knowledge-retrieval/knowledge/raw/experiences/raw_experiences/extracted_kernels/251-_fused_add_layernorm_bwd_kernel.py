@triton.heuristics({"BLOCK_SIZE_M": lambda args: ceil_div(4096, args["n_cols"])})
@libentry()
@triton.jit
def _fused_add_layernorm_bwd_kernel(
    dY_ptr,
    dY_row_stride,
    dS_out_ptr,
    dS_out_row_stride,
    dX_ptr,
    dX_row_stride,
    S_ptr,
    S_row_stride,
    W_ptr,
    Mean_ptr,
    RSTD_ptr,
    dW_ptr,
    dW_row_stride,
    dB_ptr,
    dB_row_stride,
    n_rows,
    n_cols,
    has_dS_out: tl.constexpr,
    BLOCK_SIZE_N: tl.constexpr,
    BLOCK_SIZE_M: tl.constexpr,
):
    pid = tl.program_id(axis=0)
    grid_size = tl.num_programs(axis=0)
    num_row_tasks = (n_rows + BLOCK_SIZE_M - 1) // BLOCK_SIZE_M

    dW_acc_ptr = dW_ptr + pid * dW_row_stride
    dB_acc_ptr = dB_ptr + pid * dB_row_stride

    for row_task_id in range(pid, num_row_tasks, grid_size):
        block_start_row = row_task_id * BLOCK_SIZE_M
        rows_off = block_start_row + tl.arange(0, BLOCK_SIZE_M)
        rows_mask = rows_off < n_rows

        mean_vec = tl.load(Mean_ptr + rows_off, mask=rows_mask, other=0.0).to(tl.float32)
        rstd_vec = tl.load(RSTD_ptr + rows_off, mask=rows_mask, other=0.0).to(tl.float32)

        S_ptr_row_block = S_ptr + rows_off[:, None] * S_row_stride
        dY_ptr_row_block = dY_ptr + rows_off[:, None] * dY_row_stride
        dS_out_ptr_row_block = dS_out_ptr + rows_off[:, None] * dS_out_row_stride if has_dS_out else None

        ds_dx_hat_acc = tl.zeros((BLOCK_SIZE_M,), dtype=tl.float32)
        ds_acc = tl.zeros((BLOCK_SIZE_M,), dtype=tl.float32)

        for col_offset in range(0, n_cols, BLOCK_SIZE_N):
            cols_off = col_offset + tl.arange(0, BLOCK_SIZE_N)
            cols_mask = cols_off < n_cols
            block_mask = rows_mask[:, None] & cols_mask[None, :]

            S_chunk = tl.load(S_ptr_row_block + cols_off[None, :], mask=block_mask, other=0.0).to(tl.float32)
            dY_chunk = tl.load(dY_ptr_row_block + cols_off[None, :], mask=block_mask, other=0.0).to(tl.float32)
            W_chunk = tl.load(W_ptr + cols_off, mask=cols_mask, other=0.0).to(tl.float32)

            S_hat_chunk = (S_chunk - mean_vec[:, None]) * rstd_vec[:, None]
            dY_W_chunk = dY_chunk * W_chunk[None, :]

            ds_dx_hat_acc += tl.sum(dY_W_chunk * S_hat_chunk, axis=1)
            ds_acc += tl.sum(dY_W_chunk, axis=1)

            dW_chunk_acc = tl.sum(dY_chunk * S_hat_chunk, axis=0)
            dB_chunk_acc = tl.sum(dY_chunk, axis=0)
            tl.atomic_add(dW_acc_ptr + cols_off, dW_chunk_acc, mask=cols_mask)
            tl.atomic_add(dB_acc_ptr + cols_off, dB_chunk_acc, mask=cols_mask)

        dX_ptr_row_block = dX_ptr + rows_off[:, None] * dX_row_stride
        for col_offset in range(0, n_cols, BLOCK_SIZE_N):
            cols_off = col_offset + tl.arange(0, BLOCK_SIZE_N)
            cols_mask = cols_off < n_cols
            block_mask = rows_mask[:, None] & cols_mask[None, :]

            S_chunk = tl.load(S_ptr_row_block + cols_off[None, :], mask=block_mask, other=0.0).to(tl.float32)
            dY_chunk = tl.load(dY_ptr_row_block + cols_off[None, :], mask=block_mask, other=0.0).to(tl.float32)
            W_chunk = tl.load(W_ptr + cols_off, mask=cols_mask, other=0.0).to(tl.float32)

            S_hat_chunk = (S_chunk - mean_vec[:, None]) * rstd_vec[:, None]
            dY_W_chunk = dY_chunk * W_chunk[None, :]

            grad_of_norm_input = (
                dY_W_chunk - (S_hat_chunk * ds_dx_hat_acc[:, None] + ds_acc[:, None]) / n_cols
            ) * rstd_vec[:, None]

            dS_block = grad_of_norm_input
            if has_dS_out:
                dS_out_chunk = tl.load(dS_out_ptr_row_block + cols_off[None, :], mask=block_mask, other=0.0)
                dS_block += dS_out_chunk.to(dS_block.dtype)

            tl.store(dX_ptr_row_block + cols_off[None, :], dS_block.to(dX_ptr.dtype.element_ty), mask=block_mask)