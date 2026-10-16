@triton.heuristics({"BLOCK_SIZE_M": lambda args: ceil_div(4096, args["n_cols"])})
@libentry()
@triton.jit
def _layernorm_bwd_kernel(
    DY_ptr,
    DX_ptr,
    DW_ptr,
    DB_ptr,
    X_ptr,
    W_ptr,
    Mean_ptr,
    RSTD_ptr,
    stride_dy_row,
    stride_dx_row,
    stride_x_row,
    n_rows,
    n_cols,
    X_dtype: tl.constexpr,
    BLOCK_SIZE_N: tl.constexpr,
    BLOCK_SIZE_M: tl.constexpr,
):
    pid = tl.program_id(0)
    num_programs = tl.num_programs(0)

    num_row_tasks = (n_rows + BLOCK_SIZE_M - 1) // BLOCK_SIZE_M

    cols_off = tl.arange(0, BLOCK_SIZE_N)
    cols_mask = cols_off < n_cols

    dW_acc = tl.zeros((BLOCK_SIZE_N,), dtype=tl.float32)
    dB_acc = tl.zeros((BLOCK_SIZE_N,), dtype=tl.float32)

    w = tl.load(W_ptr + cols_off, mask=cols_mask, other=0.0).to(tl.float32)

    for row_task_id in range(pid, num_row_tasks, num_programs):
        block_start_row = row_task_id * BLOCK_SIZE_M
        rows_off = block_start_row + tl.arange(0, BLOCK_SIZE_M)
        rows_mask = rows_off < n_rows
        block_mask = rows_mask[:, None] & cols_mask[None, :]

        mean = tl.load(Mean_ptr + rows_off, mask=rows_mask, other=0.0).to(tl.float32)
        rstd = tl.load(RSTD_ptr + rows_off, mask=rows_mask, other=0.0).to(tl.float32)

        dy = tl.load(DY_ptr + rows_off[:, None] * stride_dy_row + cols_off[None, :], mask=block_mask, other=0.0).to(
            tl.float32
        )

        x = tl.load(X_ptr + rows_off[:, None] * stride_x_row + cols_off[None, :], mask=block_mask, other=0.0).to(
            tl.float32
        )

        x_hat = (x - mean[:, None]) * rstd[:, None]

        dW_acc += tl.sum(dy * x_hat, axis=0)
        dB_acc += tl.sum(dy, axis=0)

        wdy = w[None, :] * dy
        c1 = tl.sum(x_hat * wdy, axis=1) / n_cols
        c2 = tl.sum(wdy, axis=1) / n_cols
        dx = (wdy - (x_hat * c1[:, None] + c2[:, None])) * rstd[:, None]

        tl.store(DX_ptr + rows_off[:, None] * stride_dx_row + cols_off[None, :], dx.to(X_dtype), mask=block_mask)

    tl.atomic_add(DW_ptr + cols_off, dW_acc, mask=cols_mask)
    tl.atomic_add(DB_ptr + cols_off, dB_acc, mask=cols_mask)