@libentry()
@triton.jit
def weight_bias_backward_kernel(
    dY,
    X,
    Mean,
    Rstd,
    dW,
    dB,
    num_groups,
    group_size,
    N,
    C,
    HW,
    BLOCK_N: tl.constexpr,
    BLOCK_HW: tl.constexpr,
):
    pid = tle.program_id(0)
    group = pid // group_size
    n_offset = tl.arange(0, BLOCK_N)
    mr_mask = n_offset < N
    mean_ptr = Mean + group + n_offset * num_groups
    rstd_ptr = Rstd + group + n_offset * num_groups
    mean = tl.load(mean_ptr, mask=mr_mask, other=0.0).to(tl.float32)[:, None]
    rstd = tl.load(rstd_ptr, mask=mr_mask, other=0.0).to(tl.float32)[:, None]

    SUB_BLOCK_HW: tl.constexpr = 64

    dw_sum = 0.0
    db_sum = 0.0

    for hw_off in range(0, BLOCK_HW, SUB_BLOCK_HW):
        hw_offset = hw_off + tl.arange(0, SUB_BLOCK_HW)
        xy_mask = n_offset[:, None] < N and hw_offset[None, :] < HW
        dY_ptr = dY + pid * HW + n_offset[:, None] * C * HW + hw_offset[None, :]
        x_ptr = X + pid * HW + n_offset[:, None] * C * HW + hw_offset[None, :]
        grad_y = tl.load(dY_ptr, mask=xy_mask, other=0.0).to(tl.float32)
        x = tl.load(x_ptr, mask=xy_mask, other=0.0)
        x_f32 = x.to(tl.float32)
        if dW is not None:
            dw_sum = dw_sum + tl.sum((x_f32 - mean) * rstd * grad_y)
        if dB is not None:
            db_sum = db_sum + tl.sum(grad_y)

    if dW is not None:
        dw = dw_sum
        tl.store(dW + pid, dw.to(mean.dtype))
    if dB is not None:
        db = db_sum
        tl.store(dB + pid, db.to(mean.dtype))