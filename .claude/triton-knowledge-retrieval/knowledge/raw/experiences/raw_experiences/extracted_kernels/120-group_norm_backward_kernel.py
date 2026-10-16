@libentry()
@triton.jit
def group_norm_backward_kernel(
    grad_y,
    X,
    W,
    Mean,
    Rstd,
    num_groups,
    group_size,
    grad_x,
    C,
    HW,
    BLOCK_GROUP_SIZE: tl.constexpr,
    BLOCK_HW_SIZE: tl.constexpr = 128,
):
    pid = tle.program_id(0)
    group = pid % num_groups
    num_elements = group_size * HW

    group_offset = tl.arange(0, BLOCK_GROUP_SIZE)
    wb_offset = group * group_size + group_offset

    wb_mask = wb_offset < C
    rstd = tl.load(Rstd + pid).to(tl.float32)
    mean = tl.load(Mean + pid).to(tl.float32)

    if W is None:
        weight = 1
    else:
        weight = tl.load(W + wb_offset, mask=wb_mask, other=0.0).to(tl.float32)[:, None]

    dx_part2 = tl.zeros([BLOCK_GROUP_SIZE, BLOCK_HW_SIZE], dtype=tl.float32)
    dx_part3 = tl.zeros([BLOCK_GROUP_SIZE, BLOCK_HW_SIZE], dtype=tl.float32)
    for off in range(0, HW, BLOCK_HW_SIZE):
        hw_offset = off + tl.arange(0, BLOCK_HW_SIZE)
        hw_mask = hw_offset < HW
        xy_offset = pid * num_elements + group_offset[:, None] * HW + hw_offset[None, :]
        xy_mask = wb_mask[:, None] & hw_mask[None, :]
        dY_val = tl.load(grad_y + xy_offset, mask=xy_mask, other=0.0).to(tl.float32)
        X_val = tl.load(X + xy_offset, mask=xy_mask, other=0.0).to(tl.float32)

        x_hat = tl.where(xy_mask, rstd * (X_val - mean), 0.0)
        dx_hat = weight * dY_val
        dx_part2 += dx_hat
        dx_part3 += dx_hat * x_hat

    dx_2 = tl.sum(dx_part2)
    dx_3 = tl.sum(dx_part3)

    for off in range(0, HW, BLOCK_HW_SIZE):
        hw_offset = off + tl.arange(0, BLOCK_HW_SIZE)
        hw_mask = hw_offset < HW
        xy_offset = pid * num_elements + group_offset[:, None] * HW + hw_offset[None, :]
        xy_mask = wb_mask[:, None] & hw_mask[None, :]

        dY_val = tl.load(grad_y + xy_offset, mask=xy_mask, other=0.0).to(tl.float32)
        X_val = tl.load(X + xy_offset, mask=xy_mask, other=0.0).to(tl.float32)

        x_hat = tl.where(xy_mask, rstd * (X_val - mean), 0.0)
        dx_hat = weight * dY_val
        dx = rstd * (dx_hat - (dx_2 + x_hat * dx_3) / num_elements)

        tl.store(grad_x + xy_offset, dx, xy_mask)