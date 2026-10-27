@libentry()
@triton.autotune(
    configs=[
        triton.Config({"BLOCK_SUB_HW_SIZE": 32}),
        triton.Config({"BLOCK_SUB_HW_SIZE": 64}),
        triton.Config({"BLOCK_SUB_HW_SIZE": 128}),
        triton.Config({"BLOCK_SUB_HW_SIZE": 256}),
        triton.Config({"BLOCK_SUB_HW_SIZE": 512}),
        triton.Config({"BLOCK_SUB_HW_SIZE": 1024}),
        triton.Config({"BLOCK_SUB_HW_SIZE": 2048}),
        triton.Config({"BLOCK_SUB_HW_SIZE": 4096}),
        triton.Config({"BLOCK_SUB_HW_SIZE": 8192}),
        triton.Config({"BLOCK_SUB_HW_SIZE": 16384}),
    ],
    key=["HW", "group_size"],
)
@triton.jit(do_not_specialize=["eps"])
def group_norm_kernel(
    X,
    Y,
    W,
    B,
    Mean,
    Rstd,
    group_size,
    C,
    HW,
    num_groups,
    eps,
    BLOCK_GROUP_SIZE: tl.constexpr,
    BLOCK_HW_SIZE: tl.constexpr,
    BLOCK_SUB_HW_SIZE: tl.constexpr,
):
    pid = tl.program_id(0)
    batch_idx = pid // num_groups
    group_idx = pid % num_groups

    # 计算当前group在整个tensor中的起始位置
    batch_offset = batch_idx * C * HW
    group_start_channel = group_idx * group_size

    num_elements = group_size * HW

    # 第一次遍历：计算均值
    X_sum = 0.0
    for hw_start in range(0, HW, BLOCK_SUB_HW_SIZE):
        hw_offsets = hw_start + tl.arange(0, BLOCK_SUB_HW_SIZE)
        hw_mask = hw_offsets < HW

        # 先按HW维度连续，再按channel维度
        for c_idx in range(BLOCK_GROUP_SIZE):
            if c_idx < group_size and (group_start_channel + c_idx) < C:
                channel_offset = group_start_channel + c_idx
                # 连续访问HW维度的数据
                base_offset = batch_offset + channel_offset * HW + hw_offsets
                X_vals = tl.load(X + base_offset, mask=hw_mask, other=0.0).to(
                    tl.float32
                )
                X_sum += tl.sum(X_vals)

    mean = X_sum / num_elements

    # 第二次遍历：计算方差
    X_var_sum = 0.0
    for hw_start in range(0, HW, BLOCK_SUB_HW_SIZE):
        hw_offsets = hw_start + tl.arange(0, BLOCK_SUB_HW_SIZE)
        hw_mask = hw_offsets < HW

        for c_idx in range(BLOCK_GROUP_SIZE):
            if c_idx < group_size and (group_start_channel + c_idx) < C:
                channel_offset = group_start_channel + c_idx
                base_offset = batch_offset + channel_offset * HW + hw_offsets
                X_vals = tl.load(X + base_offset, mask=hw_mask, other=mean).to(
                    tl.float32
                )
                x_centered = X_vals - mean
                X_var_sum += tl.sum(x_centered * x_centered)

    var = X_var_sum / num_elements
    rstd = rsqrt(var + eps)

    # 第三次遍历：归一化并写回
    for hw_start in range(0, HW, BLOCK_SUB_HW_SIZE):
        hw_offsets = hw_start + tl.arange(0, BLOCK_SUB_HW_SIZE)
        hw_mask = hw_offsets < HW

        for c_idx in range(BLOCK_GROUP_SIZE):
            if c_idx < group_size and (group_start_channel + c_idx) < C:
                channel_offset = group_start_channel + c_idx
                base_offset = batch_offset + channel_offset * HW + hw_offsets

                # 加载数据
                X_vals = tl.load(X + base_offset, mask=hw_mask, other=0.0).to(
                    tl.float32
                )

                # 归一化并应用仿射变换
                x_normalized = (X_vals - mean) * rstd
                if W is not None:
                    w_val = tl.load(W + channel_offset)
                    x_normalized = x_normalized * w_val
                if B is not None:
                    b_val = tl.load(B + channel_offset)
                    x_normalized = x_normalized + b_val

                # 存储结果
                tl.store(Y + base_offset, x_normalized, mask=hw_mask)

    # 存储均值和标准差
    mean_rstd_offset = batch_idx * num_groups + group_idx
    tl.store(Mean + mean_rstd_offset, mean)
    tl.store(Rstd + mean_rstd_offset, rstd)