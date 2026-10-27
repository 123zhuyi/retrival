@libentry()
@triton.autotune(configs=runtime.get_tuned_config("masked_fill"), key=["N"])
@triton.jit
def masked_fill_kernel(
    inp,
    expand_mask,
    value,
    out,
    N,
    BLOCK_SIZE: tl.constexpr,
    BLOCK_SIZE_SUB: tl.constexpr,
):
    pid = tle.program_id(axis=0)
    base_offset = pid * BLOCK_SIZE

    # 计算需要处理的总块数
    num_sub_blocks = BLOCK_SIZE // BLOCK_SIZE_SUB

    # 循环处理每个子块
    for sub_block_idx in range(num_sub_blocks):
        # 计算当前子块的偏移量
        sub_offset = base_offset + sub_block_idx * BLOCK_SIZE_SUB
        offsets = sub_offset + tl.arange(0, BLOCK_SIZE_SUB)
        mask = offsets < N

        # 加载 input 和 mask
        input_vals = tl.load(inp + offsets, mask=mask, other=0)
        fill_mask_vals = tl.load(expand_mask + offsets, mask=mask, other=0).to(tl.int1)

        # 先写入原始输入
        tl.store(out + offsets, input_vals, mask=mask)

        # 再在需要填充的位置覆盖写入 value
        value_to_write = tl.full([BLOCK_SIZE_SUB], value, dtype=input_vals.dtype)
        overwrite_vals = tl.where(
            fill_mask_vals, value_to_write, tl.load(out + offsets, mask=mask, other=0)
        )
        tl.store(out + offsets, overwrite_vals, mask=mask)