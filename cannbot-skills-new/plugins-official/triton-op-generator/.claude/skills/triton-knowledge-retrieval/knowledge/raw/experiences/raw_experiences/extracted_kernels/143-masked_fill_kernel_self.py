@libentry()
@triton.autotune(configs=runtime.get_tuned_config("masked_fill"), key=["N"])
@triton.jit
def masked_fill_kernel_self(
    inp, expand_mask, value, N, BLOCK_SIZE: tl.constexpr, BLOCK_SIZE_SUB: tl.constexpr
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

        # 加载 expand_mask
        fill_mask = tl.load(expand_mask + offsets, mask=mask, other=0).to(tl.int1)

        # 构造写入的值：fill_mask==1 用 value，fill_mask==0 保留原值
        orig = tl.load(inp + offsets, mask=mask, other=0)
        value_vec = tl.full([BLOCK_SIZE_SUB], value, dtype=orig.dtype)
        result = tl.where(fill_mask, value_vec, orig)

        # 存储结果
        tl.store(inp + offsets, result, mask=mask)