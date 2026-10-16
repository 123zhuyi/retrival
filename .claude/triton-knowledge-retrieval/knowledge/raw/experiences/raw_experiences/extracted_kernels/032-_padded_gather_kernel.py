@triton.jit
def _padded_gather_kernel(
    x,
    out,
    indices,
    bin_ids,
    weights,
    bins,
    padded_bins,
    INDICES_LENGTH: tl.constexpr,
    BLOCK_SIZE: tl.constexpr,
    SUB_BLOCK_SIZE: tl.constexpr,
    NUM_COLUMNS: tl.constexpr,
    BLOCK_X: tl.constexpr,
    TOP_K: tl.constexpr,
    SCALE: tl.constexpr,
):
    pid = tl.program_id(0)
    idx_begin = pid * BLOCK_SIZE
    idx_end = tl.minimum((pid + 1) * BLOCK_SIZE, INDICES_LENGTH)
    for idx_in_block in range(0, BLOCK_SIZE, SUB_BLOCK_SIZE):
        idx_offset = idx_begin + idx_in_block
        idx_offsets = tl.arange(0, SUB_BLOCK_SIZE) + idx_offset
        idx_mask = idx_offsets < idx_end

        cur_indices = tl.load(indices + idx_offsets, idx_mask, other=0)
        cur_bin_ids = tl.load(bin_ids + idx_offsets, idx_mask, other=0)
        base_bin_idx = tl.get_element(cur_bin_ids, (0,))  # expert_id
        tmp_buf = tl.zeros((SUB_BLOCK_SIZE, BLOCK_X), x.dtype.element_ty)
        for col_offset in range(0, NUM_COLUMNS, BLOCK_X):
            col_offsets = tl.arange(0, BLOCK_X) + col_offset
            col_mask = col_offsets < NUM_COLUMNS
            # 计算当前token属于当前专家处理的第几个token
            offset_in_bin = idx_offset
            if base_bin_idx > 0:
                offset_in_bin -= tl.load(bins + base_bin_idx - 1)
            idx_o = offset_in_bin
            if base_bin_idx > 0:
                idx_o += tl.load(padded_bins + base_bin_idx - 1)
            COUNT = 0
            for i in range(0, SUB_BLOCK_SIZE):
                if idx_offset + i < idx_end:
                    bin_idx = tl.get_element(cur_bin_ids, (i,))
                    if bin_idx != base_bin_idx:   # 专家号变更时，需要把前一个专家处理的数据写回gm
                        # store前一个专家要处理的token
                        o_offsets = tl.arange(0, SUB_BLOCK_SIZE) + idx_o
                        o_mask = o_offsets < idx_o + COUNT
                        tl.store(out + (o_offsets[:, None] * NUM_COLUMNS + col_offsets[None, :]),
                                 tmp_buf, o_mask[:, None] & col_mask[None, :])
                        # 更新起始偏移量和当前专家号
                        base_bin_idx = bin_idx
                        offset_in_bin = idx_offset + i
                        if base_bin_idx > 0:
                            offset_in_bin -= tl.load(bins + base_bin_idx - 1)
                        idx_o = offset_in_bin
                        if base_bin_idx > 0:
                            idx_o += tl.load(padded_bins + base_bin_idx - 1)
                        COUNT = 0
                    idx_x = tl.get_element(cur_indices, (i,)) // TOP_K * NUM_COLUMNS
                    val = tl.load(x + idx_x + col_offsets, col_mask)
                    if SCALE:
                        scale = tl.load(weights + idx_x)
                        val = (val.to(tl.float32) * scale.to(tl.float32)).to(out.dtype.element_ty)
                    tmp_buf = tl.insert_slice(tmp_buf, val[None, :], offsets=(COUNT, 0), sizes=(1, BLOCK_X),
                                              strides=(1, 1))
                    COUNT += 1
            # store last block
            o_offsets = tl.arange(0, SUB_BLOCK_SIZE) + idx_o
            o_mask = o_offsets < idx_o + COUNT
            tl.store(out + (o_offsets[:, None] * NUM_COLUMNS + col_offsets[None, :]),
                     tmp_buf, o_mask[:, None] & col_mask[None, :])