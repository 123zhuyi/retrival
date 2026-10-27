@triton.jit
def _padded_scatter_kernel(
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

        cur_sub_block_size = tl.minimum(SUB_BLOCK_SIZE, idx_end - idx_offset)
        # 统计当前sub_block_size中有多少个专家
        first_bin_idx = tl.get_element(cur_bin_ids, (0,))
        last_bin_idx = tl.get_element(cur_bin_ids, (cur_sub_block_size - 1,))
        bin_count = last_bin_idx - first_bin_idx + 1
        # 循环处理每个专家对应的数据
        begin = 0
        for i in range(bin_count):
            base_bin_idx = tl.get_element(cur_bin_ids, (begin,))  # 当前专家id
            # 找同一个专家的起止
            cur_bin = tl.load(bins + base_bin_idx)  # 当前专家及之前的专家一共要处理多少token
            count = tl.minimum(cur_bin - idx_offset - begin, cur_sub_block_size - begin)
            end = begin + count
            # 处理相同专家的数据
            offset_in_bin = idx_offset + begin
            if base_bin_idx > 0:
                offset_in_bin -= tl.load(bins + base_bin_idx - 1)
            x_offset = offset_in_bin
            if base_bin_idx > 0:
                x_offset += tl.load(padded_bins + base_bin_idx - 1)
            x_offsets = tl.arange(0, SUB_BLOCK_SIZE) + x_offset
            x_mask = x_offsets < x_offset + (end - begin)

            for col_offset in range(0, NUM_COLUMNS, BLOCK_X):
                col_offsets = tl.arange(0, BLOCK_X) + col_offset
                col_mask = col_offsets < NUM_COLUMNS
                cur_x = tl.load(x + (x_offsets[:, None] * NUM_COLUMNS + col_offsets[None, :]),
                                x_mask[:, None] & col_mask[None, :])
                for j in range(begin, end):
                    val_idx = j - begin
                    val = tl.extract_slice(cur_x, offsets=(val_idx, 0), sizes=(1, BLOCK_X), strides=(1, 1))
                    idx = tl.get_element(cur_indices, (j,))
                    if SCALE:
                        scale = tl.load(weights + idx)
                        val = val.to(tl.float32) * scale.to(tl.float32)
                    tl.store(out + idx * NUM_COLUMNS + col_offsets,
                             val.to(out.dtype.element_ty).reshape(BLOCK_X), col_mask)
            # 跳到下一个专家位置
            begin = end