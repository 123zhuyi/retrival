@triton.jit
def _binned_copy_scatter_npu(
                                a,
                                b,
                                indices,
                                bins,
                                bin_ids,
                                weights,
                                expert_capacity: tl.constexpr,
                                INDICES_LENGTH: tl.constexpr,
                                BLOCK_SIZE: tl.constexpr,
                                SUB_BLOCK_SIZE: tl.constexpr,
                                NUM_COLUMNS: tl.constexpr,
                                BLOCK_X: tl.constexpr,
                                EXPERT_NUM: tl.constexpr,
                                SCALE: tl.constexpr,
                            ):
    pid = tl.program_id(0)
    idx_begin = pid * BLOCK_SIZE
    idx_end = tl.minimum((pid + 1) * BLOCK_SIZE, INDICES_LENGTH)

    exp_offset = tl.arange(0, EXPERT_NUM)
    loaded_bins = tl.load(bins + exp_offset)

    for idx_in_block in range(0, BLOCK_SIZE, SUB_BLOCK_SIZE):
        idx_offset = idx_begin + idx_in_block
        if idx_offset < INDICES_LENGTH:
            idx_offsets = tl.arange(0, SUB_BLOCK_SIZE) + idx_offset
            idx_mask = idx_offsets < idx_end
            cur_indices = tl.load(indices + idx_offsets, idx_mask, other=0)
            cur_bin_ids = tl.load(bin_ids + idx_offsets, idx_mask, other=0)
            cur_sub_block_size = tl.minimum(SUB_BLOCK_SIZE, idx_end - idx_offset)

            base_bin_idx = tl.get_element(cur_bin_ids, (0,))
            last_bin_idx = tl.get_element(cur_bin_ids, (cur_sub_block_size - 1,))
            bin_count = last_bin_idx - base_bin_idx + 1

            begin = 0
            for i in range(bin_count):
                start_pos = 0
                if base_bin_idx > 0:
                    start_pos = tl.get_element(loaded_bins, (base_bin_idx - 1,))
                cur_bin = tl.get_element(loaded_bins, (base_bin_idx,))
                cur_bin_cut = tl.minimum(cur_bin, start_pos + expert_capacity)
                count = tl.minimum(cur_bin_cut - idx_offset - begin, cur_sub_block_size - begin)
                if count > 0:
                    x_offset = idx_offset + begin
                    if base_bin_idx > 0:
                        x_offset -= start_pos

                    for col_offset in range(0, NUM_COLUMNS, BLOCK_X):
                        col_offsets = tl.arange(0, BLOCK_X) + col_offset
                        col_mask = col_offsets < NUM_COLUMNS

                        idx_a = base_bin_idx * expert_capacity + x_offset
                        load_offsets = tl.arange(0, SUB_BLOCK_SIZE) + idx_a
                        load_mask = load_offsets < idx_a + count
                        cur_a = tl.load(a + (load_offsets[:, None] * NUM_COLUMNS + col_offsets[None, :]),
                                        load_mask[:, None] & col_mask[None, :])

                        end = begin + count
                        for j in range(begin, end):
                            idx = tl.get_element(cur_indices, (j,))
                            val = tl.extract_slice(cur_a, offsets=(j-begin, 0), sizes=(1, BLOCK_X), strides=(1, 1))
                            if SCALE:
                                scale = tl.load(weights + idx)
                                val = val.to(tl.float32) * scale.to(tl.float32)
                            tl.store(b + (idx * NUM_COLUMNS + col_offsets), val.to(b.dtype.element_ty).reshape(BLOCK_X), col_mask)

                base_bin_idx += 1
                begin = cur_bin - idx_offset