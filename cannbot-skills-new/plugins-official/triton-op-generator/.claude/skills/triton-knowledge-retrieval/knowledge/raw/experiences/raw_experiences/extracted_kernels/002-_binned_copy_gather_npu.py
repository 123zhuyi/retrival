@triton.jit
def _binned_copy_gather_npu(
                            a: torch.Tensor,
                            b: torch.Tensor,
                            indices: torch.Tensor,
                            bins: torch.Tensor,
                            bin_ids: torch.Tensor,
                            expert_capacity: tl.constexpr,
                            INDICES_LENGTH: tl.constexpr,
                            BLOCK_SIZE: tl.constexpr,
                            SUB_BLOCK_SIZE: tl.constexpr,
                            NUM_COLUMNS: tl.constexpr,
                            BLOCK_X: tl.constexpr,
                            TOP_K: tl.constexpr,
                        ):
    idx_begin = tl.program_id(0) * BLOCK_SIZE

    for idx_in_block in range(0, BLOCK_SIZE, SUB_BLOCK_SIZE):
        idx_offset = idx_begin + idx_in_block
        idx_offsets = tl.arange(0, SUB_BLOCK_SIZE) + idx_offset
        idx_mask = idx_offsets < INDICES_LENGTH
        if idx_offset < INDICES_LENGTH:
            cur_indices = tl.load(indices + idx_offsets, idx_mask)
            cur_bin_ids = tl.load(bin_ids + idx_offsets, idx_mask)
            base_bin_idx = tl.get_element(cur_bin_ids, (0,))

            for col_offset in range(0, NUM_COLUMNS, BLOCK_X):
                tmp_buf = tl.zeros((SUB_BLOCK_SIZE, BLOCK_X), a.dtype.element_ty)
                col_offsets = tl.arange(0, BLOCK_X) + col_offset
                col_mask = col_offsets < NUM_COLUMNS

                offset_in_bin = idx_offset
                if base_bin_idx > 0:
                    offset_in_bin -= tl.load(bins + base_bin_idx - 1)
                idx_b = base_bin_idx * expert_capacity + offset_in_bin

                COUNT = 0
                for item_offset in range(0, SUB_BLOCK_SIZE):
                    inner_offset = idx_offset + item_offset
                    if inner_offset < (idx_begin + BLOCK_SIZE):
                        bin_idx = tl.get_element(cur_bin_ids, (item_offset,))

                        if bin_idx != base_bin_idx:
                            b_offsets = tl.arange(0, SUB_BLOCK_SIZE) + idx_b
                            b_mask = b_offsets < idx_b + COUNT
                            tl.store(b + (b_offsets[:, None] * NUM_COLUMNS + col_offsets[None, :]),
                                    tmp_buf, b_mask[:, None] & col_mask[None, :])

                            # 更新起始偏移量和当前专家号
                            base_bin_idx = bin_idx
                            offset_in_bin = inner_offset
                            if base_bin_idx > 0:
                                offset_in_bin -= tl.load(bins + base_bin_idx - 1)
                            idx_b = base_bin_idx * expert_capacity + offset_in_bin
                            COUNT = 0

                        if offset_in_bin + COUNT < expert_capacity:
                            idx_a = (tl.get_element(cur_indices, (item_offset,)) // TOP_K) * NUM_COLUMNS
                            val = tl.load(a + idx_a + col_offsets, col_mask)

                            tmp_buf = tl.insert_slice(tmp_buf, val[None, :], offsets=(COUNT, 0), sizes=(1, BLOCK_X),
                                                        strides=(1, 1))
                            COUNT += 1
                # end for SUB_BLOCK_SIZE
                # store last block
                if COUNT != 0:
                    b_offsets = tl.arange(0, SUB_BLOCK_SIZE) + idx_b
                    b_mask = b_offsets < idx_b + COUNT
                    tl.store(b + (b_offsets[:, None] * NUM_COLUMNS + col_offsets[None, :]),
                            tmp_buf, b_mask[:, None] & col_mask[None, :])