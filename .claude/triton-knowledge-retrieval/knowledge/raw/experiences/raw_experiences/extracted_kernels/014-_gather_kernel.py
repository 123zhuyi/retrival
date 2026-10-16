@triton.jit
def _gather_kernel(
    x,
    out,
    indices,
    INDICES_LENGTH: tl.constexpr,
    BLOCK_SIZE: tl.constexpr,
    SUB_BLOCK_SIZE: tl.constexpr,
    NUM_COLUMNS: tl.constexpr,
    BLOCK_X: tl.constexpr,
    TOP_K: tl.constexpr,
):
    pid = tl.program_id(0)
    idx_begin = pid * BLOCK_SIZE
    idx_end = tl.minimum((pid + 1) * BLOCK_SIZE, INDICES_LENGTH)
    for idx_in_block in range(0, BLOCK_SIZE, SUB_BLOCK_SIZE):
        idx_offset = idx_begin + idx_in_block
        if idx_offset < idx_end:
            idx_offsets = tl.arange(0, SUB_BLOCK_SIZE) + idx_offset
            idx_mask = idx_offsets < idx_end
            cur_indices = tl.load(indices + idx_offsets, idx_mask, other=0)
            for col_offset in range(0, NUM_COLUMNS, BLOCK_X):
                tmp_buf = tl.zeros((SUB_BLOCK_SIZE, BLOCK_X), x.dtype.element_ty)
                col_offsets = tl.arange(0, BLOCK_X) + col_offset
                col_mask = col_offsets < NUM_COLUMNS
                for i in range(0, SUB_BLOCK_SIZE):
                    idx = tl.get_element(cur_indices, (i,)) // TOP_K * NUM_COLUMNS
                    val = tl.load(x + idx + col_offsets, col_mask)
                    tmp_buf = tl.insert_slice(tmp_buf, val[None,:], offsets=(i, 0), sizes=(1, BLOCK_X), strides=(1, 1))
                tl.store(out + idx_offsets[:, None] * NUM_COLUMNS + col_offsets[None, :],
                         tmp_buf, idx_mask[:, None] & col_mask[None, :])