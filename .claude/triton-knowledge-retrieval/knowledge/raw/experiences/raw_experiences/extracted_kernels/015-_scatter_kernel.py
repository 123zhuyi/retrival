@triton.jit
def _scatter_kernel(
    x,
    out,
    weights,
    indices,
    INDICES_LENGTH: tl.constexpr,
    BLOCK_SIZE: tl.constexpr,
    SUB_BLOCK_SIZE: tl.constexpr,
    NUM_COLUMNS: tl.constexpr,
    BLOCK_X: tl.constexpr,
    TOP_K: tl.constexpr,
    SCALE:  tl.constexpr,
):
    pid = tl.program_id(0)
    idx_begin = pid * BLOCK_SIZE
    idx_end = tl.minimum((pid + 1) * BLOCK_SIZE, INDICES_LENGTH)
    for idx_in_block in range(0, BLOCK_SIZE, SUB_BLOCK_SIZE):
        idx_offset = idx_begin + idx_in_block
        idx_offsets = tl.arange(0, SUB_BLOCK_SIZE) + idx_offset
        idx_mask = idx_offsets < idx_end
        cur_indices = tl.load(indices + idx_offsets, idx_mask, other=0)
        for col_offset in range(0, NUM_COLUMNS, BLOCK_X):
            col_offsets = tl.arange(0, BLOCK_X) + col_offset
            col_mask = col_offsets < NUM_COLUMNS
            cur_x = tl.load(x + idx_offsets[:, None] * NUM_COLUMNS + col_offsets[None, :],
                            idx_mask[:, None] & col_mask[None, :])
            for i in range(0, SUB_BLOCK_SIZE):
                if i + idx_offset < idx_end:
                    idx = tl.get_element(cur_indices, (i,))
                    val = tl.extract_slice(cur_x, offsets=(i, 0), sizes=(1, BLOCK_X), strides=(1, 1))
                    if SCALE:
                        scale = tl.load(weights + idx)
                        val = val.to(tl.float32) * scale.to(tl.float32)

                    tl.store(out + idx * NUM_COLUMNS + col_offsets, val.to(out.dtype.element_ty).reshape(BLOCK_X), col_mask)