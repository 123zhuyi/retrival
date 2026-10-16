@triton.jit
def _scatter_wgrad_kernel(
    x,
    grads,
    wgrad,
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

    col_offsets = tl.arange(0, BLOCK_X)
    col_mask = col_offsets < NUM_COLUMNS
    for idx_in_block in range(0, BLOCK_SIZE, SUB_BLOCK_SIZE):
        idx_offset = idx_begin + idx_in_block
        if SUB_BLOCK_SIZE > 1:
            idx_offsets = tl.arange(0, SUB_BLOCK_SIZE) + idx_offset
            idx_mask = idx_offsets < idx_end
            cur_indices = tl.load(indices + idx_offsets, idx_mask, other=0)
            cur_x = tl.load(x + idx_offsets[:, None] * NUM_COLUMNS + col_offsets[None, :],
                            idx_mask[:, None] & col_mask[None, :], other=0)
            for i in range(0, SUB_BLOCK_SIZE):
                if i + idx_offset < idx_end:
                    idx = tl.get_element(cur_indices, (i,))
                    data = tl.extract_slice(cur_x, offsets=(i, 0), sizes=(1, BLOCK_X), strides=(1, 1)).to(tl.float32)
                    grad = tl.load(grads + (idx // TOP_K) * NUM_COLUMNS + col_offsets, col_mask, other=0).to(tl.float32)
                    out = tl.sum(data.reshape(BLOCK_X) * grad.reshape(BLOCK_X))
                    tl.store(wgrad+idx, out.to(wgrad.dtype.element_ty))

        elif idx_offset < idx_end:
            idx = tl.load(indices + idx_offset)
            acc = tl.zeros((BLOCK_X,), dtype=tl.float32)
            for col_offset in range(0, NUM_COLUMNS, BLOCK_X):
                data = tl.load(x + idx * NUM_COLUMNS + col_offsets, col_mask, other=0)
                grad = tl.load(grads + (idx // TOP_K) * NUM_COLUMNS + col_offsets, col_mask, other=0)
                acc += data.to(tl.float32) * grad.to(tl.float32)
            acc = tl.sum(acc)
            tl.store(wgrad + idx, acc.to(wgrad.dtype.element_ty))