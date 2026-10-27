@triton.jit
def _binned_copy_wgrad_npu(
                            x,
                            grad,
                            wgrad,
                            expert_capacity,
                            indices,
                            bins,
                            NUM_COLUMNS: tl.constexpr,
                            TOP_K: tl.constexpr,
                            BLOCK_X: tl.constexpr,
                            BLOCK_SIZE: tl.constexpr,
                            ):
    expert_idx = tl.program_id(0)
    entry_idx = tl.program_id(1)

    bins_start = 0
    if expert_idx > 0:
        bins_start = tl.load(bins + expert_idx - 1)
    bins_end = tl.load(bins + expert_idx)
    expert_num_tokens = bins_end - bins_start

    for idx_in_block in range(0, BLOCK_SIZE):
        # Calculate our offset into the output.
        index_x = expert_idx * expert_capacity + entry_idx * BLOCK_SIZE + idx_in_block
        out_token_num = entry_idx * BLOCK_SIZE + idx_in_block + 1
        if not (out_token_num > expert_num_tokens):
            index_out = tl.load(indices + bins_start + entry_idx * BLOCK_SIZE + idx_in_block)

            wgrad_ptr = wgrad + index_out
            grad_ptr = grad + tl.multiple_of((index_out // TOP_K) * NUM_COLUMNS, NUM_COLUMNS)
            x_ptr = x + tl.multiple_of(index_x * NUM_COLUMNS, NUM_COLUMNS)
            offsets = tl.max_contiguous(tl.arange(0, BLOCK_X), BLOCK_X)

            acc = tl.zeros((BLOCK_X,), dtype=tl.float32)
            iterations = tl.cdiv(NUM_COLUMNS, BLOCK_X)

            for _ in range(iterations):
                mask = offsets < NUM_COLUMNS
                data = tl.load(x_ptr + offsets, mask=mask).to(tl.float32)
                scale = tl.load(grad_ptr + offsets, mask=mask).to(tl.float32)
                acc += data * scale
                offsets += BLOCK_X

            # Reduce to get the final result and store.
            out = tl.sum(acc).to(wgrad.dtype.element_ty)
            tl.store(wgrad_ptr, out)