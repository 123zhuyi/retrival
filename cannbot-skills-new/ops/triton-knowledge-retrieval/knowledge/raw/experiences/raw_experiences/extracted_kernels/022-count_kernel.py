@triton.jit
def count_kernel(
    x_ptr,  # [num_kv_heads, total_len, topk]
    y_ptr,  # [num_kv_heads, total_blocks]
    cu_seqlens,  # [batch_size + 1]
    cu_seqblocks,  # [batch_size + 1]
    topk,
    stride_xh,
    stride_xn,
    stride_xk,
    stride_yh,
    stride_yn,
    BLOCK_SIZE_N: tl.constexpr,
    BLOCK_SIZE_K: tl.constexpr,
    BLOCK_SIZE_R: tl.constexpr,
):
    pid_h = tl.program_id(0)
    pid_b = tl.program_id(1)
    # get start and len after rmpad
    seq_start = tl.load(cu_seqlens + pid_b)
    seq_len = tl.load(cu_seqlens + pid_b + 1) - seq_start
    blocks_start = tl.load(cu_seqblocks + pid_b)
    num_blocks = tl.load(cu_seqblocks + pid_b + 1) - blocks_start
    # load x
    off_k = tl.arange(0, BLOCK_SIZE_K)
    off_n = tl.arange(0, BLOCK_SIZE_N)
    x_ptr = x_ptr + pid_h * stride_xh + seq_start * stride_xn

    # loop
    for j in range(0, num_blocks, BLOCK_SIZE_R):
        y = tl.zeros((BLOCK_SIZE_R,), dtype=tl.int32)
        for i in range(0, seq_len, BLOCK_SIZE_N):
            mask = (off_n < seq_len - i)[:, None] & (off_k < topk)[None, :]
            x = tl.load(
                x_ptr
                + off_n[:, None] * stride_xn
                + off_k[None, :] * stride_xk
                + i * stride_xn,
                mask=mask,
                other=-1,
            )
            x = tl.where(mask, x, -1)
            x = tl.ravel(x)
            bins = tl.arange(0, BLOCK_SIZE_R) + j
            # Create comparison mask: x[i] == bins[j]
            equal_mask = x[:, None] == bins[None, :]  # shape: (len(x), BLOCK_SIZE_R)
            # Only count valid indices (e.g., not -1)
            valid_mask = x[:, None] >= 0
            count_mask = tl.where(valid_mask, equal_mask, 0)
            # Sum over the x dimension to get count per bin
            hist = tl.sum(count_mask, axis=0)  # (BLOCK_SIZE_R,)
            # Accumulate
            y += hist
        # store result
        off_r = tl.arange(0, BLOCK_SIZE_R) + j
        y_ptrs = (
            y_ptr + pid_h * stride_yh + blocks_start * stride_yn + off_r * stride_yn
        )
        tl.store(y_ptrs, y.to(y_ptr.dtype.element_ty), mask=off_r < num_blocks)