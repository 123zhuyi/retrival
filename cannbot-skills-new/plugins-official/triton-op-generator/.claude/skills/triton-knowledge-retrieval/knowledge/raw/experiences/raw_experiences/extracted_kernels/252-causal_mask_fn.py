@triton.jit
def causal_mask_fn(
    mask_ptr, mask_size, mask_stride_m, mask_stride_n, q_start, kv_start, q_end, kv_end, Q_BLOCK, KV_BLOCK
):
    offset_causal = min(max(kv_start - q_start, -mask_size), mask_size)
    offsets_mask_causal = (
        (tl.arange(0, Q_BLOCK)[:, None]) * mask_stride_m
        + (mask_size + offset_causal + tl.arange(0, KV_BLOCK)[None, :]) * mask_stride_n
    )
    mask_causal = tl.load(mask_ptr + offsets_mask_causal).to(tl.int1)

    return mask_causal