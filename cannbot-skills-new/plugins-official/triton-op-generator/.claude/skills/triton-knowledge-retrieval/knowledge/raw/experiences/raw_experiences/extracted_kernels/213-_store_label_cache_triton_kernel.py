@triton.jit
def _store_label_cache_triton_kernel(
    label_cache_ptr,
    key_lr_ptr,
    block_idx_list_ptr,
    token_idx_list_ptr,
    head_num: tl.constexpr,
    head_dim: tl.constexpr,
    token_num: tl.constexpr,
    l_stride_b: tl.constexpr,
    l_stride_h: tl.constexpr,
    l_stride_t: tl.constexpr,
    l_stride_d: tl.constexpr,
    k_stride_s: tl.constexpr,
    k_stride_h: tl.constexpr,
    k_stride_d: tl.constexpr,
    BATCH_BLOCK_NUM: tl.constexpr,
):
    pid_b = tl.program_id(0)
    b_start = pid_b * BATCH_BLOCK_NUM
    b_end = tl.minimum(b_start + BATCH_BLOCK_NUM, token_num)
    b = tl.arange(0, BATCH_BLOCK_NUM) + b_start
    b_3d = b[:, None, None]
    h = tl.arange(0, head_num)
    h_3d = h[None, :, None]
    d = tl.arange(0, head_dim)
    d_3d = d[None, None, :]
    block_idx = tl.load(block_idx_list_ptr + b_3d, mask=(b_3d < b_end), other=0)
    token_idx = tl.load(token_idx_list_ptr + b_3d, mask=(b_3d < b_end), other=0)

    label_cache_addr = block_idx * l_stride_b + h_3d * l_stride_h + token_idx * l_stride_t + d_3d * l_stride_d
    key_lr_offset = b_3d * k_stride_s + h_3d * k_stride_h + d_3d * k_stride_d

    valid_mask = (b_3d < b_end) & (h_3d < head_num) & (d_3d < head_dim)

    key_lr_data = tl.load(key_lr_ptr + key_lr_offset, mask=valid_mask, other=0.0)
    tl.store(label_cache_ptr + label_cache_addr, key_lr_data, mask=valid_mask)