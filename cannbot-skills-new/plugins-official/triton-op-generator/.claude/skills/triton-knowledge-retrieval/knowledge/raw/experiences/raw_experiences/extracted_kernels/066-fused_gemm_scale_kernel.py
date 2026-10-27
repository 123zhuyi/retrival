@triton.jit
def fused_gemm_scale_kernel(x_ptr, weight_ptr, bias_ptr, scale_ptr,
    output_ptr, batch_size, in_features, out_features, stride_xb, stride_xf,
    stride_wf, stride_wo, stride_ob, stride_of, BLOCK_SIZE_M: tl.constexpr,
    BLOCK_SIZE_N: tl.constexpr, BLOCK_SIZE_K: tl.constexpr):
    pid0 = tl.program_id(0)
    pid1 = tl.program_id(1)
    batch_offsets = pid0 * BLOCK_SIZE_M + tl.arange(0, BLOCK_SIZE_M)
    feat_offsets = pid1 * BLOCK_SIZE_N + tl.arange(0, BLOCK_SIZE_N)
    batch_mask = batch_offsets < batch_size
    feat_mask = feat_offsets < out_features
    acc = tl.zeros((BLOCK_SIZE_M, BLOCK_SIZE_N), dtype=tl.float32)
    for k in range(0, in_features, BLOCK_SIZE_K):
        k_offsets = k + tl.arange(0, BLOCK_SIZE_K)
        k_mask = k_offsets < in_features
        x_vals = tl.load(x_ptr + batch_offsets[:, None] * stride_xb + 
            k_offsets[None, :] * stride_xf, mask=batch_mask[:, None] &
            k_mask[None, :], other=0.0)
        w_vals = tl.load(weight_ptr + k_offsets[:, None] * stride_wf + 
            feat_offsets[None, :] * stride_wo, mask=k_mask[:, None] &
            feat_mask[None, :], other=0.0)
        acc += tl.dot(x_vals, w_vals)
    bias_vals = tl.load(bias_ptr + feat_offsets, mask=feat_mask, other=0.0)
    acc += bias_vals[None, :]
    scale_vals = tl.load(scale_ptr + feat_offsets, mask=feat_mask, other=0.0)
    acc *= scale_vals[None, :]
    output_ptrs = output_ptr + batch_offsets[:, None
        ] * stride_ob + feat_offsets[None, :] * stride_of
    tl.store(output_ptrs, acc, mask=batch_mask[:, None] & feat_mask[None, :])