@triton.jit
def _mhc_post_kernel(
    comb_res_mix_ptr,
    residual_ptr,
    post_layer_mix_ptr,
    x_ptr,
    out_ptr,
    stride_comb_n,
    stride_comb_hc1,
    stride_comb_hc2,
    stride_residual_n,
    stride_residual_hc,
    stride_residual_h,
    stride_post_layer_mix_n,
    stride_post_layer_mix_hc,
    stride_xn,
    stride_xh,
    stride_out_n,
    stride_out_hc,
    stride_out_h,
    N,
    H: tl.constexpr,
    HC: tl.constexpr,
    H_BLK: tl.constexpr,
    NUM_CORES: tl.constexpr,
):
    core_idx = tl.program_id(0)
    offs_hc1 = tl.arange(0, HC)[:, None]
    offs_j = tl.arange(0, HC)[None, :]
    offs_hblk = tl.arange(0, H_BLK)[None, :]

    for pid_n in range(core_idx, N, NUM_CORES):
        a_ptrs = (
            comb_res_mix_ptr
            + pid_n * stride_comb_n
            + offs_hc1 * stride_comb_hc1
            + offs_j * stride_comb_hc2
        )
        a_data = tl.trans(tl.load(a_ptrs))
        post_layer_mix_ptrs = (
            post_layer_mix_ptr
            + pid_n * stride_post_layer_mix_n
            + tl.arange(0, HC) * stride_post_layer_mix_hc
        )
        post_layer_mix_data = tl.load(post_layer_mix_ptrs)
        n_blocks = tl.cdiv(H, H_BLK)
        for blk_idx in range(n_blocks):
            start_h = blk_idx * H_BLK
            residual_offs_h = start_h + offs_hblk
            residual_ptrs = (
                residual_ptr
                + pid_n * stride_residual_n
                + offs_hc1 * stride_residual_hc
                + residual_offs_h * stride_residual_h
            )
            residual_data = tl.load(residual_ptrs, mask=residual_offs_h < H)
            dot_res = tl.dot(a_data, residual_data.to(tl.float32))
            x_offs_h = start_h + tl.arange(0, H_BLK)
            x_ptrs = x_ptr + pid_n * stride_xn + x_offs_h * stride_xh
            x_data = tl.load(x_ptrs, mask=x_offs_h < H, other=0.0)
            result = (post_layer_mix_data[:, None] * x_data[None, :]) + dot_res
            out_ptrs = (
                out_ptr
                + pid_n * stride_out_n
                + offs_hc1 * stride_out_hc
                + (start_h + offs_hblk) * stride_out_h
            )
            tl.store(out_ptrs, result.to(tl.bfloat16), mask=offs_hblk < (H - start_h))