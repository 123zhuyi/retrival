@triton.jit
def _mhc_pre_gemm_sqrsum_kernel(
    x_ptr,  # (num_tokens, hc_hidden_size), bfloat16
    fn_ptr,  # (hc_mult3, hc_hidden_size), float32
    out_ptr,  # (num_tokens, hc_mult3), float32
    sqrsum_ptr,  # (num_tokens,), float32
    num_tokens,
    hc_mult3,  # <= 32
    hc_hidden_size,
    token_block: tl.constexpr = 32,
    hidden_block: tl.constexpr = 256,
):
    pid = tl.program_id(0)
    offs_m = pid * token_block + tl.arange(0, token_block)
    mask_m = offs_m < num_tokens
    acc_sqr = tl.zeros((token_block,), dtype=tl.float32)
    acc_out = tl.zeros((token_block, 32), dtype=tl.float32)
    for k in range(0, hc_hidden_size, hidden_block):
        offs_k = k + tl.arange(0, hidden_block)
        x_ptrs = x_ptr + offs_m[:, None] * hc_hidden_size + offs_k[None, :]
        x_block_bf16 = tl.load(x_ptrs, mask=mask_m[:, None], other=0.0)
        x_block_f32 = x_block_bf16.to(tl.float32)
        block_sqr = tl.sum(x_block_f32 * x_block_f32, axis=1)
        acc_sqr += block_sqr
        offs_n = tl.arange(0, 32)
        mask_fn_row = offs_n < hc_mult3
        fn_ptrs = fn_ptr + offs_n[:, None] * hc_hidden_size + offs_k[None, :]
        fn_block = tl.load(fn_ptrs, mask=mask_fn_row[:, None], other=0.0)
        fn_block_T = tl.trans(fn_block)
        block_out = tl.dot(x_block_f32, fn_block_T, allow_tf32=True)
        acc_out += block_out

    sqrsum_ptrs = sqrsum_ptr + offs_m
    tl.store(sqrsum_ptrs, acc_sqr, mask=mask_m)
    offs_n_out = tl.arange(0, 32)
    out_ptrs = out_ptr + offs_m[:, None] * hc_mult3 + offs_n_out[None, :]
    mask_out = mask_m[:, None] & (offs_n_out[None, :] < hc_mult3)
    tl.store(out_ptrs, acc_out, mask=mask_out)