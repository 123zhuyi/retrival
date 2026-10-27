@triton.jit
def add_gemma_rms_norm_kernel(
    hidden_state_ptr,
    hidden_state_stride_bs,
    weight_ptr,
    residual_ptr,
    add_output_ptr,
    norm_output_ptr,
    variance_epsilon,
    batch,
    dim: tl.constexpr,
    BLOCK_M: tl.constexpr,
):
    core_id = tl.program_id(0)
    core_num = tl.num_programs(0)
    batch_per_core = tl.cdiv(batch, core_num)
    start_batch = core_id * batch_per_core
    end_batch = tl.minimum(start_batch + batch_per_core, batch)
    offset_d = tl.arange(0, dim)

    for row_start in tl.range(start_batch, end_batch, BLOCK_M):
        offset_row = row_start + tl.arange(0, BLOCK_M)
        offset_hidden = offset_row[:, None] * hidden_state_stride_bs + offset_d[None, :]
        mask_hidden = offset_row < batch
        mask_bs = mask_hidden[:, None]

        x = tl.load(hidden_state_ptr + offset_hidden, mask=mask_bs)
        residual = tl.load(residual_ptr + offset_hidden, mask=mask_bs)
        add_val = x + residual
        tl.store(add_output_ptr + offset_hidden, add_val, mask=mask_bs)

        x_fp32 = add_val.to(tl.float32)
        w = tl.load(weight_ptr + offset_d).to(tl.float32)
        variance = tl.sum(x_fp32 * x_fp32, axis=-1) / dim
        x_fp32 = x_fp32 * tl.rsqrt(variance[:, None] + variance_epsilon)
        x_fp32 = x_fp32 * (w + 1.0)
        output = x_fp32.to(x.dtype)
        tl.store(norm_output_ptr + offset_hidden, output, mask=mask_bs)