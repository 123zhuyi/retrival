@triton.jit
def rms_norm_ckv_kernel(
    X,
    W,
    O,
    n_rows,
    input_row_stride: tl.constexpr,
    output_row_stride: tl.constexpr,
    eps: tl.constexpr,
    KV_LORA_RANK: tl.constexpr,
    BLOCK: tl.constexpr,
    NUM_CORES: tl.constexpr,
):
    # X是matmul_ckv_kr_out，加载kv_lora_rank部分进行rms norm
    # X shape: [b, seq, 1, kv_lora_rank + qk_rope_head_dim]
    pid = tl.program_id(0)
    dim_range = tl.arange(0, KV_LORA_RANK)
    w = tl.load(W + dim_range)
    w = tl.expand_dims(w, 0)
    w = tl.broadcast_to(w, (BLOCK, KV_LORA_RANK))
    NUM_BLOCKS = tl.cdiv(n_rows, BLOCK)
    for row_block_id in range(pid, NUM_BLOCKS, NUM_CORES):
        pos_offset = row_block_id * BLOCK + tl.arange(0, BLOCK)
        pos_mask = (pos_offset < n_rows)[:, None]
        x = tl.load(
            X + pos_offset[:, None] * input_row_stride + dim_range[None, :],
            mask=pos_mask,
        )
        xf = x.to(tl.float32)
        var = tl.sum(xf * xf, 1) / KV_LORA_RANK
        qrt = tl.expand_dims(tl.math.rsqrt(var + eps), 1)
        out = xf * tl.broadcast_to(qrt, (BLOCK, KV_LORA_RANK))
        out = w * out.to(x.dtype)
        tl.store(
            O + pos_offset[:, None] * output_row_stride + dim_range[None, :],
            out,
            mask=pos_mask,
        )