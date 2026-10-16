@triton.jit
def _lightningattn_attn_decode_kernel(
    q_ptr,
    k_ptr,
    v_ptr,
    kv_cache_ptr,
    slope_rate,
    output_ptr,
    B: tl.constexpr,
    H: tl.constexpr,
    D: tl.constexpr,
    qkv_b_stride,
    qkv_h_stride,
    cache_b_stride,
    cache_h_stride,
    cache_d_stride,
    cache_e_stride,
    NUM_CORES: tl.constexpr,
    BLOCK_SIZE: tl.constexpr,
):
    """
    Kernel for lightning attention decoding with KV cache.
    """
    pid_c = tl.program_id(0)
    qk_d_offsets = tl.arange(0, D)
    v_d_offsets = tl.arange(0, BLOCK_SIZE)
    qk_mask = qk_d_offsets < D
    v_mask = v_d_offsets < D

    # preload qkv
    batch_id = pid_c // H
    head_id = pid_c % H
    q_offset = batch_id * qkv_b_stride + head_id * qkv_h_stride
    k_offset = batch_id * qkv_b_stride + head_id * qkv_h_stride
    v_offset = batch_id * qkv_b_stride + head_id * qkv_h_stride
    q = tl.load(q_ptr + q_offset + qk_d_offsets, mask=qk_mask, other=0.0)
    k = tl.load(k_ptr + k_offset + qk_d_offsets, mask=qk_mask, other=0.0)
    v = tl.load(v_ptr + v_offset + v_d_offsets, mask=v_mask, other=0.0)

    for pid_bh in range(pid_c, B * H, NUM_CORES):
        pid_b = pid_bh // H
        pid_h = pid_bh % H
        for pid_d in range(D // BLOCK_SIZE):
            batch_id = pid_b
            head_id = pid_h

            # Load decay rate for the current head
            ratio = tl.load(slope_rate + pid_h)

            # Calculate offsets for dimensions
            qk_d_offsets = tl.arange(0, D)
            v_d_offsets = tl.arange(0, BLOCK_SIZE) + pid_d * BLOCK_SIZE
            cache_d_offsets = (
                qk_d_offsets[:, None] * cache_d_stride
                + v_d_offsets[None, :] * cache_e_stride
            )

            # Calculate offsets for the current batch and head
            q_offset = batch_id * qkv_b_stride + head_id * qkv_h_stride
            k_offset = batch_id * qkv_b_stride + head_id * qkv_h_stride
            v_offset = batch_id * qkv_b_stride + head_id * qkv_h_stride

            cache_offset = batch_id * cache_b_stride + head_id * cache_h_stride

            # Create masks for loading tensors
            qk_mask = qk_d_offsets < D
            v_mask = v_d_offsets < D

            # Load query, key, and value tensors
            q = tl.load(q_ptr + q_offset + qk_d_offsets, mask=qk_mask, other=0.0)
            k = tl.load(k_ptr + k_offset + qk_d_offsets, mask=qk_mask, other=0.0)
            v = tl.load(v_ptr + v_offset + v_d_offsets, mask=v_mask, other=0.0)

            # Compute key-value outer product
            kv_outer = k[:, None] * v[None, :]
            kv_mask = qk_mask[:, None] & v_mask[None, :]

            # Apply decay to previous KV cache
            ratio = tl.exp(-ratio)
            kv_ptr = kv_cache_ptr + cache_offset + cache_d_offsets
            kv_cache_old = tl.load(kv_ptr, mask=kv_mask, other=0.0)
            kv_outer = kv_outer + ratio * kv_cache_old

            # Compute attention output
            output = q[:, None].to(tl.float32) * kv_outer
            output = tl.sum(output, axis=0)
            tl.store(kv_ptr, kv_outer, mask=kv_mask)
            tl.store(output_ptr + q_offset + v_d_offsets, output, mask=v_mask)