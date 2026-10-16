@triton.jit(do_not_specialize=["seq_len"])
def _rope_forward_kernel(
    q_ptr,
    q_batch_stride,
    q_seq_stride,
    k_ptr,
    k_batch_stride,
    k_seq_stride,
    cos_ptr,
    cos_row_stride,
    sin_ptr,
    sin_row_stride,
    seq_len,
    num_seq_blocks,
    chunk_indices_ptr,
    kv_lens_ptr,
    bs: tl.constexpr,
    cos_bs: tl.constexpr,
    n_qh: tl.constexpr,
    n_kh: tl.constexpr,
    hd: tl.constexpr,
    nope_dim: tl.constexpr,
    rope_dim: tl.constexpr,
    half_rope_dim: tl.constexpr,
    TOKEN_BLOCK_SIZE: tl.constexpr,
    IS_VARLEN: tl.constexpr,
    HAS_KV_LENS: tl.constexpr,
    ALIGNED: tl.constexpr,
):
    pid = tl.program_id(axis=0)
    grid_size = tl.num_programs(axis=0)

    total_blocks = bs * num_seq_blocks

    for block_id in range(pid, total_blocks, grid_size):
        if IS_VARLEN:
            chunk_idx = tl.load(chunk_indices_ptr + block_id * 5 + 1)
            seq_start = tl.load(chunk_indices_ptr + block_id * 5 + 2)
            sin_cos_offset = tl.load(chunk_indices_ptr + block_id * 5 + 3)
            actual_seq_len = tl.load(chunk_indices_ptr + block_id * 5 + 4)

            block_start_seq_idx = chunk_idx * TOKEN_BLOCK_SIZE
            seq_offsets = block_start_seq_idx + tl.arange(0, TOKEN_BLOCK_SIZE)
            seq_mask = seq_offsets < actual_seq_len

            global_seq_offsets = seq_start + seq_offsets

            sin_cos_seq_offsets = sin_cos_offset + seq_offsets
            cos_token_ptr = cos_ptr + sin_cos_seq_offsets[:, None] * cos_row_stride
            sin_token_ptr = sin_ptr + sin_cos_seq_offsets[:, None] * sin_row_stride

            batch_idx = 0
        else:
            batch_idx = block_id // num_seq_blocks
            seq_block_id = block_id % num_seq_blocks

            block_start_seq_idx = seq_block_id * TOKEN_BLOCK_SIZE
            seq_offsets = block_start_seq_idx + tl.arange(0, TOKEN_BLOCK_SIZE)
            seq_mask = seq_offsets < seq_len

            global_seq_offsets = seq_offsets

            if HAS_KV_LENS:
                kv_len = tl.load(kv_lens_ptr + batch_idx)
                sin_cos_seq_offsets = kv_len + seq_offsets
                cos_token_ptr = cos_ptr + sin_cos_seq_offsets[:, None] * cos_row_stride
                sin_token_ptr = sin_ptr + sin_cos_seq_offsets[:, None] * sin_row_stride
            else:
                sin_cos_batch_offset = tl.where(cos_bs == 1, 0, batch_idx * seq_len)
                cos_token_ptr = cos_ptr + (sin_cos_batch_offset + seq_offsets[:, None]) * cos_row_stride
                sin_token_ptr = sin_ptr + (sin_cos_batch_offset + seq_offsets[:, None]) * sin_row_stride

        half_rope_dim_offsets = tl.arange(0, half_rope_dim)
        half_rope_dim_mask = half_rope_dim_offsets < half_rope_dim

        cos_block_2d = tl.load(
            cos_token_ptr + half_rope_dim_offsets[None, :],
            mask=seq_mask[:, None] & half_rope_dim_mask[None, :],
            other=0,
        )
        sin_block_2d = tl.load(
            sin_token_ptr + half_rope_dim_offsets[None, :],
            mask=seq_mask[:, None] & half_rope_dim_mask[None, :],
            other=0,
        )

        head_q_offsets = tl.arange(0, n_qh)
        head_k_offsets = tl.arange(0, n_kh)

        cos_tile = tl.reshape(cos_block_2d, (TOKEN_BLOCK_SIZE, 1, half_rope_dim), can_reorder=True)
        sin_tile = tl.reshape(sin_block_2d, (TOKEN_BLOCK_SIZE, 1, half_rope_dim), can_reorder=True)

        if ALIGNED:
            rope_dim_offsets = tl.arange(0, rope_dim)
            rope_dim_mask = rope_dim_offsets < rope_dim

            q_offsets = (
                batch_idx * q_batch_stride
                + global_seq_offsets[:, None, None] * q_seq_stride
                + head_q_offsets[None, :, None] * hd
                + nope_dim
                + rope_dim_offsets[None, None, :]
            )
            q_mask = seq_mask[:, None, None] & (head_q_offsets[None, :, None] < n_qh) & rope_dim_mask[None, None, :]

            q_tile = tl.load(q_ptr + q_offsets, mask=q_mask, other=0.0).to(sin_block_2d.dtype)
            q_tile = _compute_rope(q_tile, sin_tile, cos_tile, n_qh, half_rope_dim, TOKEN_BLOCK_SIZE, False)
            tl.store(q_ptr + q_offsets, q_tile, mask=q_mask)

            k_offsets = (
                batch_idx * k_batch_stride
                + global_seq_offsets[:, None, None] * k_seq_stride
                + head_k_offsets[None, :, None] * hd
                + nope_dim
                + rope_dim_offsets[None, None, :]
            )
            k_mask = seq_mask[:, None, None] & (head_k_offsets[None, :, None] < n_kh) & rope_dim_mask[None, None, :]

            k_tile = tl.load(k_ptr + k_offsets, mask=k_mask, other=0).to(sin_block_2d.dtype)
            k_tile = _compute_rope(k_tile, sin_tile, cos_tile, n_kh, half_rope_dim, TOKEN_BLOCK_SIZE, False)
            tl.store(k_ptr + k_offsets, k_tile, mask=k_mask)
        else:
            q_offsets_half1 = (
                batch_idx * q_batch_stride
                + global_seq_offsets[:, None, None] * q_seq_stride
                + head_q_offsets[None, :, None] * hd
                + nope_dim
                + half_rope_dim_offsets[None, None, :]
            )
            q_offsets_half2 = (
                batch_idx * q_batch_stride
                + global_seq_offsets[:, None, None] * q_seq_stride
                + head_q_offsets[None, :, None] * hd
                + nope_dim
                + half_rope_dim
                + half_rope_dim_offsets[None, None, :]
            )
            q_half_mask = (
                seq_mask[:, None, None] & (head_q_offsets[None, :, None] < n_qh) & half_rope_dim_mask[None, None, :]
            )

            q_tile_1 = tl.load(q_ptr + q_offsets_half1, mask=q_half_mask, other=0.0).to(sin_block_2d.dtype)
            q_tile_2 = tl.load(q_ptr + q_offsets_half2, mask=q_half_mask, other=0.0).to(sin_block_2d.dtype)
            new_q_1, new_q_2 = _compute_rope_separated(q_tile_1, q_tile_2, sin_tile, cos_tile, False)
            tl.store(q_ptr + q_offsets_half1, new_q_1, mask=q_half_mask)
            tl.store(q_ptr + q_offsets_half2, new_q_2, mask=q_half_mask)

            k_offsets_half1 = (
                batch_idx * k_batch_stride
                + global_seq_offsets[:, None, None] * k_seq_stride
                + head_k_offsets[None, :, None] * hd
                + nope_dim
                + half_rope_dim_offsets[None, None, :]
            )
            k_offsets_half2 = (
                batch_idx * k_batch_stride
                + global_seq_offsets[:, None, None] * k_seq_stride
                + head_k_offsets[None, :, None] * hd
                + nope_dim
                + half_rope_dim
                + half_rope_dim_offsets[None, None, :]
            )
            k_half_mask = (
                seq_mask[:, None, None] & (head_k_offsets[None, :, None] < n_kh) & half_rope_dim_mask[None, None, :]
            )

            k_tile_1 = tl.load(k_ptr + k_offsets_half1, mask=k_half_mask, other=0.0).to(sin_block_2d.dtype)
            k_tile_2 = tl.load(k_ptr + k_offsets_half2, mask=k_half_mask, other=0.0).to(sin_block_2d.dtype)
            new_k_1, new_k_2 = _compute_rope_separated(k_tile_1, k_tile_2, sin_tile, cos_tile, False)
            tl.store(k_ptr + k_offsets_half1, new_k_1, mask=k_half_mask)
            tl.store(k_ptr + k_offsets_half2, new_k_2, mask=k_half_mask)