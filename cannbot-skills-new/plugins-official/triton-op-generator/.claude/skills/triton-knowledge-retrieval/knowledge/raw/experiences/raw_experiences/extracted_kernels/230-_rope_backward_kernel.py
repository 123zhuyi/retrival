@triton.jit(do_not_specialize=["seq_len"])
def _rope_backward_kernel(
    dq_ptr,
    dq_batch_stride,
    dq_seq_stride,
    dk_ptr,
    dk_batch_stride,
    dk_seq_stride,
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

        cos_tile = tl.reshape(cos_block_2d, (TOKEN_BLOCK_SIZE, 1, half_rope_dim), can_reorder=True)
        sin_tile = tl.reshape(sin_block_2d, (TOKEN_BLOCK_SIZE, 1, half_rope_dim), can_reorder=True)

        head_q_offsets = tl.arange(0, n_qh)
        head_k_offsets = tl.arange(0, n_kh)

        if ALIGNED:
            rope_dim_offsets = tl.arange(0, rope_dim)
            rope_dim_mask = rope_dim_offsets < rope_dim

            dq_offsets = (
                batch_idx * dq_batch_stride
                + global_seq_offsets[:, None, None] * dq_seq_stride
                + head_q_offsets[None, :, None] * hd
                + nope_dim
                + rope_dim_offsets[None, None, :]
            )
            q_mask = seq_mask[:, None, None] & (head_q_offsets[None, :, None] < n_qh) & rope_dim_mask[None, None, :]

            dq_tile = tl.load(dq_ptr + dq_offsets, mask=q_mask, other=0).to(sin_block_2d.dtype)
            dq_tile = _compute_rope(dq_tile, sin_tile, cos_tile, n_qh, half_rope_dim, TOKEN_BLOCK_SIZE, True)
            tl.store(dq_ptr + dq_offsets, dq_tile, mask=q_mask)

            dk_offsets = (
                batch_idx * dk_batch_stride
                + global_seq_offsets[:, None, None] * dk_seq_stride
                + head_k_offsets[None, :, None] * hd
                + nope_dim
                + rope_dim_offsets[None, None, :]
            )
            k_mask = seq_mask[:, None, None] & (head_k_offsets[None, :, None] < n_kh) & rope_dim_mask[None, None, :]

            dk_tile = tl.load(dk_ptr + dk_offsets, mask=k_mask, other=0).to(sin_block_2d.dtype)
            dk_tile = _compute_rope(dk_tile, sin_tile, cos_tile, n_kh, half_rope_dim, TOKEN_BLOCK_SIZE, True)
            tl.store(dk_ptr + dk_offsets, dk_tile, mask=k_mask)
        else:
            dq_offsets_half1 = (
                batch_idx * dq_batch_stride
                + global_seq_offsets[:, None, None] * dq_seq_stride
                + head_q_offsets[None, :, None] * hd
                + nope_dim
                + half_rope_dim_offsets[None, None, :]
            )
            dq_offsets_half2 = (
                batch_idx * dq_batch_stride
                + global_seq_offsets[:, None, None] * dq_seq_stride
                + head_q_offsets[None, :, None] * hd
                + nope_dim
                + half_rope_dim
                + half_rope_dim_offsets[None, None, :]
            )
            q_half_mask = (
                seq_mask[:, None, None] & (head_q_offsets[None, :, None] < n_qh) & half_rope_dim_mask[None, None, :]
            )

            dq_tile_1 = tl.load(dq_ptr + dq_offsets_half1, mask=q_half_mask, other=0.0).to(sin_block_2d.dtype)
            dq_tile_2 = tl.load(dq_ptr + dq_offsets_half2, mask=q_half_mask, other=0.0).to(sin_block_2d.dtype)
            new_dq_1, new_dq_2 = _compute_rope_separated(dq_tile_1, dq_tile_2, sin_tile, cos_tile, True)
            tl.store(dq_ptr + dq_offsets_half1, new_dq_1, mask=q_half_mask)
            tl.store(dq_ptr + dq_offsets_half2, new_dq_2, mask=q_half_mask)

            dk_offsets_half1 = (
                batch_idx * dk_batch_stride
                + global_seq_offsets[:, None, None] * dk_seq_stride
                + head_k_offsets[None, :, None] * hd
                + nope_dim
                + half_rope_dim_offsets[None, None, :]
            )
            dk_offsets_half2 = (
                batch_idx * dk_batch_stride
                + global_seq_offsets[:, None, None] * dk_seq_stride
                + head_k_offsets[None, :, None] * hd
                + nope_dim
                + half_rope_dim
                + half_rope_dim_offsets[None, None, :]
            )
            k_half_mask = (
                seq_mask[:, None, None] & (head_k_offsets[None, :, None] < n_kh) & half_rope_dim_mask[None, None, :]
            )

            dk_tile_1 = tl.load(dk_ptr + dk_offsets_half1, mask=k_half_mask, other=0.0).to(sin_block_2d.dtype)
            dk_tile_2 = tl.load(dk_ptr + dk_offsets_half2, mask=k_half_mask, other=0.0).to(sin_block_2d.dtype)
            new_dk_1, new_dk_2 = _compute_rope_separated(dk_tile_1, dk_tile_2, sin_tile, cos_tile, True)
            tl.store(dk_ptr + dk_offsets_half1, new_dk_1, mask=k_half_mask)
            tl.store(dk_ptr + dk_offsets_half2, new_dk_2, mask=k_half_mask)