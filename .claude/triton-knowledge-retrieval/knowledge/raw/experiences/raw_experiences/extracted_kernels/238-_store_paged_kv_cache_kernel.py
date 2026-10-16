@triton.jit
def _store_paged_kv_cache_kernel(
    k_ptr,
    v_ptr,
    key_cache_ptr,
    value_cache_ptr,
    block_table_ptr,
    cu_seqlens_ptr,
    chunk_indices_ptr,
    stride_k_tok,
    stride_k_head,
    stride_k_dim,
    stride_v_tok,
    stride_v_head,
    stride_v_dim,
    stride_kc_blk,
    stride_kc_head,
    stride_kc_tok,
    stride_kc_dim,
    stride_vc_blk,
    stride_vc_head,
    stride_vc_tok,
    stride_vc_dim,
    stride_bt_batch,
    stride_bt_blk,
    num_kv_heads,
    head_dim: tl.constexpr,
    block_size: tl.constexpr,
    total_chunks,
    CHUNK_SIZE: tl.constexpr,
):
    pid = tl.program_id(0)
    num_programs = tl.num_programs(0)

    for chunk_id_linear in range(pid, total_chunks, num_programs):
        meta_ptr = chunk_indices_ptr + chunk_id_linear * 3
        batch_idx = tl.load(meta_ptr)
        token_offset_in_seq = tl.load(meta_ptr + 1)
        logical_kv_start = tl.load(meta_ptr + 2)

        seq_start_tok = tl.load(cu_seqlens_ptr + batch_idx)
        global_token_idx = seq_start_tok + token_offset_in_seq

        seq_len_curr = tl.load(cu_seqlens_ptr + batch_idx + 1) - seq_start_tok
        valid_len = seq_len_curr - token_offset_in_seq

        curr_log_pos = logical_kv_start
        curr_kv_pos = global_token_idx

        remain_chunk_len = CHUNK_SIZE
        remain_chunk_len = tl.minimum(remain_chunk_len, valid_len)

        processed = 0
        while processed < remain_chunk_len:
            block_table_idx = curr_log_pos // block_size
            block_inner_off = curr_log_pos % block_size

            physical_block_id = tl.load(block_table_ptr + batch_idx * stride_bt_batch + block_table_idx * stride_bt_blk)

            space_in_block = block_size - block_inner_off
            sub_len = tl.minimum(remain_chunk_len - processed, space_in_block).to(tl.int32)

            offs_sub = tl.arange(0, CHUNK_SIZE)
            mask_sub = offs_sub < sub_len

            offs_d = tl.arange(0, head_dim)

            for h in range(num_kv_heads):
                src_k_ptr = (
                    k_ptr
                    + (curr_kv_pos + offs_sub[:, None]) * stride_k_tok
                    + h * stride_k_head
                    + offs_d[None, :] * stride_k_dim
                )

                k_val = tl.load(src_k_ptr, mask=mask_sub[:, None], other=0.0)

                dst_k_ptr = (
                    key_cache_ptr
                    + physical_block_id * stride_kc_blk
                    + h * stride_kc_head
                    + (block_inner_off + offs_sub[:, None]) * stride_kc_tok
                    + offs_d[None, :] * stride_kc_dim
                )

                tl.store(dst_k_ptr, k_val, mask=mask_sub[:, None])

                src_v_ptr = (
                    v_ptr
                    + (curr_kv_pos + offs_sub[:, None]) * stride_v_tok
                    + h * stride_v_head
                    + offs_d[None, :] * stride_v_dim
                )

                v_val = tl.load(src_v_ptr, mask=mask_sub[:, None], other=0.0)

                dst_v_ptr = (
                    value_cache_ptr
                    + physical_block_id * stride_vc_blk
                    + h * stride_vc_head
                    + (block_inner_off + offs_sub[:, None]) * stride_vc_tok
                    + offs_d[None, :] * stride_vc_dim
                )

                tl.store(dst_v_ptr, v_val, mask=mask_sub[:, None])

            processed += sub_len
            curr_log_pos += sub_len
            curr_kv_pos += sub_len