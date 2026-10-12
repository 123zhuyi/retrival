
import torch
import torch_npu
import numpy as np


def convert_absorb_to_naive(kv_cache, pe_cache, wkv_b1, wkv_b2):
    _, _, kv_lora_rank = kv_cache.shape
    _, _, qk_rope_head_dim = pe_cache.shape 
    n_heads, qk_nope_head_dim, kv_lora_rank = wkv_b1.shape
    n_heads, v_head_dim, kv_lora_rank = wkv_b2.shape

    wkv_b1 = wkv_b1.reshape(n_heads*qk_nope_head_dim, kv_lora_rank)
    wkv_b2 = wkv_b2.reshape(n_heads*v_head_dim, kv_lora_rank)

    k_nope = torch.matmul(kv_cache, wkv_b1.T)
    k_nope = k_nope.view(-1, n_heads, qk_nope_head_dim)

    pe_cache = pe_cache.expand(-1, n_heads, -1)
    dense_k_cache = torch.cat([k_nope, pe_cache], dim=-1)

    dense_v_cache = torch.matmul(kv_cache, wkv_b2.T)
    dense_v_cache = dense_v_cache.view(-1, n_heads, v_head_dim)

    return dense_k_cache, dense_v_cache

def kv_to_paged(kv_cache, pe_cache, seqlens, block_size):
    assert len(kv_cache.shape) == 3
    assert len(pe_cache.shape) == 3
    assert kv_cache.shape[0] == sum(seqlens)
    assert pe_cache.shape[0] == sum(seqlens)
    assert kv_cache.shape[1] == 1
    assert pe_cache.shape[1] == 1

    _, _, kv_lora_rank = kv_cache.shape
    _, _, qk_rope_head_dim = pe_cache.shape

    bsz = len(seqlens)
    max_k_seqlen = max(seqlens)
    max_num_blocks_per_seq = (max_k_seqlen + block_size - 1) // block_size

    num_blocks = max_num_blocks_per_seq*bsz
    kv_cache_paged = torch.zeros(size=[num_blocks, block_size, 1, kv_lora_rank], dtype=kv_cache.dtype, device=kv_cache.device)
    pe_cache_paged = torch.zeros(size=[num_blocks, block_size, 1, qk_rope_head_dim], dtype=pe_cache.dtype, device=pe_cache.device)

    seq_offset = 0
    for b in range(bsz):
        seqlen = seqlens[b]

        num_blocks_batch = int(np.ceil(seqlen / block_size))
        for i in range(num_blocks_batch):
            start_ind = i*block_size
            end_ind = min((i+1)*block_size, seqlen)
            act_block_size = end_ind-start_ind

            kv_cache_paged[b*max_num_blocks_per_seq + i, 0:act_block_size, :, :] = kv_cache[seq_offset+start_ind:seq_offset+end_ind, :, :]
            pe_cache_paged[b*max_num_blocks_per_seq + i, 0:act_block_size, :, :] = pe_cache[seq_offset+start_ind:seq_offset+end_ind, :, :]

        seq_offset += seqlen

    return kv_cache_paged.contiguous(), pe_cache_paged.contiguous()

def convert_to_dense(seqlens, kv_cache_stage1, pe_cache_stage1, kv_cache_stage2, pe_cache_stage2):
    bsz = len(seqlens[1:])
    _, _, kv_lora_rank = kv_cache_stage1.shape
    _, _, qk_rope_head_dim = pe_cache_stage1.shape

    shared_len = seqlens[0]
    max_seqlen = max(seqlens[1:])

    dense_kv_cache = torch.zeros(size=[bsz, (shared_len+max_seqlen), 1, kv_lora_rank], dtype=kv_cache_stage1.dtype, device=kv_cache_stage1.device)
    dense_pe_cache = torch.zeros(size=[bsz, (shared_len+max_seqlen), 1, qk_rope_head_dim], dtype=kv_cache_stage1.dtype, device=kv_cache_stage1.device)

    kv_offset = 0
    for b in range(bsz):
        if shared_len > 0:
            dense_kv_cache[b, :shared_len, :, :] = kv_cache_stage1
            dense_pe_cache[b, :shared_len, :, :] = pe_cache_stage1
        if sum(seqlens[1:]) > 0:
            dense_kv_cache[b, shared_len:shared_len+seqlens[b+1], :, :] = kv_cache_stage2[kv_offset:kv_offset+seqlens[b+1], :, :]
            dense_pe_cache[b, shared_len:shared_len+seqlens[b+1], :, :] = pe_cache_stage2[kv_offset:kv_offset+seqlens[b+1], :, :]
        kv_offset += seqlens[b+1]

    return dense_kv_cache.squeeze(2), dense_pe_cache.squeeze(2)
