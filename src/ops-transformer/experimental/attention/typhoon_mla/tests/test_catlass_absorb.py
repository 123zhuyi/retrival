import torch 
import torch_npu
import numpy as np
import pytest 

from src.catlass_mla import catlass_score_mla, catlass_kernel_prepare
from src.utils import softmax

torch.set_printoptions(sci_mode=False)

dtype_map = {
    torch.float16: "float16",
    torch.bfloat16: "bf16",
    torch.int8: "int8"
}

def baseline(q_nope, q_rope, kv_cache, pe_cache, kv_seqlens, scale):
    bsz, n_heads, kv_lora_rank = q_nope.shape
    bsz, n_heads, qk_rope_head_dim = q_rope.shape

    max_seqlen = max(kv_seqlens)
    kv_cache = kv_cache.reshape(bsz, max_seqlen, kv_lora_rank)
    pe_cache = pe_cache.reshape(bsz, max_seqlen, qk_rope_head_dim)

    num_blocks, block_size, kv_lora_rank = kv_cache.shape
    num_blocks, block_size, qk_rope_head_dim = pe_cache.shape
    kv_cache = kv_cache.reshape(num_blocks*block_size, kv_lora_rank) # now shape is bszxseqlen, kv_lora_rank
    pe_cache = pe_cache.reshape(num_blocks*block_size, qk_rope_head_dim) # now shape is bszxseqlen, kv_lora_rank

    kv_cache = kv_cache.reshape(bsz, -1, kv_lora_rank)
    pe_cache = pe_cache.reshape(bsz, -1, qk_rope_head_dim)

    scores = torch.einsum("bshc,btc->bsht", q_nope.unsqueeze(1), kv_cache) 
    scores += torch.einsum("bshr,btr->bsht", q_rope.unsqueeze(1), pe_cache)

    mask = []
    for b in range(bsz):
        seqlen = kv_seqlens[b]
        mask.append([0] * seqlen + [-np.inf] * (max_seqlen - seqlen))
    mask = torch.tensor(mask, device=q_nope.device).unsqueeze(1).unsqueeze(2).expand_as(scores)

    scores += mask

    scores = scores * scale

    scores, lse = softmax(scores)

    x = torch.einsum("bsht,btc->bshc", scores, kv_cache).squeeze(1)

    return x, lse.squeeze(1).squeeze(-1)

@pytest.mark.parametrize("bsz, kv_seqlen", [
    (4, 128),
    (16, 256),
    (1, 8192),
])
def test_catlass_absorb(bsz, kv_seqlen):
    n_heads = 128
    kv_lora_rank = 512
    qk_rope_head_dim = 64

    kv_seqlens = [kv_seqlen] * bsz 

    block_size = 128
    
    max_num_blocks_per_seq = int(np.ceil(kv_seqlen/block_size))
    num_blocks = bsz * max_num_blocks_per_seq

    scale = 1/np.sqrt(128.0)

    dtype = torch.bfloat16
    device = torch.device("npu:0")

    q_nope = torch.randn(size=[bsz, n_heads, kv_lora_rank], dtype=dtype, device=device)
    q_rope = torch.randn(size=[bsz, n_heads, qk_rope_head_dim], dtype=dtype, device=device)
    k_nope = torch.randn(size=[num_blocks, block_size, 1, kv_lora_rank], dtype=dtype, device=device)
    k_rope = torch.randn(size=[num_blocks, block_size, 1, qk_rope_head_dim], dtype=dtype, device=device)

    block_tables = torch.arange(
        bsz * max_num_blocks_per_seq, device=device, dtype=torch.int32
    )
    
    device_mem, lse_idxs = catlass_kernel_prepare(bsz, n_heads, kv_lora_rank, qk_rope_head_dim,
                                                    num_blocks, block_size, np.array(kv_seqlens), device, dtype_map[dtype])

    out, lse = catlass_score_mla(q_nope, q_rope, k_nope, k_rope, np.array(kv_seqlens), block_tables, device_mem, scale, return_lse=True, lse_idxs=lse_idxs)
    out_gt, lse_gt = baseline(q_nope, q_rope, k_nope, k_rope, kv_seqlens, scale)
    
    print(out[-1, -1, :])
    print(out_gt[-1, -1, :])

    print(lse[-1, :])
    print(lse_gt[-1, :])

    assert torch.allclose(out, out_gt, atol=1e-1, rtol=0), f"out-out_gt max. err: {torch.abs(out-out_gt).max():.3e}"
    assert torch.allclose(lse, lse_gt, atol=1e-1, rtol=0), f"lse-lse_gt max. err: {torch.abs(lse-lse_gt).max():.3e}"

if __name__=="__main__":
    test_catlass_absorb(1, 8192)