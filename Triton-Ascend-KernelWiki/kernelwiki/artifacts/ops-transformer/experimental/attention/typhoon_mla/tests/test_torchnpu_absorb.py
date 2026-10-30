
import torch 
import torch_npu
import numpy as np 
import pytest 

from src.utils import softmax

torch.set_printoptions(sci_mode=False)

class TorchNPUPagedMLA:
    def __init__(self, bsz, seqlens, n_heads, qk_nope_head_dim, qk_rope_head_dim, kv_lora_rank, v_head_dim, softmax_scale, device, dtype):
        self.bsz = bsz
        self.seqlens = seqlens
        self.n_heads = n_heads
        self.qk_nope_head_dim = qk_nope_head_dim
        self.qk_rope_head_dim = qk_rope_head_dim
        self.kv_lora_rank = kv_lora_rank
        self.v_head_dim = v_head_dim
        self.softmax_scale = softmax_scale
        self.device = device 
        self.dtype = dtype
        self.with_projections = True

        max_seq_len = max(seqlens)
        self.block_size = 128
        assert max_seq_len % self.block_size == 0

        # number of KV cache blocks per batch
        self.num_blocks  = max_seq_len // self.block_size

        # Block table
        self.block_table = torch.arange(self.bsz * self.num_blocks,
                                dtype=torch.int32, device=device)
        self.block_table = self.block_table.view(bsz, self.num_blocks)

        self.context_lens = torch.tensor(seqlens, dtype=torch.int32, device="cpu")

    def run(self, q_nope, q_rope, block_cache, wkv_b1, wkv_b2):
        q_nope = torch.einsum("bhd,hdc->bhc", q_nope, wkv_b1)
        query = torch.cat([q_nope, q_rope], dim=-1).contiguous()

        out = torch.zeros_like(q_nope)
        torch_npu._npu_paged_attention_mla(
            query=query,
            key_cache=block_cache,
            num_kv_heads=1,
            num_heads=self.n_heads,
            scale_value=self.softmax_scale,
            block_table=self.block_table,
            context_lens=self.context_lens,
            mla_vheadsize=self.kv_lora_rank,
            out=out
        )
        out = torch.einsum("bhc,hdc->bhd", out, wkv_b2)
        return out

    def generate_input(self):
        q_nope = torch.randn(size=[self.bsz, self.n_heads, self.qk_nope_head_dim], device=self.device, dtype=self.dtype)
        q_rope = torch.randn(size=[self.bsz, self.n_heads, self.qk_rope_head_dim], device=self.device, dtype=self.dtype)

        wkv_b1 = torch.randn(size=[self.n_heads, self.qk_nope_head_dim, self.kv_lora_rank], dtype=self.dtype, device=self.device)
        wkv_b2 = torch.randn(size=[self.n_heads, self.v_head_dim, self.kv_lora_rank], dtype=self.dtype, device=self.device)

        block_cache = torch.rand(size=[self.bsz*self.num_blocks, self.block_size, 1, self.kv_lora_rank + self.qk_rope_head_dim], dtype=self.dtype, device=self.device)

        return q_nope, q_rope, block_cache, wkv_b1, wkv_b2

    def perf(self, warm_up, n_repeat):
        q_nope, q_rope, block_cache, wkv_b1, wkv_b2 = self.generate_input()
        
        for i in range(warm_up):
            self.run(q_nope, q_rope, block_cache, wkv_b1, wkv_b2)
            torch.npu.synchronize()

        start = [torch.npu.Event(enable_timing=True) for i in range(n_repeat)]
        end = [torch.npu.Event(enable_timing=True) for i in range(n_repeat)]

        cache = torch.empty(int(256e6 // 4), dtype=torch.int, device=self.device)

        elapsed  = []
        for i in range(n_repeat):
            cache.zero_()

            start[i].record()

            self.run(q_nope, q_rope, block_cache, wkv_b1, wkv_b2)

            end[i].record()
            torch.npu.synchronize()

        torch.npu.synchronize()
        elapsed = [start[i].elapsed_time(end[i]) for i in range(n_repeat)]
        m_elapsed = np.median(elapsed)

        return m_elapsed

def baseline(q_nope, q_rope, kv_cache, pe_cache, wkv_b1, wkv_b2, kv_seqlens, scale):
    bsz, n_heads, qk_nope_head_dim = q_nope.shape
    bsz, n_heads, qk_rope_head_dim = q_rope.shape
    n_heads, qk_nope_head_dim, kv_lora_rank = wkv_b1.shape

    q_nope = torch.einsum("bhd,hdc->bhc", q_nope, wkv_b1)
    
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

    out = torch.einsum("bsht,btc->bshc", scores, kv_cache).squeeze(1)
    out = torch.einsum("bhc,hdc->bhd", out, wkv_b2)

    return out, lse.squeeze(1).squeeze(-1)

@pytest.mark.parametrize("bsz, kv_seqlen", [
    (4, 128),
    (16, 256),
    (1, 8192),
])
def test_torchnpu_absorb(bsz, kv_seqlen):
    n_heads = 128
    n_kv_heads = 128
    qk_nope_head_dim = 128
    qk_rope_head_dim = 64
    kv_lora_rank = 512
    v_head_dim = 128
    softmax_scale = 1/np.sqrt(128)

    device = torch.device("npu:0")
    dtype = torch.bfloat16

    block_size = 128
    num_blocks  = kv_seqlen // block_size

    kv_seqlens = [kv_seqlen] * bsz

    torch_npu_paged_mla = TorchNPUPagedMLA(bsz, kv_seqlens, n_heads, qk_nope_head_dim, qk_rope_head_dim, kv_lora_rank, v_head_dim, softmax_scale, device, dtype)

    q_nope = torch.randn(size=[bsz, n_heads, qk_nope_head_dim], device=device, dtype=dtype)
    q_rope = torch.randn(size=[bsz, n_heads, qk_rope_head_dim], device=device, dtype=dtype)

    wkv_b = 1e-2*torch.randn(size=[n_heads * (qk_nope_head_dim + v_head_dim), kv_lora_rank], dtype=dtype, device=device)
    wkv_b1, wkv_b2 = wkv_b.view(n_heads, (qk_nope_head_dim + v_head_dim), kv_lora_rank).split([qk_nope_head_dim, v_head_dim], dim=1)

    kv_cache = torch.rand(size=[bsz, kv_seqlen, 1, kv_lora_rank], dtype=dtype, device=device)
    pe_cache = torch.rand(size=[bsz, kv_seqlen, 1, qk_rope_head_dim], dtype=dtype, device=device)

    block_cache = torch.cat([kv_cache, pe_cache], dim=-1).view([bsz*num_blocks, block_size, 1, kv_lora_rank + qk_rope_head_dim])

    out = torch_npu_paged_mla.run(q_nope, q_rope, block_cache, wkv_b1, wkv_b2)
    out_gt, lse_gt = baseline(q_nope, q_rope, kv_cache, pe_cache, wkv_b1, wkv_b2, kv_seqlens, softmax_scale)

    print(out[-1, -1, :])
    print(out_gt[-1, -1, :])

    assert torch.allclose(out, out_gt, atol=1e-1, rtol=0), f"out-out_gt max. err: {torch.abs(out-out_gt).max():.3e}"

if __name__=="__main__":
    test_torchnpu_absorb(4, 256)