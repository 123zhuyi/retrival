import torch 
import torch_npu
import numpy as np
import pytest 

from src.typhoon_mla import typhoon_mla_prepare, typhoon_mla_run

from tests.test_torchnpu_absorb import TorchNPUPagedMLA
from tests.utils import convert_absorb_to_naive, kv_to_paged, convert_to_dense

torch.set_printoptions(sci_mode=False)

class TyphoonMLA:
    def __init__(self, bsz, seqlens, n_heads, qk_nope_head_dim, qk_rope_head_dim, kv_lora_rank, v_head_dim, softmax_scale, device, dtype):
        self.bsz = bsz
        self.seqlens = seqlens
        self.n_heads = n_heads
        self.qk_nope_head_dim = qk_nope_head_dim
        self.qk_rope_head_dim = qk_rope_head_dim
        self.kv_lora_rank = kv_lora_rank
        self.v_head_dim = v_head_dim
        self.dtype = dtype
        self.device = device
        self.block_size = 128

        self.device = device
        self.dtype = dtype

        self.softmax_scale = softmax_scale

        self.seqlen_stage1 = seqlens[0]
        self.seqlen_stage2 = seqlens[1]

        self.catlass_ctx = typhoon_mla_prepare(self.bsz, self.seqlens[1:], self.n_heads, self.kv_lora_rank, self.qk_rope_head_dim, self.block_size, self.device, self.dtype)

    def run(self, q, q_nope, q_rope, naive_k_cache, naive_v_cache, absorb_kv_cache, absorb_pe_cache, wkv_b1, wkv_b2):        
        return typhoon_mla_run(q, q_nope, q_rope, 
            naive_k_cache, 
            naive_v_cache, 
            absorb_kv_cache, 
            absorb_pe_cache, 
            wkv_b1, wkv_b2, 
            self.catlass_ctx, 
            self.seqlens, 
            self.softmax_scale, 
            run_in_single_stage=False
        )
    
    def generate_input(self):
        q = torch.randn(size=[self.bsz, self.n_heads, self.qk_nope_head_dim+self.qk_rope_head_dim], device=self.device, dtype=self.dtype)
        q_nope = torch.randn(size=[self.bsz, self.n_heads, self.qk_nope_head_dim], device=self.device, dtype=self.dtype)
        q_rope = torch.randn(size=[self.bsz, self.n_heads, self.qk_rope_head_dim], device=self.device, dtype=self.dtype)
        
        wkv_b1 = torch.randn(size=[self.n_heads, self.qk_nope_head_dim, self.kv_lora_rank], dtype=self.dtype, device=self.device)
        wkv_b2 = torch.randn(size=[self.n_heads, self.v_head_dim, self.kv_lora_rank], dtype=self.dtype, device=self.device)

        naive_k_cache = torch.randn(size=(self.seqlens[0], self.n_heads, self.qk_nope_head_dim+self.qk_rope_head_dim), dtype=self.dtype, device=self.device)
        naive_v_cache = torch.randn(size=(self.seqlens[0], self.n_heads, self.qk_nope_head_dim), dtype=self.dtype, device=self.device)

        absorb_kv_cache = torch.randn((sum(self.seqlens[1:]) // self.block_size, self.block_size, 1, self.kv_lora_rank), dtype=self.dtype, device=self.device)
        absorb_pe_cache = torch.randn((sum(self.seqlens[1:]) // self.block_size, self.block_size, 1, self.qk_rope_head_dim), dtype=self.dtype, device=self.device)

        return q, q_nope, q_rope, naive_k_cache, naive_v_cache, absorb_kv_cache, absorb_pe_cache, wkv_b1, wkv_b2


    def perf(self, warm_up=25, n_repeat=100):
        q, q_nope, q_rope, naive_k_cache, naive_v_cache, absorb_kv_cache, absorb_pe_cache, wkv_b1, wkv_b2 = self.generate_input()
        
        for i in range(warm_up):
            self.run(q, q_nope, q_rope, naive_k_cache, naive_v_cache, absorb_kv_cache, absorb_pe_cache, wkv_b1, wkv_b2)
            torch.npu.synchronize()

        start = [torch.npu.Event(enable_timing=True) for i in range(n_repeat)]
        end = [torch.npu.Event(enable_timing=True) for i in range(n_repeat)]

        cache = torch.empty(int(256e6 // 4), dtype=torch.int, device=self.device)

        elapsed  = []
        for i in range(n_repeat):
            cache.zero_()

            start[i].record()

            self.run(q, q_nope, q_rope, naive_k_cache, naive_v_cache, absorb_kv_cache, absorb_pe_cache, wkv_b1, wkv_b2)

            end[i].record()
            torch.npu.synchronize()

        torch.npu.synchronize()
        elapsed = [start[i].elapsed_time(end[i]) for i in range(n_repeat)]
        m_elapsed = np.median(elapsed)

        return m_elapsed

@pytest.mark.parametrize("bsz, shared_seqlen, nonshared_seqlen", [
    (4, 256, 256),
    (16, 512, 128),
    (1, 8192, 512),
])
def test_typhoon_mla(bsz, shared_seqlen, nonshared_seqlen):
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
    seqlens = [shared_seqlen] + [nonshared_seqlen] * bsz
    
    test_typhoon_mla = TyphoonMLA(bsz, seqlens, n_heads, qk_nope_head_dim, qk_rope_head_dim, kv_lora_rank, v_head_dim, softmax_scale, device, dtype)

    q_nope = torch.randn(size=[bsz, n_heads, qk_nope_head_dim], device=device, dtype=dtype)
    q_rope = torch.randn(size=[bsz, n_heads, qk_rope_head_dim], device=device, dtype=dtype)
    q = torch.cat([q_nope, q_rope], dim=-1)

    wkv_b = 1e-2*torch.randn(size=[n_heads * (qk_nope_head_dim + v_head_dim), kv_lora_rank], dtype=dtype, device=device)
    wkv_b1, wkv_b2 = wkv_b.view(n_heads, (qk_nope_head_dim + v_head_dim), kv_lora_rank).split([qk_nope_head_dim, v_head_dim], dim=1)

    shared_kv_cache = torch.randn(size=(seqlens[0], 1, kv_lora_rank), dtype=dtype, device=device)
    shared_pe_cache = torch.randn(size=(seqlens[0], 1, qk_rope_head_dim), dtype=dtype, device=device)
    naive_k_cache, naive_v_cache = convert_absorb_to_naive(shared_kv_cache, shared_pe_cache, wkv_b1, wkv_b2)

    nonshared_kv_cache = torch.randn((sum(seqlens[1:]), 1, kv_lora_rank), dtype=dtype, device=device)
    nonshared_pe_cache = torch.randn((sum(seqlens[1:]), 1, qk_rope_head_dim), dtype=dtype, device=device)
    absorb_kv_cache, absorb_pe_cache = kv_to_paged(nonshared_kv_cache, nonshared_pe_cache, seqlens[1:], block_size)

    out = test_typhoon_mla.run(q, q_nope, q_rope, naive_k_cache, naive_v_cache, absorb_kv_cache, absorb_pe_cache, wkv_b1, wkv_b2)
    
    print(f"TyphoonMLA out: \n{out[0,0,0:32]}")
    print(out.shape)

    dense_seqlens = [shared_seqlen + nonshared_seqlen] * bsz
    test_torchnpu_paged_mla = TorchNPUPagedMLA(bsz, dense_seqlens, n_heads, qk_nope_head_dim, qk_rope_head_dim, kv_lora_rank, v_head_dim, softmax_scale, device, dtype)

    dense_kv_cache, dense_pe_cache = convert_to_dense(seqlens, shared_kv_cache, shared_pe_cache, nonshared_kv_cache, nonshared_pe_cache)
    num_blocks  = (shared_seqlen + nonshared_seqlen) // block_size
    block_cache = torch.cat([dense_kv_cache, dense_pe_cache], dim=-1).view([bsz*num_blocks, block_size, 1, kv_lora_rank + qk_rope_head_dim])

    gt_out = test_torchnpu_paged_mla.run(q_nope, q_rope, block_cache, wkv_b1, wkv_b2)
    
    print(f"torch_npu out: \n{gt_out[0,0,0:32]}")
    print(gt_out.shape)

    max_abs_diff = torch.max(torch.abs(out-gt_out))
    print(f"max_abs_diff: {max_abs_diff:e}")
    if torch.allclose(out, gt_out, atol=0.125, rtol=0):
        print(f"✅ Accuracy Test OK.")
    else:
        print(f"❌ Accuracy Test failed.")      

if __name__=="__main__":
    test_typhoon_mla(4, 256, 256)