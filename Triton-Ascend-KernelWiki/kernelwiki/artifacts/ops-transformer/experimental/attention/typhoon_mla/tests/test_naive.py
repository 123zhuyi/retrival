
import pytest
import torch 
import torch_npu
import numpy as np

from src.typhoon_mla import fused_infer_attention_score_shared
from src.utils import softmax

torch.set_printoptions(sci_mode=False)



def baseline(q, k, v, scale):
    bsz, n_heads, q_headdim = q.shape
    seqlen, n_kv_heads, k_headdim = k.shape
    seqlen, n_kv_heads, v_headdim = v.shape

    scores = torch.einsum("shc,thc->sht", q, k)
    scores = scores * scale

    scores, lse = softmax(scores, use_pytorch=False)

    out = torch.einsum("sht,thc->shc", scores, v)
    return out, lse

@pytest.mark.parametrize("bsz, seqlen", [
    (4, 128),
    (16, 256),
    (1, 8192),
])
def test_naive(bsz, seqlen):
    n_heads = 128
    n_kv_heads = 128
    q_headdim = 192
    k_headdim = 192
    v_headdim = 128
    
    softmax_scale = 1/np.sqrt(128)

    dtype = torch.bfloat16
    device = torch.device("npu:0")

    q = torch.rand([bsz, n_heads, q_headdim], dtype=dtype, device=device)
    k = torch.rand([seqlen, n_kv_heads, k_headdim], dtype=dtype, device=device)
    v = torch.rand([seqlen, n_kv_heads, v_headdim], dtype=dtype, device=device)

    actseqlen = [bsz]
    actseqlenkv = [seqlen]

    out, lse = fused_infer_attention_score_shared(q, k, v, actseqlen, actseqlenkv, softmax_scale)
    out_gt, lse_gt = baseline(q, k, v, softmax_scale)

    print(out[-1, -1, :])
    print(out_gt[-1, -1, :])

    print(lse[-1, :, 0])
    print(lse_gt[-1, :, 0])

    assert torch.allclose(out, out_gt, atol=1e-2, rtol=0), f"out-out_gt max. err: {torch.abs(out-out_gt).max():.3e}"
    assert torch.allclose(lse, lse_gt, atol=1e-2, rtol=0), f"lse-lse_gt max. err: {torch.abs(lse-lse_gt).max():.3e}"

if __name__=="__main__":
    test_naive(4, 256)