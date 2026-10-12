# -----------------------------------------------------------------------------------------------------------
# Copyright (c) 2025 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------

import os
import csv
import argparse

import torch 
import torch_npu
import numpy as np

from src.catlass_mla import catlass_kernel_prepare, catlass_score_mla

torch.set_printoptions(sci_mode=False)

dtype_map = {
    torch.float16: "float16",
    torch.bfloat16: "bf16",
    torch.int8: "int8"
}


def typhoon_mla_prepare(bsz, seqlens, n_heads, kv_lora_rank, qk_rope_head_dim, block_size, device, dtype):
    max_seq_len = max(seqlens)
    num_blocks = bsz * max_seq_len // block_size

    block_tables = torch.arange(num_blocks, dtype=torch.int32).to(device)
    
    device_mem, lse_idxs = catlass_kernel_prepare(
        batch=bsz,
        num_heads=n_heads,
        head_size=kv_lora_rank,
        head_size_rope=qk_rope_head_dim,
        num_blocks=num_blocks,
        block_size=block_size,
        kv_lens=np.array(seqlens),
        device=device,
        dtype_str=dtype_map[dtype]
    )
    catlass_ctx = {
        "num_blocks": num_blocks,
        "block_tables": block_tables,
        "device_mem": device_mem,
        "lse_idxs": lse_idxs
    }
    return catlass_ctx


def typhoon_mla_run(q, q_nope, q_rope, naive_k_cache, naive_v_cache, absorb_kv_cache, absorb_pe_cache, wkv_b1, wkv_b2,
                    catlass_ctx, kv_seqlens, softmax_scale, run_in_single_stage=False):
    if run_in_single_stage:
        return _run_full_absorb(q_nope, q_rope, wkv_b1, wkv_b2) 
    else:
        return _run_in_2stage(
            q, q_nope, q_rope, 
            naive_k_cache, 
            naive_v_cache,
            absorb_kv_cache,
            absorb_pe_cache,
            wkv_b1, wkv_b2, catlass_ctx, kv_seqlens, softmax_scale
        )


def _run_in_2stage(q, q_nope, q_rope, naive_k_cache, naive_v_cache, absorb_kv_cache, absorb_pe_cache, wkv_b1, wkv_b2, 
                   catlass_ctx, kv_seqlens, softmax_scale):
    bsz = q_nope.shape[0]
    shared_kvlen = kv_seqlens[0]
    nonshared_kvlens = kv_seqlens[1:]

    if shared_kvlen > 0:
        actseqlen = [bsz]
        actseqlenkv = [shared_kvlen]
        out1, softmax_lse1 = fused_infer_attention_score_shared(
            q, 
            naive_k_cache, 
            naive_v_cache, 
            actseqlen, 
            actseqlenkv, 
            softmax_scale)
        softmax_lse1 = softmax_lse1.squeeze(-1)

    if sum(nonshared_kvlens) > 0:
        q_nope = torch.einsum("bhd,hdc->bhc", q_nope, wkv_b1)
        q_nope = q_nope.contiguous()

        torch.npu.synchronize() # without this synch, I get NaNs
        out2, softmax_lse2 = catlass_score_mla(
            q_nope, q_rope,
            absorb_kv_cache, 
            absorb_pe_cache, 
            np.array(nonshared_kvlens),
            catlass_ctx["block_tables"], 
            catlass_ctx["device_mem"], 
            softmax_scale=softmax_scale,
            return_lse=True,
            lse_idxs=catlass_ctx["lse_idxs"]
        )
        
        out2 = torch.einsum("bhc,hdc->bhd", out2, wkv_b2)

        if shared_kvlen == 0:
            return out2
        else:
            return combine_lse(out1, softmax_lse1, out2, softmax_lse2)
    else:
        return out1


def combine_lse(out1: torch.Tensor, lse1: torch.Tensor, out2: torch.Tensor, lse2: torch.Tensor):
    """
    Combines two attention results using their LSEs.

    Out1/2 shape: [batch, seq_len, qheads, hdim]
    lse1/2 shape: [batch, seq_len, qheads]
    """

    input_type = out1.dtype
    max_lse = torch.maximum(lse1, lse2)

    adjust_factor1 = (lse1 - max_lse).exp()
    adjust_factor2 = (lse2 - max_lse).exp()

    new_denominator = adjust_factor1 + adjust_factor2

    aggregated = (
        out1 * adjust_factor1.unsqueeze(-1) + out2 * adjust_factor2.unsqueeze(-1)
    ) / new_denominator.unsqueeze(-1)

    return aggregated.to(input_type)


def fused_infer_attention_score_shared(q, k, v, actseqlen, actseqlenkv, scale):
    bsz, n_heads, q_headdim = q.shape
    seqlen, n_kv_heads, k_headdim = k.shape
    seqlen, n_kv_heads, v_headdim = v.shape

    out, softmax_lse = torch_npu.npu_fused_infer_attention_score(q, k, v, 
        actual_seq_lengths=actseqlen, actual_seq_lengths_kv=actseqlenkv,
        num_heads=n_heads, num_key_value_heads=n_kv_heads, input_layout="TND", 
        scale=scale, softmax_lse_flag=True
    )

    return out, softmax_lse