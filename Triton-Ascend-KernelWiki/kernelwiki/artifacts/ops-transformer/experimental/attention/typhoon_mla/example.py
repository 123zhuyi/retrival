# -----------------------------------------------------------------------------------------------------------
# Copyright (c) 2025 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------

import logging 

import torch 
import torch_npu
import numpy as np

from src.typhoon_mla import typhoon_mla_prepare, typhoon_mla_run

logging.basicConfig(level=logging.INFO) # Specifies the log file path, etc.

if __name__ == "__main__":
    n_heads = 128 # number of attention heads
    qk_nope_head_dim = 128 # noPE head dim
    qk_rope_head_dim = 64 # RoPE head dim
    kv_lora_rank = 512 # KV lora rank
    v_head_dim = 128 # V head dim
    softmax_scale = 1 / np.sqrt(128) # Softmax scale

    device = torch.device("npu:0") 
    dtype = torch.bfloat16

    torch.set_default_device(device)
    torch.set_default_dtype(dtype)

    block_size = 128 # Block size for paged KV-cache

    bsz = 4 # Batch size

    shared_seqlen = 4096 # Length of shared sequence (e.g., number of tokens in the system prompt)
    
    nonshared_seqlen = 128 # Length of non-shared sequence (e.g., number of tokens in each prompt + generated tokens)

    # Sequence lengths assuming each request has equal number of non-shared tokens. 
    # Each request can have a variable seqlen, too.
    seqlens = [shared_seqlen] + [nonshared_seqlen] * bsz 

    # query vectors
    q_nope = torch.randn(size=[bsz, n_heads, qk_nope_head_dim])
    q_rope = torch.randn(size=[bsz, n_heads, qk_rope_head_dim])
    q = torch.cat([q_nope, q_rope], dim=-1)

    # KV up-scaling projection matrix
    wkv_b = torch.randn(size=[n_heads * (qk_nope_head_dim + v_head_dim), kv_lora_rank])
    wkv_b1, wkv_b2 = wkv_b.view(
        n_heads, (qk_nope_head_dim + v_head_dim), kv_lora_rank
    ).split([qk_nope_head_dim, v_head_dim], dim=1)

    # KV-cache for the shared sequence (in naive formulation)
    naive_k_cache = torch.randn(
        size=(seqlens[0], n_heads, qk_nope_head_dim + qk_rope_head_dim)
    )
    naive_v_cache = torch.randn(
        size=(seqlens[0], n_heads, qk_nope_head_dim)
    )

    # KV and PE cache for the non-shared sequence (in absorb formulation)
    absorb_kv_cache = torch.randn(
        (sum(seqlens[1:]) // block_size, block_size, 1, kv_lora_rank)
    )
    absorb_pe_cache = torch.randn(
        (sum(seqlens[1:]) // block_size, block_size, 1, qk_rope_head_dim)
    )

    # Allocate memory required for CATLASS kernel
    catlass_ctx = typhoon_mla_prepare(
        bsz=bsz, 
        seqlens=seqlens[1:], 
        n_heads=n_heads, 
        kv_lora_rank=kv_lora_rank, 
        qk_rope_head_dim=qk_rope_head_dim, 
        block_size=block_size, 
        device=device, 
        dtype=dtype
    )

    # Run the TyphoonMLA kernel with random input
    out = typhoon_mla_run(
        q=q, 
        q_nope=q_nope, 
        q_rope=q_rope, 
        naive_k_cache=naive_k_cache, 
        naive_v_cache=naive_v_cache, 
        absorb_kv_cache=absorb_kv_cache, 
        absorb_pe_cache=absorb_pe_cache, 
        wkv_b1=wkv_b1, 
        wkv_b2=wkv_b2, 
        catlass_ctx=catlass_ctx, 
        kv_seqlens=seqlens, 
        softmax_scale=softmax_scale, 
        run_in_single_stage=False
    )

    logging.info("--- Output ---")
    logging.info(out)
    logging.info(out.shape)