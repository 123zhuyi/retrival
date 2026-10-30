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
import math
import json

import numpy as np
import torch
import torch_npu

from torch_catlass_attention import catlass_mla_prepare, mla


def catlass_device_malloc(device, aic_core_num, o_core_temp_size, l_size):
    aic_core_num = aic_core_num.astype(np.int64)
    l_size = l_size.astype(np.int64)
    o_core_temp_size = o_core_temp_size.astype(np.int64)

    block_size_db = 65536 # -> Hard coded in kernel
    return {
        "s": torch.empty(aic_core_num * block_size_db * 2,
                        dtype=torch.float32, device=device).contiguous(),
        "p": torch.empty(aic_core_num * block_size_db * 2,
                        dtype=torch.float16, device=device).contiguous(),
        "result_temp": torch.empty(aic_core_num * block_size_db * 2,
                                dtype=torch.float32, device=device).contiguous(),
        "global_o": torch.empty(aic_core_num * block_size_db,
                                dtype=torch.float32, device=device).contiguous(),
        "l": torch.empty(l_size,
                        dtype=torch.float32, device=device).contiguous(),
        "o_core_temp": torch.empty(o_core_temp_size,
                                dtype=torch.float32, device=device).contiguous()
    }


def catlass_compute_lse_indices(length, kv_split_num, batch_size, num_heads, device) -> torch.Tensor:
    n = int(kv_split_num)
    jump_n = n - 1
    target_count = batch_size * num_heads

    indices = []
    i = 0
    cycle = 0

    while i < length:
        short_block = (cycle % 11 == 10)
        block_size = 8 if short_block else 12
        end = min(i + block_size, length)

        for k in range(i, end):
            indices.append(k)
            if len(indices) >= target_count:
                return torch.tensor(indices, dtype=torch.long, device=device)

        if end == length:
            break

        jump_size = block_size * jump_n
        i = end + jump_size
        cycle += 1

    return torch.tensor(indices, dtype=torch.long, device=device)


def catlass_select_lse(l_host, idx_t, bsz, n_heads) -> torch.Tensor:
    return l_host.index_select(0, idx_t).view(bsz, n_heads)


def catlass_kernel_prepare(
    batch: int,
    num_heads: int,
    head_size: int,
    head_size_rope: int,
    num_blocks: int,
    block_size: int,
    kv_lens: np.ndarray,
    device: torch.device | str = "npu:0",
    dtype_str: str = "float16",
):
    device = torch.device(device)
    if dtype_str == "float16":
        torch_dtype = torch.float16
    elif dtype_str == "bf16":
        torch_dtype = torch.bfloat16
    else:
        raise ValueError("dtype_str must be 'float16' or 'bf16'")
    
    max_seq_len = int(kv_lens.max())
    min_num_blocks = batch * max_seq_len // block_size

    if num_blocks < min_num_blocks:
        raise ValueError(
            f"num_blocks ({num_blocks}) is too small for batch size ({batch}) and max_seq_len ({max_seq_len})."
            f"It should be at least batch * max_seq_len / block_size.")
    
    kv_heads = 1 # -> Hard coded in kernel

    q_nope_pt = torch.empty((batch, num_heads, head_size), dtype=torch_dtype, device=device).contiguous()
    q_rope_pt = torch.empty((batch, num_heads, head_size_rope), dtype=torch_dtype, device=device).contiguous()

    k_nope_pt = torch.empty((num_blocks, block_size, kv_heads, head_size), 
                            dtype=torch_dtype, device=device).contiguous()
    k_rope_pt = torch.empty((num_blocks, block_size, kv_heads, head_size_rope), 
                            dtype=torch_dtype, device=device).contiguous()

    block_tables = np.arange(num_blocks, dtype=np.int32)
    block_tables_pt = torch.from_numpy(block_tables.reshape(-1).astype(np.int32)).to(device=device)

    place_holder = torch.empty(1, dtype=torch.float32, device=device).contiguous()

    torch.npu.synchronize()
    kernel_prep = np.array(
        catlass_mla_prepare(
            q_nope_pt,
            q_rope_pt,
            k_nope_pt,
            k_rope_pt,
            int(kv_lens.max()),
            kv_lens,                
            block_tables_pt,        
            place_holder, place_holder, place_holder,
            place_holder, place_holder, place_holder,
            dtype_str
        ),
        dtype=np.uint64
    )
    # The shape of kernel_prep is (aic_core_num, kv_split_core_num, o_core_temp_size, l_size)

    torch.npu.synchronize()

    kv_split_core_num = int(kernel_prep[1])

    device_mem = catlass_device_malloc(device, kernel_prep[0], kernel_prep[2], kernel_prep[3])

    lse_idxs = catlass_compute_lse_indices(
        device_mem["l"].shape[0],
        kv_split_core_num,
        batch,
        num_heads,
        device
    )

    return device_mem, lse_idxs


def catlass_mla_run(q_nope_pt, q_rope_pt, k_nope_pt, k_rope_pt, kv_lens, 
                    block_tables_pt, device_mem, dtype_str, softmax_scale=0.088388):
    return mla(
        q_nope_pt, q_rope_pt, k_nope_pt, k_rope_pt, 
        int(kv_lens.max()), kv_lens, block_tables_pt, 
        device_mem["s"], device_mem["p"], device_mem["result_temp"], 
        device_mem["global_o"], device_mem["l"], device_mem["o_core_temp"], 
        dtype_str, softmax_scale
        )


dtype_map = {
    torch.float16: "float16",
    torch.bfloat16: "bf16",
    torch.int8: "int8"
}


def catlass_score_mla(q, q_rope, k, k_rope, kv_seq_lens, 
                      block_tables, device_mem, softmax_scale, return_lse=True, lse_idxs=None):
    ret = catlass_mla_run(q, q_rope, k, k_rope, kv_seq_lens, 
        block_tables, device_mem, dtype_map[q.dtype], softmax_scale=softmax_scale)

    if return_lse:
        bsz = q.shape[0]
        n_heads = q.shape[1]
        lse_val = catlass_select_lse(device_mem["l"], lse_idxs, bsz, n_heads)
        return ret, lse_val
    else:
        return ret
