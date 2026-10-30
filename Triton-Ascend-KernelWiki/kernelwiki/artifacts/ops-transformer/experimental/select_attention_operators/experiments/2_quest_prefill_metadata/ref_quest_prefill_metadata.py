# -*- coding: utf-8 -*-
# Copyright (c) 2025 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.

from typing import Tuple
import torch
from gen_data_quest_prefill_metadata import ceil_div


def ref_quest_prefill_metadata(
        k_cache: torch.Tensor,          # (num_blocks_total, block_size, num_kv_heads, head_dim)
        block_tables: torch.Tensor,      # (batch_size, max_blocks_per_req)
        seq_lens: torch.Tensor,          # (batch_size,)
        metadata_block_tables: torch.Tensor,  # (batch_size,)
        maxblocks: torch.Tensor,
        minblocks: torch.Tensor
):
    """
    Reference implementation of the QUEST prefill-metadata kernel.
    For every request r (0..batch_size-1) and every KV-head n (0..num_kv_heads-1):
        1.  Iterate over the real KV blocks of that request
            (seq_lens[r] tokens → ⌈seq_lens[r]/BLOCK_SIZE⌉ blocks).
        2.  Per block, load **only head n** of the 4-head_dim K-cache slice
            k_cache[kv_block_id, :, n, :]  → shape (block_size, head_dim)
        3.  Reduce-max and reduce-min **along the token axis** (dim 0)
            maxblock[meta_block_id, :, n, :] ← max(tokens)   (block_size → 1 per channel)
            minblock[meta_blk_id, :, n, :] ← min(tokens)   (block_size → 1 per channel)
        4.  Write the two metadata blocks to
            maxblocks[metadata_block_tables[r], :, n, :]
            minblocks[metadata_block_tables[r], :, n, :]

    The function returns the **full** maxblocks/minblocks tensors so you can
    compare them directly against the Ascend-C outputs.
    """
    device = k_cache.device
    dtype = k_cache.dtype
    
    batch_size = block_tables.shape[0]
    block_size = k_cache.shape[1]
    num_kv_heads = k_cache.shape[2]
    head_dim = k_cache.shape[3]

    if (block_size != 128):
        raise ValueError("block_size must be 128")
    
    if (head_dim != 128):
        raise ValueError("head_dim must be 128")

    # ---- iterate over a single request at a time ----
    for r in range(batch_size):
        num_kv_blocks_in_request = ceil_div(seq_lens[r], block_size)
        num_meta_blocks_in_request = ceil_div(num_kv_blocks_in_request, block_size)

        # iterate over block_size kv blocks tokens of this request to produce one metadata block
        for meta_blk in range(num_meta_blocks_in_request):
            num_kv_blocks_completed = meta_blk * block_size
            num_kv_blocks_todo_curr_iter = min(num_kv_blocks_in_request - num_kv_blocks_completed, block_size);            
            meta_blk_id = metadata_block_tables[r, meta_blk].item()
    
            # iterate over one kv block (block_size tokens)
            for blk in range(num_kv_blocks_todo_curr_iter):
                is_last = (blk == num_kv_blocks_todo_curr_iter - 1) and (meta_blk == num_meta_blocks_in_request - 1)
                ntokens_to_reduce = _calculate_tokens_to_reduce(is_last, seq_lens[r], meta_blk, blk, block_size)
                
                kv_block_id = block_tables[r, meta_blk * block_size + blk].item()   # global block id
                kv_block = k_cache[kv_block_id, :ntokens_to_reduce, :, :]  # (block_size, num_kv_heads, head_dim)
                maxblocks[meta_blk_id, blk, :, :] = kv_block.max(dim=0)[0]
                minblocks[meta_blk_id, blk, :, :] = kv_block.min(dim=0)[0]

            # tail filling with zeros
            num_unused_metadata_tokens = block_size - num_kv_blocks_todo_curr_iter
            if (num_unused_metadata_tokens > 0):
                maxblocks[meta_blk_id, num_kv_blocks_todo_curr_iter:, :, :] = 0
                minblocks[meta_blk_id, num_kv_blocks_todo_curr_iter:, :, :] = 0              


def _calculate_tokens_to_reduce(is_last_block: bool, seq_len: torch.Tensor, meta_blk: int, blk: int, 
                                block_size: int) -> int:
    """Calculate number of tokens to reduce for a block."""
    if is_last_block:
        ntokens_reduced_so_far = meta_blk * block_size * block_size + blk * block_size
        return seq_len - ntokens_reduced_so_far
    else:
        return block_size
