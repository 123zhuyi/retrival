# -*- coding: utf-8 -*-
# Copyright (c) 2025 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.


import math
import torch


def ceil_div(a, b):
    """Ceiling division: ceil(a / b)"""
    return -(a // -b)


def ref_quest_block_select_paged_w(query: torch.Tensor,         # (batch_size, num_heads, head_dim)
                                 maxblocks: torch.Tensor,       # (num_meta_blocks, block_size, num_kv_heads, head_dim)
                                 minblocks: torch.Tensor,       # (num_meta_blocks, block_size, num_kv_heads, head_dim)
                                 metadata_block_tables: torch.Tensor,  # (batch_size, mmbpr)
                                 seq_lens: torch.Tensor,           # (batch_size)
                                 tokens_since_metadata_update: int,
                                 k: int,
                                 fast: bool = True) -> torch.Tensor:  # (batch_size, num_kv_heads, k)
    """
    Reference implementation of the paged QUEST block selection operator
    Selects the top-k important KV blocks based on approximate attention scores.

    Input:
        query - fp16 tensor of shape (batch_size, num_heads, head_dim)
        maxblocks - fp16 tensor of shape (num_meta_blocks, block_size, num_kv_heads, head_dim)
        minblocks - fp16 tensor of shape (num_meta_blocks, block_size, num_kv_heads, head_dim)
        metadata_block_tables - int32 tensor of shape (batch_size, mmbpr)
        seq_lens - int32 tensor of shape (batch_size)
        k - integer specifying how many block-indices to return for each kv-head

    Algorithm steps:
    1. For every batch, reduce-mean the query tensor across num_heads dimension 
        such that every group of num_heads/num_kv_heads vectors of shape head_dim are reduced to one vector
        of shape head_dim denotes as "grouped_query[b,n]".
    2. For each batch b and KV-head n:
        2.1. Determine how many metadata blocks are valid for this request: 
             num_valid_blocks = floor_div(seq_lens[b], block_size * block_size)
        2.2. For each valid metadata block:
             - Get the block metadata from maxblocks and minblocks
             - Calculate block scores using the same method as non-paged version
        2.3. Find the top-k indices across all valid blocks
    3. Return selected_indices
    """
    if fast:
        return ref_quest_paged_fast(query, maxblocks, minblocks, metadata_block_tables, seq_lens, 
                                    tokens_since_metadata_update, k)
    else:
        return ref_quest_paged_slow(query, maxblocks, minblocks, metadata_block_tables, seq_lens, 
                                    tokens_since_metadata_update, k)


def ref_quest_paged_slow(query: torch.Tensor,              # (batch_size, num_heads, head_dim)
                         maxblocks: torch.Tensor,          # (num_meta_blocks, block_size, num_kv_heads, head_dim)
                         minblocks: torch.Tensor,          # (num_meta_blocks, block_size, num_kv_heads, head_dim)
                         metadata_block_tables: torch.Tensor,  # (batch_size, mmbpr)
                         seq_lens: torch.Tensor,           # (batch_size)
                         tokens_since_metadata_update: int,
                         k: int) -> torch.Tensor:          # (batch_size, num_kv_heads, k)
    """
    SLOW paged quest reference functionality - explicit nested loop implementation
    """        
    batch_size, num_heads, head_dim = query.shape
    num_meta_blocks, block_size, num_kv_heads, head_dim_blocks = maxblocks.shape
    mmbpr = metadata_block_tables.shape[1]
    
    if query.dtype == torch.bfloat16: 
        query = query.float()   

    if head_dim != head_dim_blocks:
        raise ValueError(f"Query dimension {head_dim} doesn't match block dimension {head_dim_blocks}")
    
    # Output tensor for selected indices
    selected_indices = torch.zeros(batch_size, num_kv_heads, k, dtype=torch.int32, device=query.device) - 1
    
    # Reduce query across num_heads dimension to get grouped_query [batch_size, num_kv_heads, head_dim]
    group_size = num_heads // num_kv_heads
    grouped_query = torch.zeros(batch_size, num_kv_heads, head_dim, dtype=query.dtype, device=query.device)
    
    for group in range(num_kv_heads):
        # Sum all heads in this group
        group_sum = torch.zeros(batch_size, head_dim, dtype=query.dtype, device=query.device)
        for n in range(group_size):
            head_idx = group * group_size + n
            group_sum += query[:, head_idx, :]
        
        # Average to get grouped query
        grouped_query[:, group, :] = group_sum / group_size
    
    # Step 2: Process each (batch, head) pair instead of nested loops
    for b, n in [(b, n) for b in range(batch_size) for n in range(num_kv_heads)]:
        # Determine how many metadata blocks are valid for this request
        # yes, floor divide - assuming we created metadata tokens_since_metadata_update tokens ago, and only for the 
        # round  multiple of block_size tokens 
        num_kv_blocks_with_metadata = (seq_lens[b] - tokens_since_metadata_update) // block_size 
        num_valid_blocks = ceil_div(num_kv_blocks_with_metadata, block_size)
        num_valid_blocks = min(num_valid_blocks, mmbpr)  # Cap at mmbpr

        # Get current grouped query vector
        current_query = grouped_query[b, n, :]  # Shape: (head_dim,)
        
        # Process each valid metadata block
        all_scores = torch.zeros(num_valid_blocks * block_size, dtype=query.dtype, device=query.device)
        
        for block_idx in range(num_valid_blocks):
            # Get the actual metadata block index
            meta_block_id = metadata_block_tables[b, block_idx].item()

            # Get max and min blocks for this metadata block
            maxblock = maxblocks[meta_block_id, :, n, :]  # (block_size, head_dim)
            minblock = minblocks[meta_block_id, :, n, :]  # (block_size, head_dim)

            if maxblock.dtype == torch.bfloat16:
                maxblock = maxblock.float()                
            if minblock.dtype == torch.bfloat16:
                minblock = minblock.float()      
                
            
            # Elementwise multiply grouped_query with maxblock and minblock
            product_max = current_query.unsqueeze(0) * maxblock  # (block_size, head_dim)
            product_min = current_query.unsqueeze(0) * minblock  # (block_size, head_dim)
            
            # Elementwise max between product_min and product_max
            channel_max_product = torch.maximum(product_max, product_min)  # (block_size, head_dim)
            
            # Reduce sum the last dimension (head_dim to 1)
            scores = torch.sum(channel_max_product, dim=1)  # (block_size,)
            
            # Store scores with their global indices
            all_scores[block_idx * block_size: (block_idx + 1) * block_size] = scores
        
        # add sink
        if (tokens_since_metadata_update >= 0): 
            all_scores[0] = torch.finfo(all_scores.dtype).max

        eff_num_scores = len(all_scores)
        eff_k = min(k, eff_num_scores)            
        selected_indices[b, n, :] = torch.topk(all_scores, eff_k, dim=-1)[1] 

        # Add check whether last index should be added based on not-yet-updated metadata (as indicated by 
        # tokens_since_metadata_update and seq_len)
        if (tokens_since_metadata_update >= 0): 
            # mru = sequence length of this request at the most recent metadata update
            mru = seq_lens[b] - tokens_since_metadata_update;  

            # win_size = number of the most recent KV-blocks in the sequence, which are not yet in the metadata
            win_size = (mru % block_size != 0) + (seq_lens[b] // block_size) - (mru // block_size); 
            
            # faster computation version due to statically known fact that BLOCK_SZIE is a powers of two --> 7
            # int32_t win_size = ((mru & 0x7f) != 0) + (seq_lens[b] >> 7) - (mru >> 7); 
            for w in range(1, win_size + 1):
                selected_indices[b, n, k - w] = ((seq_lens[b] + block_size - 1) // block_size) - w

    return selected_indices


def ref_quest_paged_fast(query: torch.Tensor,              # (batch_size, num_heads, head_dim)
                         maxblocks: torch.Tensor,          # (num_meta_blocks, block_size, num_kv_heads, head_dim)
                         minblocks: torch.Tensor,          # (num_meta_blocks, block_size, num_kv_heads, head_dim)
                         metadata_block_tables: torch.Tensor,  # (batch_size, mmbpr)
                         seq_lens: torch.Tensor,           # (batch_size)
                         tokens_since_metadata_update: int,
                         k: int) -> torch.Tensor:          # (batch_size, num_kv_heads, k)
    """
    FAST paged quest reference functionality - vectorized implementation
    """
    mmbpr = metadata_block_tables.shape[1]
    num_meta_blocks, block_size, num_kv_heads, head_dim_blocks = maxblocks.shape
    batch_size, num_heads, head_dim = query.shape

    if query.dtype == torch.bfloat16:
        query = query.float()
    
    if head_dim != head_dim_blocks:
        raise ValueError(f"Query dimension {head_dim} doesn't match block dimension {head_dim_blocks}")
    
    # Step 1: Reduce query across num_heads dimension to get grouped_query [batch_size, num_kv_heads, head_dim]
    group_size = num_heads // num_kv_heads


    # The shape of grouped_query should be [batch_size, num_kv_heads, head_dim]
    grouped_query = query.view(batch_size, num_kv_heads, group_size, head_dim).mean(dim=2)  

    # Output tensor for selected indices
    selected_indices = torch.zeros(batch_size, num_kv_heads, k, dtype=torch.int32, device=query.device) - 1
    
    # Process each batch
    for b in range(batch_size):
        num_valid_blocks = min(ceil_div(seq_lens[b].item(), block_size * block_size), mmbpr)
        if num_valid_blocks == 0:
            continue
            
        # Grab the metadata block indices for this request
        meta_block_ids = metadata_block_tables[b, :num_valid_blocks]  # [num_valid_blocks]
        batch_query = grouped_query[b]  # [num_kv_heads, head_dim]
        
        # Get all relevant maxblocks and minblocks [num_valid_blocks, block_size, num_kv_heads, head_dim]
        relevant_minblocks = minblocks[meta_block_ids]  # [num_valid_blocks, block_size, num_kv_heads, head_dim]
        relevant_maxblocks = maxblocks[meta_block_ids]  # [num_valid_blocks, block_size, num_kv_heads, head_dim]

        if relevant_minblocks.dtype == torch.bfloat16:
            relevant_minblocks = relevant_minblocks.float()
        if relevant_maxblocks.dtype == torch.bfloat16:
            relevant_maxblocks = relevant_maxblocks.float()


        # Reshape for broadcasting: [num_kv_heads, head_dim] -> [1, 1, num_kv_heads, head_dim]
        batch_query_reshaped = batch_query.unsqueeze(0).unsqueeze(0)  # [1, 1, num_kv_heads, head_dim]
        
        # Elementwise multiply [num_valid_blocks, block_size, num_kv_heads, head_dim]
        product_max = batch_query_reshaped * relevant_maxblocks
        product_min = batch_query_reshaped * relevant_minblocks
        
        # Elementwise max [num_valid_blocks, block_size, num_kv_heads, head_dim]
        channel_max_product = torch.maximum(product_max, product_min)
        
        # Reduce sum along head_dim dimension [num_valid_blocks, block_size, num_kv_heads]
        block_scores = torch.sum(channel_max_product, dim=-1)
        
        # Reshape to combine blocks and block_size [num_valid_blocks * block_size, num_kv_heads]
        all_scores = block_scores.permute(2, 0, 1).reshape(num_kv_heads, -1)  

        # add sink
        if (tokens_since_metadata_update >= 0): 
            all_scores[:, 0] = torch.finfo(all_scores.dtype).max

        # Get top-k indices from the global indices
        eff_num_scores = all_scores.shape[-1]
        eff_k = min(k, eff_num_scores)    
        selected_indices[b, :, :eff_k] = torch.topk(all_scores, eff_k, dim=-1)[1]

        # Add check whether last index should be added based on not-yet-updated metadata (as indicated by 
        # tokens_since_metadata_update and seq_len)
        if (tokens_since_metadata_update >= 0): 
            # mru = sequence length of this request at the most recent metadata update
            mru = seq_lens[b] - tokens_since_metadata_update;  

            # number of the most recent KV-blocks in the sequence, which are not yet registered by the metadata
            win_size = (mru % block_size != 0) + (seq_lens[b] // block_size) - (mru // block_size); 
            
            # faster computation version due to statically known fact that BLOCK_SZIE is a powers of two --> 7
            # int32_t win_size = ((mru & 0x7f) != 0) + (seq_lens[b] >> 7) - (mru >> 7); 
            for w in range(1, win_size + 1):
                selected_indices[b, :, k - w] = ((seq_lens[b] + block_size - 1) // block_size) - w        
    
    return selected_indices