# -*- coding: utf-8 -*-
# Copyright (c) 2025 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.

import logging
import math
from typing import Tuple, List
import torch
import torch_npu

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(message)s')
logger = logging.getLogger(__name__)

SEED = 42


def gen_quest_paged_inputs(batch_size: int, 
                           num_heads: int,  # number of query heads
                           num_kv_heads: int,  # number of KV heads
                           block_size: int, 
                           head_dim: int,
                           num_meta_blocks: int,  # number of metadata blocks to allocate (not all of them will be 
                                                  # used. Effectively, only just-enough blocks needed for the sequence 
                                                  # length of each request will be actually loaded by 
                                                  # the quest_block_select_paged kernel)
                           mmbpr: int,  # Max Metadata Blocks Per Request
                           same_seq_len_all_reqs: bool = True,
                           device: str = 'npu:0',
                           dtype: torch.dtype = torch.float16
                           ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    Produces random matrices for paged Quest block selection operation:
    - query: [batch_size, num_heads, head_dim] 
    - maxblocks: [num_meta_blocks, block_size, num_kv_heads, head_dim]
    - minblocks: [num_meta_blocks, block_size, num_kv_heads, head_dim]
    - metadata_block_tables: [batch_size, mmbpr]
    - seq_lens: [batch_size]
    """
    if not (num_meta_blocks >= batch_size * mmbpr):
        raise ValueError(f"num_meta_blocks ({num_meta_blocks}) must be >= batch_size * mmbpr ({batch_size * mmbpr})")

    torch.manual_seed(SEED)
    metadata_block_tables = torch.randint(0, num_meta_blocks, (batch_size, mmbpr), dtype=torch.int32)
    minblocks = torch.empty(num_meta_blocks, block_size, num_kv_heads, head_dim, dtype=dtype).uniform_(-1, 1)
    maxblocks = torch.empty(num_meta_blocks, block_size, num_kv_heads, head_dim, dtype=dtype).uniform_(-1, 1)
    query = torch.empty(batch_size, num_heads, head_dim, dtype=dtype).uniform_(-1, 1)
    
    # Generate sequence lengths [batch_size]
    # Sequence lengths should be reasonable values (e.g., between 1 and some max length)
    max_seq_len = mmbpr * block_size * block_size
    if same_seq_len_all_reqs:
        seq_lens = torch.tensor([max_seq_len] * batch_size, dtype=torch.int32, device=device)
    else:
        seq_lens = torch.randint(low=0, high=max_seq_len + 1, size=(batch_size,), dtype=torch.int32, device=device)
    
    # Move all tensors to device global memory (GM)
    query = query.to(device).contiguous()
    maxblocks = maxblocks.to(device).contiguous()
    minblocks = minblocks.to(device).contiguous()
    metadata_block_tables = metadata_block_tables.to(device).contiguous()
    seq_lens = seq_lens.to(device).contiguous()
    
    return query, maxblocks, minblocks, metadata_block_tables, seq_lens


def compare_indices(reference: torch.Tensor, custom: torch.Tensor, 
                    tol_percentage: float = 0.02, 
                    verbose: bool = False) -> bool:
    """
    Compares tensors of integer numbers, requiring the last dimension to contain 
    the same set of numbers.

    Args:
    - reference: Tensor assumed to be correct
    - custom: Tesnor under test
    - compare the last dimension of "reference" vs "custom", disregarding the 
      order (set comparison).
    - tol_percentage - fraction of total elemetns that are allowed to be 
      incorrect. If higher than 0.0, then at least 1 element will be allowed to
      be corrupt.
    - verbose = True <--> print PASSED or FAILED, and specify the number of 
      violations.
    
    Return:
     True <--> reference and custom contain the same (within tolerance)
    """

    sorted_reference = torch.sort(reference, dim=-1).values
    sorted_custom = torch.sort(custom, dim=-1).values

    # Precompute the difference count: number of elements in custom not in reference for each [b, n]
    batch_size, num_kv_heads, k = reference.shape
    diff_count = torch.zeros((batch_size, num_kv_heads), dtype=torch.int32, device=reference.device)

    tol_count = int(math.ceil(batch_size * num_kv_heads * k * tol_percentage))

    # Iterate over each element in the batch and the second dimension
    for b in range(batch_size):
        for n in range(num_kv_heads):
            ref_set = set(reference[b, n].cpu().numpy())
            custom_set = set(custom[b, n].cpu().numpy())
            diff_count[b, n] = len(ref_set - custom_set)
    n_violating_elems = diff_count.sum().sum()
    
    # test summary
    test_ok = n_violating_elems <= tol_count
    if verbose: 
        logger.info(f"{'PASSED' if test_ok else 'FAILED'} - ", end='')
        logger.info(f"{n_violating_elems}/{reference.numel()} indices are incorrect (allowed:{tol_count})")  
    
    return test_ok


def ceil_div(x: int, y: int) -> int:
    return (x + y - 1) // y
