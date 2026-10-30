# -*- coding: utf-8 -*-
# Copyright (c) 2025 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
#
# synthetic data factory for quest_prefill_metadata

from typing import Tuple
import logging
import torch
import torch_npu

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(message)s')
logger = logging.getLogger(__name__)

SEED = 42


def gen_quest_prefill_inputs(
        batch_size: int, num_kv_heads: int, block_size: int, head_dim: int,
        num_kv_blocks: int = 100,
        num_meta_blocks: int = 20,
        mkbpr: int = 128,
        mmbpr: int = 1,
        same_seq_len_all_reqs: bool = False,
        dtype: torch.dtype = torch.float16,
        device: str = "npu:0",
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, 
           torch.Tensor, torch.Tensor]:
    """
    Creates pseudo-random inputs for quest_prefill_metadata kernel.
    Arguments:
        batch_size - batch size
        num_kv_heads - number of KV heads
        block_size - number of tokens per block (equal for metadata block and 
                     kv-cache block)
        head_dim - head dimension
        num_kv_blocks - total number of KV-cache blocks for the entire job (all 
                     requests together)
        num_meta_blocks - total number of metadata blocks for the entire job 
                    (all requests together)
        mkbpr - maximum number of KV-cache blocks per request in a batch    
        mmbpr - maximum number of metadata blocks per request in a batch    
        same_seq_len_all_reqs - True <--> all request in a batch will have the 
                     same sequence length (same number of kv blocks)

    Returns
    -------
        k_cache               : (num_kv_blocks, block_size, num_kv_heads, head_dim)
        block_tables          : (batch_size, mkbpr) indices into k_cache
        seq_lens              : (batch_size,) between block_size and block_size*mkbpr
        metadata_block_tables : (batch_size, mmbpr) indices into [0..num_meta_blocks-1]
        maxblocks             : (num_meta_blocks, block_size, num_kv_heads, head_dim) indices in 
                                [0..num_meta_blocks-1]
        minblocks             : (num_meta_blocks, block_size, num_kv_heads, head_dim) indices in 
                                [0..num_meta_blocks-1]
    """

    device = torch.device(device)
    
    # reset the SEED each time to be able to reproduce individual failed tests 
    # out of a loop of tests
    torch.manual_seed(SEED)    

    # ---- K-cache ---- #
    k_cache = torch.randn(num_kv_blocks, block_size, num_kv_heads, head_dim,
                          dtype=dtype, device=device) * 1.5 # 1.5 to increase the range

    # ---- request statistics ---- #
    max_seq_len = mkbpr * block_size
    if same_seq_len_all_reqs:
        seq_lens = torch.tensor([max_seq_len] * batch_size, dtype=torch.int32, device=device)
    else:
        seq_lens = torch.randint(low=0, high=max_seq_len + 1, size=(batch_size,), dtype=torch.int32, device=device)

    # ---- block tables ---- #
    perm_kv_blk_ids = torch.randperm(num_kv_blocks, device=device)[:num_kv_blocks]      
    block_tables = perm_kv_blk_ids.reshape((batch_size, mkbpr)).to(dtype=torch.int32, device=device)                       

    # ---- metadata_block_tables ---- #
    perm_meta_blk_ids = torch.randperm(num_meta_blocks, device=device)[:num_meta_blocks]      
    metadata_block_tables = perm_meta_blk_ids.reshape((batch_size, mmbpr)).to(dtype=torch.int32, device=device)      

    # placeholder for kernel outputs
    maxblocks = torch.empty(num_meta_blocks, block_size, num_kv_heads, head_dim, dtype=dtype, device=device)
    minblocks = torch.empty(num_meta_blocks, block_size, num_kv_heads, head_dim, dtype=dtype, device=device)

    # Ensure all tensors are contiguous
    k_cache = k_cache.contiguous()
    block_tables = block_tables.contiguous()
    seq_lens = seq_lens.contiguous()
    metadata_block_tables = metadata_block_tables.contiguous()
    maxblocks = maxblocks.contiguous()
    minblocks = minblocks.contiguous()

    ret = (k_cache, block_tables, seq_lens, metadata_block_tables, maxblocks, minblocks)
    
    return ret


def compare_tensors(ref: torch.Tensor, custom: torch.Tensor, *, 
                    rtol: float = 1e-2, atol: float = 1e-3, 
                    verbose: bool = True) -> bool:
    """
    compare tensors with a relaxed fp16 tolerance
    """
    if ref.shape != custom.shape:
        logger.info(f"ERROR: shape mismatch  ref={ref.shape}  custom={custom.shape}")
        return False
    try:
        torch.testing.assert_close(ref, custom, rtol=rtol, atol=atol)
        if verbose: 
            logger.info("PASSED")
        return True
    except AssertionError as e:
        if verbose: 
            logger.info("FAILED")
        logger.info(e)
        return False


def ceil_div(x: int, y: int) -> int:
    return (x + y - 1) // y