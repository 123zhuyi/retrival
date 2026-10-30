# -*- coding: utf-8 -*-
# Copyright (c) 2025 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.

"""
Single input testing (for debugging) -> Run this file with python <filename>
Wide range testing (for validation) -> Run this file with pytest <filename>
"""
import logging
from itertools import product
from typing import Callable
import pytest
import torch
import torch_npu
from select_attn_ops import quest_block_select_paged, quest_block_select_paged_in_out
from gen_data_quest_block_select_paged import gen_quest_paged_inputs, compare_indices
from ref_quest_block_select_paged import ref_quest_block_select_paged


# Configure logging
logging.basicConfig(level=logging.INFO, format='%(message)s')
logger = logging.getLogger(__name__)

DEVICE = "npu:0"
BLOCK_SIZE = 128
HEAD_DIM = 128
SAME_SEQ_LEN_ALL_REQS = False


# --------------------------------------------------------------------------- #
# Central test worker  (assertion crashes <--> test failed)
# --------------------------------------------------------------------------- #
@pytest.mark.skip(reason="Skipping direct invocation of the main test_quest_paged_kernel function")
def test_quest_paged_kernel(custom_kernel: Callable, dtype: torch.dtype, batch_size: int, num_heads: int, 
                            num_kv_heads: int, block_size: int, head_dim: int, mmbpr: int, k: int, 
                            verbose: bool = False) -> None:
    """
    Main test function for paged quest kernel
    """
    # Generate input data
    num_meta_blocks = batch_size * mmbpr
    query, maxblocks, minblocks, metadata_block_tables, seq_lens = gen_quest_paged_inputs(
        batch_size, num_heads, num_kv_heads, block_size, head_dim, num_meta_blocks, mmbpr, SAME_SEQ_LEN_ALL_REQS,
        DEVICE, dtype
    )

    # Run reference implementation
    ref_ids = ref_quest_block_select_paged(
        query, maxblocks, minblocks, metadata_block_tables, seq_lens, k, fast=True
    )
    
    # Run custom implementation
    if custom_kernel == quest_block_select_paged:
        custom_ids = quest_block_select_paged(
            query, maxblocks, minblocks, metadata_block_tables, seq_lens, k
        )
    elif custom_kernel == quest_block_select_paged_in_out:
        custom_ids = torch.zeros_like(ref_ids)
        quest_block_select_paged_in_out(
            query, maxblocks, minblocks, metadata_block_tables, seq_lens, custom_ids
        )        
    else:
        raise ValueError(f"Unknown custom kernel: {custom_kernel}")
    
    # Compare results
    cfg_str = f"{dtype=}, {batch_size=}, {num_heads=}, {num_kv_heads=}, {block_size=}, {head_dim=}, {mmbpr=}, {k=}"
    if verbose:
        logger.info("Input shapes:")
        logger.info(f"  query: {query.shape}")
        logger.info(f"  maxblocks: {maxblocks.shape}")
        logger.info(f"  minblocks: {minblocks.shape}")
        logger.info(f"  metadata_block_tables: {metadata_block_tables.shape}")
        logger.info(f"  seq_lens: {seq_lens.shape}")
        logger.info(f"  Sequence lengths: {seq_lens.tolist()}")
        logger.info(" ========== Reference torch implementation output ========= ")
        logger.info("ids ref:", ref_ids)
        logger.info(ref_ids.shape)       
        logger.info(" ========== Custom ascendc implementation output ========== ")
        logger.info("ids custom:", custom_ids)
        logger.info(custom_ids.shape)
        logger.info(" ========== DIFF =========== ")
        logger.info("ids difference:", ref_ids - custom_ids)
        logger.info(" ==================== SUMMARY ================== ")
        logger.info(cfg_str)
    indices_match = compare_indices(ref_ids, custom_ids, verbose=verbose)               
    
    # Assert all comparisons pass
    assert indices_match, f"Indices comparison failed for {cfg_str}"


def construct_quest_paged_parameter_sets(custom_kernel_vals, dtype_vals, batch_size_vals, num_heads_vals, 
                                         num_kv_heads_vals, mmbpr_vals, k_vals):
    """
    Construct parameter sets for paged Quest kernel testing with constraints:
    - head_dim, block_size are fixed
    - num_kv_heads ≤ num_heads
    - num_heads is a multiple of num_kv_heads (num_heads = num_kv_heads * G for natural G)
    """
    param_set_lst = []
    for ck, dtype, b, h, n, mmbpr, k in product(custom_kernel_vals, dtype_vals, batch_size_vals, 
                                               num_heads_vals, num_kv_heads_vals, mmbpr_vals, k_vals):
        # Input constraints: 
        # num_kv_heads ≤ num_heads and num_heads is multiple of num_kv_heads
        constraint_a = n <= h and h % n == 0
        # k should be multiple of 8 for "out-preallcated" kernel
        constraint_b = k % 8 == 0

        # Apply constraints
        if constraint_a and constraint_b:
            param_set_lst.append((ck, dtype, b, h, n, BLOCK_SIZE, HEAD_DIM, mmbpr, k))
    return param_set_lst



########################### Test 1 - Basic functionality ###########################
parameter_sets = construct_quest_paged_parameter_sets(custom_kernel_vals=[quest_block_select_paged, 
                                                                          quest_block_select_paged_in_out],
                                                      dtype_vals=[torch.float16, torch.bfloat16], 
                                                      batch_size_vals=[1, 2], 
                                                      num_heads_vals=[8, 16], 
                                                      num_kv_heads_vals=[4, 8], 
                                                      mmbpr_vals=[1, 2, 3, 4], 
                                                      k_vals=[4, 8])


@pytest.mark.parametrize(
    "custom_kernel, dtype, batch_size, num_heads, num_kv_heads, block_size, head_dim, mmbpr, k", parameter_sets,
    ids=[f"custom_kernel={ck}, dtype={dtype}, batch_size={b}, num_heads={h}, num_kv_heads={n}, "
         f"block_size={block_size}, head_dim={d}, mmbpr={mmbpr}, k={k}" 
         for ck, dtype, b, h, n, block_size, d, mmbpr, k in parameter_sets]    
)


@torch.inference_mode()
def test_quest_paged_basic(custom_kernel: Callable, dtype: torch.dtype, batch_size: int, num_heads: int, 
                           num_kv_heads: int, block_size: int, head_dim: int, mmbpr: int, k: int) -> None:
    test_quest_paged_kernel(custom_kernel, dtype, batch_size, num_heads, num_kv_heads, block_size, head_dim, mmbpr, k)

########################### Test 2 - Edge cases ###########################
parameter_sets = construct_quest_paged_parameter_sets(custom_kernel_vals=[quest_block_select_paged, 
                                                                          quest_block_select_paged_in_out],
                                                      dtype_vals=[torch.float16, torch.bfloat16],
                                                      batch_size_vals=[1], 
                                                      num_heads_vals=[1, 2, 4, 8, 16, 32], 
                                                      num_kv_heads_vals=[1, 2, 4, 8, 16], 
                                                      mmbpr_vals=[1, 2, 4], 
                                                      k_vals=[1, 4, 16])


@pytest.mark.parametrize(
    "custom_kernel, dtype, batch_size, num_heads, num_kv_heads, block_size, head_dim, mmbpr, k", parameter_sets,
    ids=[f"custom_kernel={ck}, dtype={dtype}, batch_size={b}, num_heads={h}, num_kv_heads={n}, "
         f"block_size={block_size}, head_dim={d}, mmbpr={mmbpr}, k={k}" 
         for ck, dtype, b, h, n, block_size, d, mmbpr, k in parameter_sets]  
)


@torch.inference_mode()
def test_quest_paged_edge_cases(custom_kernel: Callable, dtype: torch.dtype, batch_size: int, num_heads: int, 
                                num_kv_heads: int, block_size: int, head_dim: int, mmbpr: int, k: int) -> None:
    test_quest_paged_kernel(custom_kernel, dtype, batch_size, num_heads, num_kv_heads, block_size, head_dim, mmbpr, k)

########################### Test 3 - Extensive testing ###########################
parameter_sets = construct_quest_paged_parameter_sets(custom_kernel_vals=[quest_block_select_paged, 
                                                                          quest_block_select_paged_in_out],
                                                      dtype_vals=[torch.float16, torch.bfloat16], 
                                                      batch_size_vals=[1, 2, 8], 
                                                      num_heads_vals=[8, 16, 32, 64], 
                                                      num_kv_heads_vals=[2, 4, 8, 16], 
                                                      mmbpr_vals=[1, 2, 3, 4], 
                                                      k_vals=[2, 4, 8, 16, 32])


@pytest.mark.parametrize(
    "custom_kernel, dtype, batch_size, num_heads, num_kv_heads, block_size, head_dim, mmbpr, k", parameter_sets,
    ids=[f"custom_kernel={ck}, dtype={dtype}, batch_size={b}, num_heads={h}, num_kv_heads={n}, "
         f"block_size={block_size}, head_dim={d}, mmbpr={mmbpr}, k={k}" 
         for ck, dtype, b, h, n, block_size, d, mmbpr, k in parameter_sets]  
)


@torch.inference_mode()
def test_quest_paged_extensive(custom_kernel: Callable, dtype: torch.dtype, batch_size: int, num_heads: int, 
                               num_kv_heads: int, block_size: int, head_dim: int, mmbpr: int, k: int) -> None:
    test_quest_paged_kernel(custom_kernel, dtype, batch_size, num_heads, num_kv_heads, block_size, head_dim, mmbpr, k)

########################### Test 4 - Large scale ###########################
parameter_sets = construct_quest_paged_parameter_sets(custom_kernel_vals=[quest_block_select_paged, 
                                                                          quest_block_select_paged_in_out],
                                                      dtype_vals=[torch.float16, torch.bfloat16], 
                                                      batch_size_vals=[16, 20, 24, 32], 
                                                      num_heads_vals=[32, 64, 128], 
                                                      num_kv_heads_vals=[4, 8, 16, 32, 64], 
                                                      mmbpr_vals=[1, 2, 4], 
                                                      k_vals=[4, 8, 32, 64])


@pytest.mark.parametrize(
    "custom_kernel, dtype, batch_size, num_heads, num_kv_heads, block_size, head_dim, mmbpr, k", parameter_sets,
    ids=[f"custom_kernel={ck}, dtype={dtype}, batch_size={b}, num_heads={h}, num_kv_heads={n}, "
         f"block_size={block_size}, head_dim={d}, mmbpr={mmbpr}, k={k}" 
         for ck, dtype, b, h, n, block_size, d, mmbpr, k in parameter_sets]  
)


@torch.inference_mode()
def test_quest_paged_large_scale(custom_kernel: Callable, dtype: torch.dtype, batch_size: int, num_heads: int, 
                                 num_kv_heads: int, block_size: int, head_dim: int, mmbpr: int, k: int) -> None:
    test_quest_paged_kernel(custom_kernel, dtype, batch_size, num_heads, num_kv_heads, block_size, head_dim, mmbpr, k)


# --------------------------------------------------------------------------- #
# Quick manual run (kept for copy-paste debugging)
# ---------------------------------------------------------------------------#
if __name__ == "__main__":
    test_quest_paged_kernel(custom_kernel=quest_block_select_paged_in_out,
                            dtype=torch.bfloat16, batch_size=20, num_heads=32, num_kv_heads=8, block_size=128, 
                            head_dim=128, mmbpr=1, k=16, verbose=True)  # fails bfloat16
    logger.info("Manual smoke test PASSED")
