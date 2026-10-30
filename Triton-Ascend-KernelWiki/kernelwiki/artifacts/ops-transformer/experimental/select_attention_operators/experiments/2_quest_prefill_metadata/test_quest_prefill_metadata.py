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
from itertools import product
import logging
import pytest
import torch
import torch_npu
from select_attn_ops import quest_prefill_metadata
from ref_quest_prefill_metadata import ref_quest_prefill_metadata
from gen_data_quest_prefill_metadata import gen_quest_prefill_inputs, compare_tensors, ceil_div


# Configure logging
logging.basicConfig(level=logging.INFO, format='%(message)s')
logger = logging.getLogger(__name__)

DEVICE = "npu:0"
BLOCK_SIZE = 128
HEAD_DIM = 128


# --------------------------------------------------------------------------- #
# Central test worker  (assertion crashes <--> test failed)
# --------------------------------------------------------------------------- #
@pytest.mark.skip(reason="internal worker - not called directly by pytest")
def test_prefill_kernel(dtype: torch.dtype, 
                        batch_size: int, 
                        num_kv_heads: int, 
                        block_size: int, 
                        head_dim: int,
                        mkbpr: int = 128,
                        mmbpr: int = 1,
                        ssar: bool = True,
                        verbose: bool = False) -> None:
    """Run reference vs Ascend-C and assert bit-accurate match.
    
    Arguments:
        batch_size - batch size
        num_kv_heads - number of KV heads
        block_size - nmuumbe of tokens per block (equal for metadata block and 
                     kv-cahce block)
        head_dim - head dimeansion
        mkbpr - maximum number of KV-cache blocks per request in a batch    
        mmbpr - maximum number of metadata blocks per request in a batch    
        ssar - "same_seq_len_all_reqs" 
               if true, the the data generated will have same sequence length 
               for all request in the batch.
               if false: each request wll have a randomly number of tokens.
    
    """

    mmbpr = ceil_div(mkbpr, block_size)
    num_kv_blocks = batch_size * mkbpr  # total number of KV-cache blocks for the entire job (all requests together)
    num_meta_blocks = batch_size * mmbpr  # total number of metadata blocks for the entire job (all requests together)

    # ---- generate input/output tensors ---- #
    k_cache, block_tables, seq_lens, meta_ids, max_out, min_out = gen_quest_prefill_inputs(
        batch_size, num_kv_heads, block_size, head_dim,
        num_kv_blocks=num_kv_blocks,
        num_meta_blocks=num_meta_blocks,
        mkbpr=mkbpr,
        mmbpr=mmbpr,
        same_seq_len_all_reqs=ssar, 
        dtype=dtype,
        device=DEVICE)
    
    # ---- reference kernel ---- #
    max_ref = min_out.clone() 
    min_ref = max_out.clone()
    ref_quest_prefill_metadata(
        k_cache, block_tables, seq_lens, meta_ids,
        max_ref, min_ref)

    # ---- custom kernel ---- #
    quest_prefill_metadata(
        k_cache, block_tables, seq_lens, meta_ids,
        max_out, min_out)

    # ---- compare ---- #
    max_ok = compare_tensors(max_ref, max_out)
    min_ok = compare_tensors(min_ref, min_out)
    assert max_ok and min_ok, "maxblocks or minblocks mismatch"
    if verbose:
        logger.info(" ==================== maxblocks =================== ")
        logger.info(f"{max_ref=}")
        logger.info(f"{max_out=}")
        logger.info(" ==================== minblocks =================== ")
        logger.info(f"{min_ref=}")
        logger.info(f"{min_out=}")    
        logger.info(" ==================== SUMMARY =================== ")
        logger.info(f"{batch_size=} {num_kv_heads=} {block_size=} {head_dim=} "
                    f"{mkbpr=} {mmbpr=} {dtype=} {num_kv_blocks=} {num_meta_blocks=}")
        logger.info("maxblocks - ", end='')
        compare_tensors(max_ref, max_out)    
        logger.info("minblocks - ", end='')
        compare_tensors(min_ref, min_out)    


# --------------------------------------------------------------------------- #
# 3.  Param-set builder  (same pattern as block-select file)
# --------------------------------------------------------------------------- #
@pytest.mark.skip(reason="internal helper")
def construct_prefill_parameter_sets(dtype_vals, batch_size_vals, num_kv_heads_vals, mkbpr_vals, ssar_vals):
    """
    Build legal (batch_size, num_kv_heads, mkbpr) tuples.
    block_size & head_dim are global constants.
    """
    param_set_lst = []
    for dtype, b, n, mkbpr, ssar in product(dtype_vals, batch_size_vals, num_kv_heads_vals, mkbpr_vals, ssar_vals):
        param_set_lst.append((dtype, b, n, mkbpr, ssar))
    return param_set_lst


# --------------------------------------------------------------------------- #
# Test 1 – Basic functionality
# ---------------------------------------------------------------------------#
parameter_sets = construct_prefill_parameter_sets(dtype_vals=[torch.float16, torch.bfloat16], 
                                                  batch_size_vals=[1, 2], 
                                                  num_kv_heads_vals=[4, 8], 
                                                  mkbpr_vals=[1, 64, 126, 128], 
                                                  ssar_vals=[True, False])


@pytest.mark.parametrize(
    "dtype, batch_size, num_kv_heads, mkbpr, ssar", parameter_sets,
    ids=[f"{dtype=},{batch_size=},{num_kv_heads=},{mkbpr=},{ssar=}" 
         for dtype, batch_size, num_kv_heads, mkbpr, ssar in parameter_sets]
)
@torch.inference_mode()
def test_basic_functionality(dtype: torch.dtype, batch_size: int, num_kv_heads: int, mkbpr: int, ssar: int):
    test_prefill_kernel(dtype, batch_size, num_kv_heads, BLOCK_SIZE, HEAD_DIM, mkbpr, ssar)


# --------------------------------------------------------------------------- #
# Test 2 – Edge cases
# ---------------------------------------------------------------------------#
parameter_sets = construct_prefill_parameter_sets(dtype_vals=[torch.float16, torch.bfloat16], 
                                                  batch_size_vals=[1, 2], 
                                                  num_kv_heads_vals=[1, 2, 4, 7, 8, 9, 16, 21, 32, 33], 
                                                  mkbpr_vals=[1, 2, 3, 63, 64, 65, 126, 127, 128, 129, 
                                                              150, 255, 256, 257], 
                                                  ssar_vals=[True, False])


@pytest.mark.parametrize(
    "dtype, batch_size, num_kv_heads, mkbpr, ssar", parameter_sets,
    ids=[f"{dtype=},{batch_size=},{num_kv_heads=},{mkbpr=},{ssar=}" 
         for dtype, batch_size, num_kv_heads, mkbpr, ssar in parameter_sets]
)
@torch.inference_mode()
def test_edge_cases(dtype: torch.dtype, batch_size: int, num_kv_heads: int, mkbpr: int, ssar: int):
    test_prefill_kernel(dtype, batch_size, num_kv_heads, BLOCK_SIZE, HEAD_DIM, mkbpr, ssar)


# --------------------------------------------------------------------------- #
# Test 3 – Test Large sequence
# ---------------------------------------------------------------------------#
parameter_sets = construct_prefill_parameter_sets(dtype_vals=[torch.float16, torch.bfloat16], 
                                                  batch_size_vals=[1, 2, 4, 8], 
                                                  num_kv_heads_vals=[2, 4, 8], 
                                                  mkbpr_vals=[1, 64, 126, 128, 130, 135, 150, 151, 170, 
                                                              200, 210, 211, 212, 256, 300, 400, 512], 
                                                  ssar_vals=[True, False])


@pytest.mark.parametrize(
    "dtype, batch_size, num_kv_heads, mkbpr, ssar", parameter_sets,
    ids=[f"{dtype=},{batch_size=},{num_kv_heads=},{mkbpr=},{ssar=}" 
         for dtype, batch_size, num_kv_heads, mkbpr, ssar in parameter_sets]
)
@torch.inference_mode()
def test_large_lequence(dtype: torch.dtype, batch_size: int, num_kv_heads: int, mkbpr: int, ssar: int):
    test_prefill_kernel(dtype, batch_size, num_kv_heads, BLOCK_SIZE, HEAD_DIM, mkbpr, ssar)


# --------------------------------------------------------------------------- #
# Test 4 – Large batch
# ---------------------------------------------------------------------------#
parameter_sets = construct_prefill_parameter_sets(dtype_vals=[torch.float16, torch.bfloat16], 
                                                  batch_size_vals=[16, 20, 24, 32], 
                                                  num_kv_heads_vals=[4, 8, 16], 
                                                  mkbpr_vals=[1, 64, 126, 128, 130, 141], 
                                                  ssar_vals=[True, False])


@pytest.mark.parametrize(
    "dtype, batch_size, num_kv_heads, mkbpr, ssar", parameter_sets,
    ids=[f"{dtype=},{batch_size=},{num_kv_heads=},{mkbpr=},{ssar=}" 
         for dtype, batch_size, num_kv_heads, mkbpr, ssar in parameter_sets]
)
@torch.inference_mode()
def test_large_batch(dtype: torch.dtype, batch_size: int, num_kv_heads: int, mkbpr: int, ssar: int):
    test_prefill_kernel(dtype, batch_size, num_kv_heads, BLOCK_SIZE, HEAD_DIM, mkbpr, ssar)

# --------------------------------------------------------------------------- #
# Quick manual run (kept for copy-paste debugging)
# ---------------------------------------------------------------------------#
if __name__ == "__main__":
    test_prefill_kernel(dtype=torch.bfloat16, batch_size=20, num_kv_heads=8, block_size=128, 
                        head_dim=128, mkbpr=128, ssar=False, verbose=True) # passes 
    logger.info("Manual smoke test PASSED")
