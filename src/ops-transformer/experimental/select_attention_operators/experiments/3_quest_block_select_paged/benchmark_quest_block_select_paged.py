# -*- coding: utf-8 -*-
# Copyright (c) 2025 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.

"""

Benchmark driver for quest_block_select_paged kernel.

Measures:
*  latency (usec) - NPU timer
*  effective bandwidth (TB/s) - bytes moved / time
*  correctness comparison with reference implementation
"""
import logging
from typing import Callable, Tuple, List, Any
import itertools
import torch
import torch_npu

from select_attn_ops import quest_block_select_paged, quest_block_select_paged_in_out
from ref_quest_block_select_paged import ref_quest_block_select_paged
from gen_data_quest_block_select_paged import gen_quest_paged_inputs, ceil_div, compare_indices


# Configure logging
logging.basicConfig(level=logging.INFO, format='%(message)s')
logger = logging.getLogger(__name__)

torch.npu.set_device("npu:0")
BLOCK_SIZE = 128
HEAD_DIM = 128
SAME_SEQ_LEN_ALL_REQS = True # to set equally long input length and avoid unknown actual size


# --------------------------------------------------------------------------- #
#  bytes-moved calculator
# --------------------------------------------------------------------------- #
def bytes_moved_paged_select(batch_size: int, num_heads: int, num_kv_heads: int, block_size: int, 
                            head_dim: int, mmbpr: int, k: int, seq_lens: torch.Tensor) -> int:
    """
    Global-memory traffic (read + write) for quest_block_select_paged.

    Reads:
    - query: [batch_size, num_heads, head_dim] - batch_size * num_heads * head_dim * 2 bytes (FP16)
    - maxblocks: [?, block_size, num_kv_heads, head_dim] - num_effective_metadata_blocks * block_size * 
                                                            num_kv_heads * head_dim * 2 bytes (FP16)
    - minblocks: [?, block_size, num_kv_heads, head_dim] - num_effective_metadata_blocks * block_size * 
                                                            num_kv_heads * head_dim * 2 bytes (FP16)
    - metadata_block_tables: [batch_size, mmbpr] - batch_size * mmbpr * 4 bytes (INT32)
    - seq_lens: [batch_size] - batch_size * 4 bytes (INT32)

    Writes:
    - selected_indices: [batch_size, num_kv_heads, k] - batch_size * num_kv_heads * k * 4 bytes (INT32)
    """
    toks_per_meta_block = block_size * block_size
    num_effective_metadata_blocks = torch.sum(ceil_div(seq_lens, toks_per_meta_block)).item()

    read_query = batch_size * num_heads * head_dim * 2
    read_maxblocks = num_effective_metadata_blocks * block_size * num_kv_heads * head_dim * 2
    read_minblocks = num_effective_metadata_blocks * block_size * num_kv_heads * head_dim * 2
    read_metadata_tables = batch_size * mmbpr * 4
    read_seq_lens = seq_lens.element_size() * seq_lens.numel()
    
    write_indices = batch_size * num_kv_heads * k * 4
    
    return read_query + read_maxblocks + read_minblocks + read_metadata_tables + read_seq_lens + write_indices


# --------------------------------------------------------------------------- #
#  Configuration and setup functions
# --------------------------------------------------------------------------- #
def get_benchmark_input_sets(
        custom_kernel: Callable
        ) -> Tuple[List[int], List[int], List[int], List[int], List[int]]:
    """Get benchmark configuration parameters."""
    batch_size_vals = [10, 20, 24, 32]
    num_heads_vals = [32]
    num_kv_heads_vals = [8]
    mmbpr_vals = [1, 2, 4, 6]
    
    if custom_kernel == quest_block_select_paged:
        k_vals = [4, 8, 12, 16]
    elif custom_kernel == quest_block_select_paged_in_out:
        k_vals = [8, 16, 24, 32]
    else:
        raise ValueError(f"Unknown custom_kernel: {custom_kernel}")
    
    return batch_size_vals, num_heads_vals, num_kv_heads_vals, mmbpr_vals, k_vals


def generate_input_sets(
        n_warmup: int, n_repeat: int, b: int, h: int, n: int, mmbpr: int, dtype: torch.dtype
        ) -> List[Tuple]:
    """Generate multiple input sets for benchmarking."""
    input_sets = []
    for _ in range(n_warmup + n_repeat):
        query, maxblocks, minblocks, metadata_block_tables, seq_lens = gen_quest_paged_inputs(
            b, h, n, BLOCK_SIZE, HEAD_DIM,
            num_meta_blocks=b * mmbpr,
            mmbpr=mmbpr,
            same_seq_len_all_reqs=SAME_SEQ_LEN_ALL_REQS,
            device="npu:0", 
            dtype=dtype)
        input_sets.append((query, maxblocks, minblocks, metadata_block_tables, seq_lens))
    return input_sets


# --------------------------------------------------------------------------- #
#  Correctness checking
# --------------------------------------------------------------------------- #
def check_correctness(
        custom_kernel: Callable, b: int, h: int, n: int, mmbpr: int, k: int, dtype: torch.dtype
        ) -> str:
    """Check correctness between custom and reference implementations."""
    query, maxblocks, minblocks, metadata_block_tables, seq_lens = gen_quest_paged_inputs(
        b, h, n, BLOCK_SIZE, HEAD_DIM,
        num_meta_blocks=b * mmbpr,
        mmbpr=mmbpr,
        same_seq_len_all_reqs=SAME_SEQ_LEN_ALL_REQS,
        device="npu:0", 
        dtype=dtype)
    
    ref_ids = ref_quest_block_select_paged(query, maxblocks, minblocks, metadata_block_tables, seq_lens, k)
    
    if custom_kernel == quest_block_select_paged:
        our_ids = quest_block_select_paged(query, maxblocks, minblocks, metadata_block_tables, seq_lens, k)
    elif custom_kernel == quest_block_select_paged_in_out:
        our_ids = torch.zeros((b, n, k), dtype=torch.int32, device=query.device)
        quest_block_select_paged_in_out(query, maxblocks, minblocks, metadata_block_tables, seq_lens, our_ids)
    else:
        raise ValueError(f"Unknown custom_kernel: {custom_kernel}")            
    
    tol_percentage = 0.02
    are_equal = compare_indices(ref_ids, our_ids, tol_percentage, verbose=False)
    return "yes" if are_equal else "no"


# --------------------------------------------------------------------------- #
#  Benchmark execution functions
# --------------------------------------------------------------------------- #
def run_warmup(
        custom_kernel: Callable, input_sets: List[Tuple], n_warmup: int, k: int, our_ids: torch.Tensor = None
        ) -> None:
    """Run warmup iterations."""
    for i in range(n_warmup):
        query, maxblocks, minblocks, metadata_block_tables, seq_lens = input_sets[i]
        if custom_kernel == quest_block_select_paged:
            quest_block_select_paged(query, maxblocks, minblocks, metadata_block_tables, seq_lens, k)
        elif custom_kernel == quest_block_select_paged_in_out:
            quest_block_select_paged_in_out(query, maxblocks, minblocks, metadata_block_tables, seq_lens, our_ids)
        else:
            raise ValueError(f"Unknown custom_kernel: {custom_kernel}")
        
        ref_quest_block_select_paged(query, maxblocks, minblocks, metadata_block_tables, seq_lens, k)
    
    torch.npu.synchronize()


def benchmark_implementation(custom_kernel: Callable, input_sets: List[Tuple], n_warmup: int, 
                           n_repeat: int, k: int, b: int, h: int, n: int, mmbpr: int, 
                           our_ids: torch.Tensor = None) -> Tuple[float, float]:
    """Benchmark a single implementation and return duration and bandwidth."""
    # Warmup
    run_warmup(custom_kernel, input_sets, n_warmup, k, our_ids)
    
    # Measurement
    start = torch.npu.Event(enable_timing=True)
    end = torch.npu.Event(enable_timing=True)
    
    start.record()
    for i in range(n_warmup, n_warmup + n_repeat):
        query, maxblocks, minblocks, metadata_block_tables, seq_lens = input_sets[i]
        if custom_kernel == quest_block_select_paged:
            quest_block_select_paged(query, maxblocks, minblocks, metadata_block_tables, seq_lens, k)
        elif custom_kernel == quest_block_select_paged_in_out:
            quest_block_select_paged_in_out(query, maxblocks, minblocks, metadata_block_tables, seq_lens, our_ids)
        else:
            raise ValueError(f"Unknown custom_kernel: {custom_kernel}")
    end.record()
    torch.npu.synchronize()
    
    duration = start.elapsed_time(end) / n_repeat * 1000  # ms to μs
    total_bytes = bytes_moved_paged_select(b, h, n, BLOCK_SIZE, HEAD_DIM, mmbpr, k, input_sets[0][4])
    bandwidth = total_bytes / duration / 1e6  # TB/s
    
    return duration, bandwidth


def benchmark_reference(input_sets: List[Tuple], n_warmup: int, n_repeat: int, k: int, 
                       b: int, h: int, n: int, mmbpr: int) -> Tuple[float, float]:
    """Benchmark reference implementation."""
    # Warmup
    for i in range(n_warmup):
        query, maxblocks, minblocks, metadata_block_tables, seq_lens = input_sets[i]
        ref_quest_block_select_paged(query, maxblocks, minblocks, metadata_block_tables, seq_lens, k)
    torch.npu.synchronize()
    
    # Measurement
    start = torch.npu.Event(enable_timing=True)
    end = torch.npu.Event(enable_timing=True)
    
    start.record()
    for repetition in range(n_warmup, n_warmup + n_repeat):
        query, maxblocks, minblocks, metadata_block_tables, seq_lens = input_sets[repetition]
        ref_quest_block_select_paged(query, maxblocks, minblocks, metadata_block_tables, seq_lens, k)
    end.record()
    torch.npu.synchronize()
    
    duration = start.elapsed_time(end) / n_repeat * 1000  # ms to μs
    total_bytes_moved = bytes_moved_paged_select(b, h, n, BLOCK_SIZE, HEAD_DIM, mmbpr, k, input_sets[0][4])
    bandwidth = total_bytes_moved / duration / 1e6  # TB/s
    
    return duration, bandwidth


# --------------------------------------------------------------------------- #
#  Output functions
# --------------------------------------------------------------------------- #
def log_header(custom_kernel: Callable, dtype: torch.dtype):
    """Log benchmark header."""
    logger.info(f"  custom_kernel={custom_kernel.__name__} {dtype=}\n  {BLOCK_SIZE=}  {HEAD_DIM=}  "
                f"{SAME_SEQ_LEN_ALL_REQS=}")
    logger.info(f"{'H':>3} {'N':>3} {'B':>3} {'MMBPR':>6} {'Max_seq_len':>12} {'k':>4} "
                f"{'Outputs_equal':>15} {'Ref_Latency_[usec]':>18} {'Our_Latency_[usec]':>18} "
                f"{'Ref_BW_[TB/sec]':>16} {'Our_BW_[TB/sec]':>16}")


def log_results_row(h: int, n: int, b: int, mmbpr: int, k: int, are_equal: str,
                    ref_time: float, our_time: float, 
                    ref_bw: float, our_bw: float):
    """Log a single row of benchmark results."""
    max_seq_len = mmbpr * BLOCK_SIZE * BLOCK_SIZE
    row = f"{h:>3} {n:>3} {b:>3} {mmbpr:>6} {max_seq_len:>12} {k:>4} {are_equal:>15} "
    
    if ref_time is not None:
        row += f"{ref_time:>18.2f} "
    else:
        row += f"{'N/A':>18} "

    if our_time is not None:
        row += f"{our_time:>18.2f} "
    else:
        row += f"{'N/A':>18} "

    if ref_bw is not None:
        row += f"{ref_bw:>16.3f} "
    else:
        row += f"{'N/A':>16} "

    if our_bw is not None:
        row += f"{our_bw:>16.3f}"
    else:
        row += f"{'N/A':>16}"
    
    logger.info(row)


# --------------------------------------------------------------------------- #
#  Main benchmark function
# --------------------------------------------------------------------------- #
def benchmark_quest_block_select_paged(custom_kernel: Callable, dtype: torch.dtype):
    """
    Benchmark the custom kernel against reference kernel across a range of input shapes. Show latency and bandwidth.
    :param custom_kernel:     a callable kernel: either quest_block_select_paged or quest_block_select_paged_in_out
    """
    run_our = True
    run_ref = True
    n_repeat = 10
    n_warmup = 1
    
    if not run_our and not run_ref:
        logger.info("Nothing to run, must set run_our=True or run_ref=True")
        return

    log_header(custom_kernel, dtype)
    
    batch_size_vals, num_heads_vals, num_kv_heads_vals, mmbpr_vals, k_vals = get_benchmark_input_sets(custom_kernel)

    for b, h, n, mmbpr, k in itertools.product(batch_size_vals, num_heads_vals, num_kv_heads_vals, mmbpr_vals, k_vals):
        
        # Check correctness
        are_equal = "N/A"
        if run_our and run_ref:
            are_equal = check_correctness(custom_kernel, b, h, n, mmbpr, k, dtype)
        
        # Initialize results
        our_time = None
        our_bw = None
        ref_time = None
        ref_bw = None
        
        # Our implementation benchmark
        if run_our:
            if custom_kernel == quest_block_select_paged_in_out:
                our_ids = torch.zeros((b, n, k), dtype=torch.int32, device="npu:0") 
            else: 
                our_ids = None
            input_sets = generate_input_sets(n_warmup, n_repeat, b, h, n, mmbpr, dtype)
            our_time, our_bw = (
                benchmark_implementation(custom_kernel, input_sets, n_warmup, n_repeat, k, b, h, n, mmbpr, our_ids)
            )

        # Reference implementation benchmark
        if run_ref:
            input_sets = generate_input_sets(n_warmup, n_repeat, b, h, n, mmbpr, dtype)
            ref_time, ref_bw = benchmark_reference(input_sets, n_warmup, n_repeat, k, b, h, n, mmbpr)
        
        # Log results
        log_results_row(h, n, b, mmbpr, k, are_equal, ref_time, our_time, ref_bw, our_bw)



# --------------------------------------------------------------------------- #
if __name__ == "__main__":
    benchmark_quest_block_select_paged(custom_kernel=quest_block_select_paged, dtype=torch.float16)
    benchmark_quest_block_select_paged(custom_kernel=quest_block_select_paged_in_out, dtype=torch.float16)
    benchmark_quest_block_select_paged(custom_kernel=quest_block_select_paged, dtype=torch.bfloat16)
    benchmark_quest_block_select_paged(custom_kernel=quest_block_select_paged_in_out, dtype=torch.bfloat16)
