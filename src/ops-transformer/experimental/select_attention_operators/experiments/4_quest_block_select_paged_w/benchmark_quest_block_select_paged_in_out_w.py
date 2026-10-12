# -*- coding: utf-8 -*-
# Copyright (c) 2025 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.

"""

Benchmark driver for quest_block_select_paged_in_out_w kernel.

Measures:
*  latency (usec) - NPU timer
*  effective bandwidth (TB/s) - bytes moved / time
*  correctness comparison with reference implementation
"""
import logging
import itertools
import torch
import torch_npu

from select_attn_ops import quest_block_select_paged_in_out_w
from ref_quest_block_select_paged_w import ref_quest_block_select_paged_w
from gen_data_quest_block_select_paged_w import gen_quest_paged_w_inputs, ceil_div, compare_indices


# Configure logging
logging.basicConfig(level=logging.INFO, format='%(message)s')
logger = logging.getLogger(__name__)

torch.npu.set_device("npu:0")
DTYPE = torch.bfloat16
BLOCK_SIZE = 128
HEAD_DIM = 128
SAME_SEQ_LEN_ALL_REQS = True # to set equally long input length and avoid unknown actual size


# --------------------------------------------------------------------------- #
#  bytes-moved calculator
# --------------------------------------------------------------------------- #
def bytes_moved_paged_select(batch_size: int, num_heads: int, num_kv_heads: int, block_size: int, head_dim: int,
                            mmbpr: int, k: int, seq_lens: torch.Tensor) -> int:
    """
    Global-memory traffic (read + write) for quest_block_select_paged_in_out_w.

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
#  Configuration and setup
# --------------------------------------------------------------------------- #
def get_benchmark_configurations() -> tuple:
    """Get benchmark configuration parameters."""
    batch_size_vals = [10, 20, 24, 32]
    num_heads_vals = [32]
    num_kv_heads_vals = [8]
    mmbpr_vals = [1, 2, 4, 6] if torch.bfloat16 else [1, 2, 4, 8, 16]
    k_vals = [8, 16, 24, 32]
    return batch_size_vals, num_heads_vals, num_kv_heads_vals, mmbpr_vals, k_vals


def generate_input_sets(n_warmup: int, n_repeat: int, b: int, h: int, n: int, mmbpr: int) -> list:
    """Generate multiple input sets for benchmarking."""
    input_sets = []
    for _ in range(n_warmup + n_repeat):
        query, maxblocks, minblocks, metadata_block_tables, seq_lens, tokens_since_metadata_update = \
            gen_quest_paged_w_inputs(
                b, h, n, BLOCK_SIZE, HEAD_DIM, num_meta_blocks=b * mmbpr,
                mmbpr=mmbpr, same_seq_len_all_reqs=SAME_SEQ_LEN_ALL_REQS,
                device="npu:0", dtype=DTYPE)
        input_sets.append((query, maxblocks, minblocks, 
                           metadata_block_tables, seq_lens, 
                           tokens_since_metadata_update))
    return input_sets


# --------------------------------------------------------------------------- #
#  Correctness checking
# --------------------------------------------------------------------------- #
def check_correctness(b: int, h: int, n: int, mmbpr: int, k: int) -> str:
    """Check correctness between custom and reference implementations."""
    query, maxblocks, minblocks, metadata_block_tables, seq_lens, tokens_since_metadata_update = (
        gen_quest_paged_w_inputs(
            b, h, n, BLOCK_SIZE, HEAD_DIM,
            num_meta_blocks=b * mmbpr,
            mmbpr=mmbpr,
            same_seq_len_all_reqs=SAME_SEQ_LEN_ALL_REQS,
            device="npu:0", 
            dtype=DTYPE)
    )
    ref_ids = ref_quest_block_select_paged_w(query, maxblocks, minblocks, metadata_block_tables, 
                                             seq_lens, tokens_since_metadata_update, k)
    our_ids = torch.zeros((b, n, k), dtype=torch.int32, device=query.device)
    quest_block_select_paged_in_out_w(query, maxblocks, minblocks, metadata_block_tables, seq_lens, 
                                      tokens_since_metadata_update, our_ids)
    tol_percentage = 0.02
    are_equal = compare_indices(ref_ids, our_ids, tol_percentage, verbose=False)
    return "yes" if are_equal else "no"


# --------------------------------------------------------------------------- #
#  Warmup functions
# --------------------------------------------------------------------------- #
def run_our_warmup(input_sets: list, n_warmup: int, b: int, n: int, k: int) -> torch.Tensor:
    """Run warmup iterations for our implementation."""
    our_ids = torch.zeros((b, n, k), dtype=torch.int32, device="npu:0")
    for i in range(n_warmup):
        (query, maxblocks, minblocks,
         metadata_block_tables, seq_lens,
         tokens_since_metadata_update,
        ) = input_sets[i]
        quest_block_select_paged_in_out_w(query, maxblocks, minblocks, metadata_block_tables, 
                                          seq_lens, tokens_since_metadata_update, our_ids)
    return our_ids


def run_ref_warmup(input_sets: list, n_warmup: int, k: int):
    """Run warmup iterations for reference implementation."""
    for i in range(n_warmup):
        (
            query, maxblocks, minblocks, metadata_block_tables, seq_lens, tokens_since_metadata_update
         ) = input_sets[i]
        ref_quest_block_select_paged_w(query, maxblocks, minblocks, metadata_block_tables, seq_lens, 
                                       tokens_since_metadata_update, k)


# --------------------------------------------------------------------------- #
#  Benchmark execution
# --------------------------------------------------------------------------- #
def benchmark_our_implementation(input_sets: list, n_warmup: int, n_repeat: int, 
                                b: int, n: int, k: int) -> tuple:
    """Benchmark our implementation and return duration and bandwidth."""
    our_ids = run_our_warmup(input_sets, n_warmup, b, n, k)
    torch.npu.synchronize()

    # Measurement
    start = torch.npu.Event(enable_timing=True)
    end = torch.npu.Event(enable_timing=True)

    start.record()
    for i in range(n_warmup, n_warmup + n_repeat):
        (
            query,
            maxblocks,
            minblocks,
            metadata_block_tables,
            seq_lens,
            tokens_since_metadata_update,
        ) = input_sets[i]
        quest_block_select_paged_in_out_w(query, maxblocks, minblocks, metadata_block_tables, seq_lens, 
                                          tokens_since_metadata_update, our_ids)
    end.record()
    torch.npu.synchronize()
    
    duration = start.elapsed_time(end) / n_repeat * 1000  # ms to μs
    total_bytes = bytes_moved_paged_select(b, 32, n, BLOCK_SIZE, HEAD_DIM, 
                                          input_sets[0][3].shape[1] // BLOCK_SIZE, k, input_sets[0][4])
    bandwidth = total_bytes / duration / 1e6  # TB/s
    
    return duration, bandwidth


def benchmark_ref_implementation(input_sets: list, n_warmup: int, n_repeat: int, k: int) -> tuple:
    """Benchmark reference implementation and return duration and bandwidth."""
    run_ref_warmup(input_sets, n_warmup, k)
    torch.npu.synchronize()
    
    # Measurement
    start = torch.npu.Event(enable_timing=True)
    end = torch.npu.Event(enable_timing=True)
    
    start.record()
    for i in range(n_warmup, n_warmup + n_repeat):
        (
            query, maxblocks, minblocks, metadata_block_tables, seq_lens, tokens_since_metadata_update
         ) = input_sets[i]
        ref_quest_block_select_paged_w(query, maxblocks, minblocks, metadata_block_tables, seq_lens, 
                                       tokens_since_metadata_update, k)
    end.record()
    torch.npu.synchronize()
    
    duration = start.elapsed_time(end) / n_repeat * 1000  # ms to μs
    total_bytes = bytes_moved_paged_select(input_sets[0][0].shape[0], 32, input_sets[0][0].shape[1], 
                                          BLOCK_SIZE, HEAD_DIM, input_sets[0][3].shape[1] // BLOCK_SIZE, 
                                          k, input_sets[0][4])
    bandwidth = total_bytes / duration / 1e6  # TB/s
    
    return duration, bandwidth


# --------------------------------------------------------------------------- #
#  Output functions
# --------------------------------------------------------------------------- #
def log_header():
    """Log benchmark header."""
    logger.info(f"  {DTYPE=}  {BLOCK_SIZE=}  {HEAD_DIM=}  {SAME_SEQ_LEN_ALL_REQS=}")
    logger.info(f"{'H':>3} {'N':>3} {'B':>3} {'MMBPR':>6} {'Max_seq_len':>12} {'k':>4} "
                f"{'Outputs_equal':>16} {'Ref_Latency_[usec]':>19} {'Our_Latency_[usec]':>19} {'Ref_BW_[TB/sec]':>17} "
                f"{'Our_BW_[TB/sec]':>17}")


def log_results_row(h: int, n: int, b: int, mmbpr: int, k: int, are_equal: str,
                    ref_duration: float, our_duration: float, ref_bw: float, our_bw: float):
    """Log a single row of benchmark results."""
    max_seq_len = mmbpr * BLOCK_SIZE * BLOCK_SIZE
    row = f"{h:>3} {n:>3} {b:>3} {mmbpr:>6} {max_seq_len:>12} {k:>4} {are_equal:>16} "
    
    if ref_duration is not None:
        row += f"{ref_duration:>18.2f} "
    else:
        row += f"{'N/A':>18} "

    if our_duration is not None:
        row += f"{our_duration:>19.2f} "
    else:
        row += f"{'N/A':>19} "

    if ref_bw is not None:
        row += f"{ref_bw:>17.3f} "
    else:
        row += f"{'N/A':>17} "

    if our_bw is not None:
        row += f"{our_bw:>17.3f}"
    else:
        row += f"{'N/A':>17}"
    
    logger.info(row)


# --------------------------------------------------------------------------- #
#  Main benchmark function
# --------------------------------------------------------------------------- #
def benchmark_quest_block_select_paged():
    """
    Benchmark the custom kernel against reference kernel across a range of input shapes. 
    Show latency and bandwidth.
    """
    run_our = True
    run_ref = True
    n_repeat = 10
    n_warmup = 1
    
    if not run_our and not run_ref:
        logger.info("Nothing to run, must set run_our=True or run_ref=True")
        return

    log_header()
    
    batch_size_vals, num_heads_vals, num_kv_heads_vals, mmbpr_vals, k_vals = get_benchmark_configurations()

    for b, h, n, mmbpr, k in itertools.product(batch_size_vals, num_heads_vals, num_kv_heads_vals, mmbpr_vals, k_vals):
        
        # Check correctness
        are_equal = "N/A"
        if run_our and run_ref:
            are_equal = check_correctness(b, h, n, mmbpr, k)
        
        # Initialize results
        our_duration = None
        our_bw = None
        ref_duration = None
        ref_bw = None
        
        # Our implementation benchmark
        if run_our:
            input_sets = generate_input_sets(n_warmup, n_repeat, b, h, n, mmbpr)
            our_duration, our_bw = benchmark_our_implementation(input_sets, n_warmup, n_repeat, b, n, k)

        # Reference implementation benchmark
        if run_ref:
            input_sets = generate_input_sets(n_warmup, n_repeat, b, h, n, mmbpr)
            ref_duration, ref_bw = benchmark_ref_implementation(input_sets, n_warmup, n_repeat, k)
        
        # Log results
        log_results_row(h, n, b, mmbpr, k, are_equal, ref_duration, our_duration, ref_bw, our_bw)


# --------------------------------------------------------------------------- #
if __name__ == "__main__":
    benchmark_quest_block_select_paged()
