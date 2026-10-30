# -*- coding: utf-8 -*-
# Copyright (c) 2025 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
"""
Benchmark driver for quest_prefill_metadata kernel (Ascend-C vs reference).

Measures:
*  latency (μs) - NPU timer
*  effective bandwidth (TB/s) - bytes moved / time
*  correctness comparison with reference implementation
"""
import itertools
import logging
import torch
import torch_npu

from select_attn_ops import quest_prefill_metadata
from ref_quest_prefill_metadata import ref_quest_prefill_metadata
from gen_data_quest_prefill_metadata import gen_quest_prefill_inputs, ceil_div, compare_tensors


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
def bytes_moved_prefill(batch_size: int, num_kv_heads: int, block_size: int, head_dim: int,
                        mkbpr: int, mmbpr: int, 
                        seq_lens: torch.Tensor) -> int:
    """
    Global-memory traffic (read + write) for quest_prefill_metadata.

    Arguments:
        batch_size - batch size
        num_kv_heads - number of KV heads
        block_size - number of tokens per block (equal for metadata block and 
                     kv-cache block)
        head_dim - head dimension
        mkbpr - maximum number of KV-cache blocks per request in a batch
        mmbpr - maximum number of metadata blocks per request in a batch

    Reads
    -----
    seq_lens              :  batch_size * 4                                              (int32)
    k_cache               :  num_effective_kv_blocks * block_size * num_kv_heads * head_dim * 2   (fp16)
    block_tables          :  num_effective_kv_blocks * 4                        (int32)
    metadata_block_tables :  num_effective_kv_blocks * 4                        (int32)

    Writes
    ------
    maxblocks      :  num_effective_metadata_blocks * block_size * num_kv_heads * head_dim * 2    (fp16)
    minblocks      :  num_effective_metadata_blocks * block_size * num_kv_heads * head_dim * 2    (fp16)

    Where num_effective_kv_blocks is the total number of K blocks effectively being read 
    by the entire job (all requests) and num_effective_metadata_blocks is the effective 
    number of metadata blocks being produced and written back to GM by the entire kernel
    """

    # total number of K blocks that will be read in the entire job (all requests)
    num_effective_kv_blocks = torch.sum(ceil_div(seq_lens, block_size)).item()
    
    # total number of K blocks that will be read in the entire job (all requests)
    toks_per_meta_block = block_size * block_size
    num_effective_metadata_blocks = torch.sum(ceil_div(seq_lens, toks_per_meta_block)).item()

    read_k = num_effective_kv_blocks * block_size * num_kv_heads * head_dim * 2
    read_tbl = num_effective_kv_blocks * 4
    write_meta = 2 * (num_effective_metadata_blocks * block_size * num_kv_heads * head_dim * 2)
    return read_k + read_tbl + write_meta


# --------------------------------------------------------------------------- #
#  Configuration and setup
# --------------------------------------------------------------------------- #
def get_benchmark_configurations():
    """Get benchmark configuration parameters."""
    batch_size_vals = [10, 20, 24, 32]
    num_kv_heads_vals = [8]
    mkbpr_vals = [63, 80, 94, 128, 150, 200, 256] # max_kv_blocks_per_request
    return batch_size_vals, num_kv_heads_vals, mkbpr_vals


def generate_input_sets(n_warmup: int, n_repeat: int, b: int, n: int, mkbpr: int, mmbpr: int):
    """Generate multiple input sets for benchmarking."""
    input_sets = []
    for _ in range(n_warmup + n_repeat):
        k_cache, block_tables, seq_lens, metadata_block_tables, max_out, min_out = \
            gen_quest_prefill_inputs(
                b, n, BLOCK_SIZE, HEAD_DIM,
                num_kv_blocks=b * mkbpr,
                num_meta_blocks=b * mmbpr,
                mkbpr=mkbpr,
                mmbpr=mmbpr,
                same_seq_len_all_reqs=SAME_SEQ_LEN_ALL_REQS,
                device="npu:0", 
                dtype=DTYPE)
        input_sets.append((k_cache, block_tables, seq_lens, metadata_block_tables, max_out, min_out))
    return input_sets


# --------------------------------------------------------------------------- #
#  Correctness checking
# --------------------------------------------------------------------------- #
def check_correctness(b: int, n: int, mkbpr: int, mmbpr: int) -> str:
    """Check correctness between custom and reference implementations."""
    # Create fresh output tensors for correctness check
    k_cache, block_tables, seq_lens, metadata_block_tables, max_out_our, min_out_our = gen_quest_prefill_inputs(
        b, n, BLOCK_SIZE, HEAD_DIM,
        num_kv_blocks=b * mkbpr,
        num_meta_blocks=b * mmbpr,
        mkbpr=mkbpr,
        mmbpr=mmbpr,
        same_seq_len_all_reqs=SAME_SEQ_LEN_ALL_REQS,
        device="npu:0", 
        dtype=DTYPE)
    max_out_ref = max_out_our.clone()
    min_out_ref = min_out_our.clone()
    
    # Run both implementations
    ref_quest_prefill_metadata(k_cache, block_tables, seq_lens, metadata_block_tables, max_out_ref, min_out_ref)
    quest_prefill_metadata(k_cache, block_tables, seq_lens, metadata_block_tables, max_out_our, min_out_our)
    
    if (
        compare_tensors(max_out_ref, max_out_our, verbose=False)
        and compare_tensors(min_out_ref, min_out_our, verbose=False)
    ):
        return "yes"
    else:
        return "no"


# --------------------------------------------------------------------------- #
#  Benchmark execution
# --------------------------------------------------------------------------- #
def run_warmup_custom(input_sets: list, n_warmup: int):
    """Run warmup iterations for custom implementation."""
    for i in range(n_warmup):
        k_cache, block_tables, seq_lens, metadata_block_tables, max_out, min_out = input_sets[i]
        quest_prefill_metadata(k_cache, block_tables, seq_lens, metadata_block_tables, max_out, min_out)        
    torch.npu.synchronize()


def run_warmup_reference(input_sets: list, n_warmup: int):
    """Run warmup iterations for reference implementation."""
    for i in range(n_warmup):
        k_cache, block_tables, seq_lens, metadata_block_tables, max_out, min_out = input_sets[i]
        ref_quest_prefill_metadata(k_cache, block_tables, seq_lens, metadata_block_tables, max_out, min_out)        
    torch.npu.synchronize()


def benchmark_custom_implementation(input_sets: list, n_warmup: int, n_repeat: int, 
                                  b: int, n: int, mkbpr: int, mmbpr: int) -> tuple:
    """Benchmark the custom implementation."""
    run_warmup_custom(input_sets, n_warmup)
    
    # Measure time
    start = torch.npu.Event(enable_timing=True)
    end = torch.npu.Event(enable_timing=True)
    
    start.record()
    for i in range(n_warmup, n_warmup + n_repeat):
        k_cache, block_tables, seq_lens, metadata_block_tables, max_out, min_out = input_sets[i]
        quest_prefill_metadata(k_cache, block_tables, seq_lens, metadata_block_tables, max_out, min_out)
    end.record()
    torch.npu.synchronize()
    
    duration = start.elapsed_time(end) / n_repeat * 1000  # ms to μs
    total_bytes = bytes_moved_prefill(b, n, BLOCK_SIZE, HEAD_DIM, mkbpr, mmbpr, input_sets[0][2])
    bandwidth = total_bytes / duration / 1e6  # TB/s
    
    return duration, bandwidth


def benchmark_reference_implementation(input_sets: list, n_warmup: int, n_repeat: int,
                                     b: int, n: int, mkbpr: int, mmbpr: int) -> tuple:
    """Benchmark the reference implementation."""
    run_warmup_reference(input_sets, n_warmup)
    
    # Measure time
    start = torch.npu.Event(enable_timing=True)
    end = torch.npu.Event(enable_timing=True)
    
    start.record()
    for i in range(n_warmup, n_warmup + n_repeat):
        k_cache, block_tables, seq_lens, metadata_block_tables, max_out, min_out = input_sets[i]
        ref_quest_prefill_metadata(k_cache, block_tables, seq_lens, metadata_block_tables, max_out, min_out)
    end.record()
    torch.npu.synchronize()
    
    duration = start.elapsed_time(end) / n_repeat * 1000  # ms to μs
    total_bytes = bytes_moved_prefill(b, n, BLOCK_SIZE, HEAD_DIM, mkbpr, mmbpr, input_sets[0][2])
    bandwidth = total_bytes / duration / 1e6  # TB/s
    
    return duration, bandwidth


# --------------------------------------------------------------------------- #
#  Output functions
# --------------------------------------------------------------------------- #
def log_header():
    """Log benchmark header."""
    logger.info(f"  {DTYPE=}  {BLOCK_SIZE=}  {HEAD_DIM=}  {SAME_SEQ_LEN_ALL_REQS=}")
    logger.info(f"{'N':>3} {'B':>3} {'Seq_len':>10} {'Outputs_equal':>16} "
                f"{'Ref_Latency_[usec]':>19} {'Our_Latency_[usec]':>19} "
                f"{'Ref_BW_[TB/sec]':>17} {'Our_BW_[TB/sec]':>17}")


def log_results_row(n: int, b: int, seq_len: int, are_equal: str, ref_duration: float, 
                   our_duration: float, ref_bw: float, our_bw: float):
    """Log a single row of benchmark results."""
    result_line = f"{n:>3} {b:>3} {seq_len:>10} {are_equal:>16} "
    
    if ref_duration is not None: 
        result_line += f"{ref_duration:>19.2f} "
    else: 
        result_line += f"{'N/A':>19} "
    
    if our_duration is not None: 
        result_line += f"{our_duration:>19.2f} "
    else: 
        result_line += f"{'N/A':>19} "
    
    if ref_bw is not None: 
        result_line += f"{ref_bw:>17.3f} "
    else: 
        result_line += f"{'N/A':>17} "
    
    if our_bw is not None: 
        result_line += f"{our_bw:>17.3f}"
    else: 
        result_line += f"{'N/A':>17}"
    
    logger.info(result_line)


# --------------------------------------------------------------------------- #
#  Main benchmark function
# --------------------------------------------------------------------------- #
def benchmark_quest_prefill():
    run_our = True
    run_ref = True
    n_repeat = 6
    n_warmup = 1
    
    batch_size_vals, num_kv_heads_vals, mkbpr_vals = get_benchmark_configurations()
    
    if not run_our and not run_ref:
        logger.info("Nothing to run, must set run_our=True or run_ref=True")
        return

    log_header()

    for n, b, mkbpr in itertools.product(num_kv_heads_vals, batch_size_vals, mkbpr_vals):
        seq_len = mkbpr * BLOCK_SIZE
        mmbpr = ceil_div(mkbpr, BLOCK_SIZE)
        
        # Check correctness
        are_equal = "N/A"
        if run_our and run_ref:
            are_equal = check_correctness(b, n, mkbpr, mmbpr)

        # Initialize results
        our_time = None
        our_bw = None
        ref_time = None
        ref_bw = None

        # Our implementation benchmark
        if run_our:
            input_sets = generate_input_sets(n_warmup, n_repeat, b, n, mkbpr, mmbpr)
            our_time, our_bw = benchmark_custom_implementation(input_sets, n_warmup, n_repeat, b, n, mkbpr, mmbpr)

        # Reference implementation benchmark
        if run_ref:
            input_sets = generate_input_sets(n_warmup, n_repeat, b, n, mkbpr, mmbpr)
            ref_time, ref_bw = benchmark_reference_implementation(input_sets, n_warmup, n_repeat, b, n, mkbpr, mmbpr)
        
        # Log results
        log_results_row(n, b, seq_len, are_equal, ref_time, our_time, ref_bw, our_bw)


# --------------------------------------------------------------------------- #
if __name__ == "__main__":
    benchmark_quest_prefill()
