---
schema_version: okf.v1
kind: operator_optimization
type: optimization_runbook
source_family: runbooks
category: compute-pipeline
dimension: 2.3
title: 'Scheduling Optimization: vectorized global load'
description: 'misc 场景下，Scheduling Optimization: vectorized global load，收益 40%'
tags: [多核并行与负载均衡, indexing-operations, misc]
confidence: verified
status: verified
created_at: 2026-08-21T00:00:00Z
updated_at: 2026-08-21T00:00:00Z
---

## Scheduling Optimization: vectorized global load

【优化维度】：多核并行与负载均衡
【泛化层级】：条件性（迁移条件：硬件 910b2, 910b3；失效边界：最适合 max_spec_len 4~16 的典型投机采样（备注自述）；拒绝的串行依赖用 bitmask 传播表达，向量化宽度（4/8）需按寄存器预算调；is_greedy 路径的特化要求该标志编译期/启动期可知）

### 原则
- **vectorized global load**：The kernel loads `draft_token_ids` and `target_argmax_ids` in chunks of `blockSize` (vectors) from global memory. This aggregates memory requests, significantly reducing global memory access overhead compared to scalar loading inside a loop.
- **in block serial processing**：Utilizes `tl.range` and `tl.get_element` to iterate over the loaded vector elements sequentially. This design allows for complex control flow (conditional checks and external function calls) that are difficult to express in a purely vectorized SIMD manner, specifically for the rejection sampling verification step.
- **strided output storage**：Stores the accepted `target_argmax_id` into the output buffer with a stride of 2 (`offset * 2`). This prepares the memory layout for interleaved data (e.g., storing accepted tokens alongside bonus tokens or metadata) without requiring a separate data rearrangement kernel.
- **vectorized inner loop**：Replace the scalar loop `for pos in range(num_draft_tokens)` with a vectorized loop that processes multiple tokens (e.g., 4 or 8) per iteration. This increases instruction-level parallelism and utilizes memory bandwidth more efficiently by loading chunks of `draft_token_ids`, `draft_probs`, and `target_probs` simultaneously.
- **mask based rejection propagation**：Implement a bitmask mechanism to handle the sequential dependency of rejection (stop processing once a token is rejected) within the vectorized loop. This allows the kernel to maintain correctness while still performing parallel comparisons and loads for the active (non-rejected) prefix of the sequence.
- **greedy path specialization**：Explicitly handle the `is_greedy` condition by masking out the expensive probability loading and division logic for greedy requests. This reduces register pressure and avoids unnecessary memory accesses for the greedy decoding path.

### 代码/骨架出处
- `TritonPath`:
```python
raw_experiences/extracted_kernels/168-rejection_random_sample_kernel.py
```

**坏实践（反例）**：`for pos in range(num_draft_tokens)` 标量循环逐 token load → 循环控制与串行访存主导开销 → ✅ 正解即本条：块式向量 load + 向量化内层循环（每迭代 4-8 token）+ bitmask 表达拒绝传播 + greedy 路径特化。

**已知实例**：misc §raw (`experiences/misc/exp_rejection_sample_001.yaml`)
- 声称性能收益：40%
- 备注：This optimization is particularly effective for typical speculative decoding scenarios where the speculation length (max_spec_len) is between 4 and 16. It reduces the overhead of loop control and serial memory accesses.

<!-- okf:related:start -->

# 相关

- 相关主题: [Memory Access Optimization: vectorized 3d memory access](../memory-sram/exp_cache_store_001.md) — 主题相关：共享 indexing-operations, misc
- 相关主题: [Memory Access Optimization: 2D Tiling Strategy](../memory-sram/exp_embedding_001.md) — 主题相关：共享 indexing-operations
- 相关主题: [Memory Access Optimization: operator fusion](../memory-sram/exp_fuse_tlx_001.md) — 主题相关：共享 misc
- 相关主题: [Memory Access Optimization: vectorized indexing](../memory-sram/exp_gather_001.md) — 主题相关：共享 indexing-operations
- 相关主题: [Memory Access Optimization: large block processing](../memory-sram/exp_gather_flat_001.md) — 主题相关：共享 indexing-operations
- 相关主题: [Memory Access Optimization: block pointer utilization](../memory-sram/exp_idx_copy_001.md) — 主题相关：共享 indexing-operations
- 相关主题: [Memory Access Optimization: redundant access elimination](../memory-sram/exp_masked_fill_001.md) — 主题相关：共享 indexing-operations, misc
- 相关主题: [Memory Access Optimization: register buffering](../memory-sram/exp_moe_binned_scatter_001.md) — 主题相关：共享 indexing-operations
- 相关主题: [Memory Access Optimization: vectorized metadata loading](../memory-sram/exp_moe_gather_001.md) — 主题相关：共享 indexing-operations
- 相关主题: [Memory Access Optimization: software pipelining](../memory-sram/exp_moe_scatter_001.md) — 主题相关：共享 indexing-operations
- 相关主题: [Memory Access Optimization: buffered writes](../memory-sram/exp_padded_gather_001.md) — 主题相关：共享 indexing-operations
- 相关主题: [Memory Access Optimization: head vectorization](../memory-sram/exp_paged_kv_cache_001.md) — 主题相关：共享 indexing-operations
- 相关主题: [Memory Access Optimization: vectorized memory access](../memory-sram/exp_scatter_001.md) — 主题相关：共享 indexing-operations
- 相关主题: [Memory Access Optimization: sorted bin grouping with padded alignment](../memory-sram/exp_scatter_moe_padded_001.md) — 主题相关：共享 indexing-operations
- 相关主题: [Memory Access Optimization: binning strategy](../memory-sram/exp_scatter_npu_001.md) — 主题相关：共享 indexing-operations
- 相关主题: [Memory Access Optimization: padded binning for grouped scatter reduction](../memory-sram/exp_scatter_wgrad_moe_001.md) — 主题相关：共享 indexing-operations
- 相关主题: [Memory Access Optimization: intermediate data residency](../memory-sram/exp_spec_dec_reject_001.md) — 主题相关：共享 indexing-operations, misc
- 相关主题: [Memory Access Optimization: vectorized comparison](../memory-sram/exp_spec_dec_verify_001.md) — 主题相关：共享 indexing-operations, misc
- 相关主题: [Memory Access Optimization: block ptr utilization](../memory-sram/exp_topk_pad_001.md) — 主题相关：共享 indexing-operations, misc
- 相关主题: [动态长度标量更新 kernel：do_not_specialize 防多版本编译，编译时间降 >80%](../meta-skills/exp_bonus_renew_001.md) — 主题相关：共享 indexing-operations, misc
- 相关主题: [Scheduling Optimization: grid stride loop](exp_element_mul_001.md) — 主题相关：共享 misc, 多核并行与负载均衡
- 相关主题: [Scheduling Optimization: 2D Block Tiling](exp_embedding_backward_001.md) — 主题相关：共享 indexing-operations, 多核并行与负载均衡
- 相关主题: [Scheduling Optimization: compilation specialization control](exp_expand_serial_001.md) — 主题相关：共享 indexing-operations, misc, 多核并行与负载均衡
- 相关主题: [Scheduling Optimization: deep operator fusion](exp_fused_transformer_block_001.md) — 主题相关：共享 misc, 多核并行与负载均衡
- 相关主题: [Scheduling Optimization: diagonal traversal pattern](exp_launch_diagonal_001.md) — 主题相关：共享 misc, 多核并行与负载均衡
- 相关主题: [Scheduling Optimization: persistent kernel grid stride loop](exp_linear_persistent_001.md) — 主题相关：共享 misc, 多核并行与负载均衡
- 相关主题: [Compute Optimization: vectorized binary search](exp_multinomial_001.md) — 主题相关：共享 indexing-operations, misc
- 相关主题: [Compute Optimization: compile time dispatch](exp_radix_convert_001.md) — 主题相关：共享 misc
- 相关主题: [Scheduling Optimization: lookback algorithm](exp_radix_sort_scatter_001.md) — 主题相关：共享 indexing-operations, 多核并行与负载均衡
- 相关主题: [Compute Optimization: vectorized load](exp_reject_sampler_001.md) — 主题相关：共享 indexing-operations, misc
- 相关主题: [Compute Optimization: batched reduction replacement](exp_scatter_wgrad_001.md) — 主题相关：共享 indexing-operations
- 相关主题: [Scheduling Optimization: block local sorting](exp_shuffle_blockwise_001.md) — 主题相关：共享 indexing-operations, misc, 多核并行与负载均衡
- 相关主题: [Compute Optimization: padding strategy](exp_sort_bitonic_001.md) — 主题相关：共享 indexing-operations, misc
- 相关主题: [Compute Optimization: scalar loop vectorization](exp_spec_decode_verify_scalar.md) — 主题相关：共享 misc
- 相关主题: [Compute Optimization: vectorized probability calculation](exp_top_p_sampling_001.md) — 主题相关：共享 indexing-operations, misc

<!-- okf:related:end -->
