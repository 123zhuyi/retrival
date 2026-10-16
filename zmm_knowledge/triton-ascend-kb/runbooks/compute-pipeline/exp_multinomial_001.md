---
schema_version: okf.v1
kind: operator_optimization
type: optimization_runbook
source_family: runbooks
category: compute-pipeline
dimension: 2.1
title: 'Compute Optimization: vectorized binary search'
description: 'misc 场景下，Compute Optimization: vectorized binary search，收益 35%'
tags: [AICore 单元利用/Scalar 削减, indexing-operations, misc]
confidence: verified
status: verified
created_at: 2026-08-21T00:00:00Z
updated_at: 2026-08-21T00:00:00Z
---

## Compute Optimization: vectorized binary search

【优化维度】：AICore 单元利用/Scalar 削减
【泛化层级】：条件性（迁移条件：硬件 910b2, 910b3, 91093, 91095；失效边界：scattered load 的效率依赖 mid 的局部分布，最坏情况退化为离散访存；rv 的 0.0001/0.9999 扰动 clamp 是边界正确性前提，不可省略；K（词表）< 128 时搜索空间小，向量化收益有限）

### 原则
- **vectorized binary search**：The kernel implements a vectorized binary search where the loop variables (`start`, `end`, `mid`) are vectors of size `NBLOCK`. This allows the hardware to execute the search logic for multiple samples in parallel (SIMD), significantly increasing instruction throughput compared to a scalar per-thread search.
- **dynamic shape specialization**：The use of `do_not_specialize` for arguments `K` and `N` prevents the JIT compiler from generating a new kernel instance for every variation in batch size or vocabulary size. This is critical for LLM inference workloads where sequence lengths vary, reducing compilation overhead and improving launch latency.
- **memory coalescing**：By calculating `mid` as a vector and performing `tl.load(cdf_ptr + mid, mask=n < N)`, the kernel loads scattered CDF values. While the access pattern depends on the current `mid`, the vectorized load ensures that the memory controller is utilized efficiently by servicing multiple requests in a single transaction where possible, or hiding latency better than scalar loads.
- **edge case stabilization**：The random variable `rv` is perturbed by `0.0001` and clamped to `0.9999` before the search. This avoids out-of-bounds access and incorrect index selection when the random number is exactly 0.0 or 1.0 (or when CDF buckets have zero probability), ensuring robustness without adding complex branching logic inside the hot loop.

### 代码/骨架出处
- `TritonPath`:
```python
raw_experiences/extracted_kernels/139-multinomial_with_replacement.py
```

**坏实践（反例）**：每个样本用标量循环变量做二分查找（start/end/mid 都是标量）→ SIMD 单元闲置，大词表下搜索完全串行 → ✅ 正解即本条：循环变量全部向量化（NBLOCK 宽度），多样本并行二分。

**已知实例**：misc §raw (`experiences/misc/exp_multinomial_001.yaml`)
- 声称性能收益：35%
- 备注：This implementation is highly efficient for sampling steps in generative models (e.g., LLMs) where K (vocabulary size) is large and N (batch size) varies.

<!-- okf:related:start -->

# 相关

- 实践案例: [MoeComputeExpertTokens (MoE expert token count via binary search) on Ascend NPU](../../ops/moe-compute-expert-tokens.md) — 同属优化知识，主题相关
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
- 相关主题: [Compute Optimization: block ptr tiling](exp_block_inverse_001.md) — 主题相关：共享 aicore 单元利用/scalar 削减, misc
- 相关主题: [Compute Optimization: gemm based histogram](exp_digit_hist_001.md) — 主题相关：共享 aicore 单元利用/scalar 削减, misc
- 相关主题: [Scheduling Optimization: 2D Block Tiling](exp_embedding_backward_001.md) — 主题相关：共享 indexing-operations
- 相关主题: [Scheduling Optimization: compilation specialization control](exp_expand_serial_001.md) — 主题相关：共享 indexing-operations, misc
- 相关主题: [Compute Optimization: operator fusion](exp_fused_index_001.md) — 主题相关：共享 aicore 单元利用/scalar 削减, misc
- 相关主题: [Scheduling Optimization: deep operator fusion](exp_fused_transformer_block_001.md) — 主题相关：共享 misc
- 相关主题: [Compute Optimization: numerical stability](exp_grpo_loss_001.md) — 主题相关：共享 aicore 单元利用/scalar 削减, misc
- 相关主题: [Compute Optimization: register accumulation](exp_hist_count_001.md) — 主题相关：共享 aicore 单元利用/scalar 削减, misc
- 相关主题: [Compute Optimization: Operator Fusion](exp_mhc_post_fusion_001.md) — 主题相关：共享 aicore 单元利用/scalar 削减, misc
- 相关主题: [Compute Optimization: compile time dispatch](exp_radix_convert_001.md) — 主题相关：共享 aicore 单元利用/scalar 削减, misc
- 相关主题: [Scheduling Optimization: lookback algorithm](exp_radix_sort_scatter_001.md) — 主题相关：共享 indexing-operations
- 相关主题: [Compute Optimization: high precision accumulation](exp_recompute_w_u_001.md) — 主题相关：共享 aicore 单元利用/scalar 削减, misc
- 相关主题: [Compute Optimization: vectorized load](exp_reject_sampler_001.md) — 主题相关：共享 aicore 单元利用/scalar 削减, indexing-operations, misc
- 相关主题: [Scheduling Optimization: vectorized global load](exp_rejection_sample_001.md) — 主题相关：共享 indexing-operations, misc
- 相关主题: [Compute Optimization: batched reduction replacement](exp_scatter_wgrad_001.md) — 主题相关：共享 aicore 单元利用/scalar 削减, indexing-operations
- 相关主题: [Scheduling Optimization: block local sorting](exp_shuffle_blockwise_001.md) — 主题相关：共享 indexing-operations, misc
- 相关主题: [Compute Optimization: padding strategy](exp_sort_bitonic_001.md) — 主题相关：共享 aicore 单元利用/scalar 削减, indexing-operations, misc
- 相关主题: [Compute Optimization: scalar loop vectorization](exp_spec_decode_verify_scalar.md) — 主题相关：共享 aicore 单元利用/scalar 削减, misc
- 相关主题: [Compute Optimization: vectorized probability calculation](exp_top_p_sampling_001.md) — 主题相关：共享 aicore 单元利用/scalar 削减, indexing-operations, misc
- 相关主题: [Compute Optimization: kernel fusion](exp_topp_filter_001.md) — 主题相关：共享 aicore 单元利用/scalar 削减, misc
- 相关主题: [Compute Optimization: micro kernel tiling](exp_tril_solve_001.md) — 主题相关：共享 aicore 单元利用/scalar 削减, misc

<!-- okf:related:end -->
