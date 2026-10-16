---
schema_version: okf.v1
kind: operator_optimization
type: optimization_runbook
source_family: runbooks
category: compute-pipeline
dimension: 2.3
title: 'Scheduling Optimization: lookback algorithm'
description: 'gather_scatter 场景下，Scheduling Optimization: lookback algorithm，收益 25%'
tags: [多核并行与负载均衡, indexing-operations, gather_scatter]
confidence: verified
status: verified
created_at: 2026-08-21T00:00:00Z
updated_at: 2026-08-21T00:00:00Z
---

## Scheduling Optimization: lookback algorithm

【优化维度】：多核并行与负载均衡
【泛化层级】：条件性（迁移条件：硬件 910b2, 910b3, 91093, 91095；失效边界：lookback 自旋等待在块间负载严重不均时消耗 cycle（备注自述）；volatile/.cg 同步语义是正确性前提，省略会在宽松内存一致性硬件上死锁或读脏数据；n_elements < 1024 时单块即可处理，无需 lookback）

### 原则
- **lookback algorithm**：Implements a lookback mechanism to calculate global offsets. Each block computes local histograms and then iteratively queries previous blocks' status in a lookback buffer. This eliminates the need for a separate global prefix sum kernel, reducing memory traffic and kernel launch latency.
- **volatile loading**：Uses `volatile=True` in the synchronization loop (`while partial_prefix == 0`). This forces the load to bypass the L1 cache and fetch the most recent data written by other blocks, ensuring correct synchronization and preventing deadlocks in the spin-wait logic.
- **cache streaming**：Applies `cache_modifier=".cg"` (Cache Global) when writing to the lookback buffer. This ensures that synchronization data is pushed to the L2 cache or main memory immediately, making it visible to other thread blocks performing the lookback.
- **data reuse**：Loads `key_data` and `value_data` once per block and iterates over `bins_segment` (multiple bins) within the same kernel execution. This maximizes data reuse of the input tensors, significantly reducing global memory bandwidth requirements compared to launching a kernel per bin.

### 代码/骨架出处
- `TritonPath`:
```python
raw_experiences/extracted_kernels/137-radix_sortbykey_scatter_kernel.py
```

**坏实践（反例）**：全局前缀和单独起一个 kernel，或每个 bin 起一个 kernel → 额外 launch + 输入重复读 → ✅ 正解即本条：块内直方图 + lookback 缓冲原地算全局偏移（volatile 读 + `.cg` 写保证可见性），单 kernel 处理全部 bin。

**已知实例**：gather_scatter §raw (`experiences/gather_scatter/exp_radix_sort_scatter_001.yaml`)
- 声称性能收益：25%
- 备注：The lookback loop involves a spin-wait which can consume cycles if blocks are heavily imbalanced. The `volatile` attribute is essential for correctness on NPUs/GPUs with relaxed memory consistency.

<!-- okf:related:start -->

# 相关

- 开发指南: [Gather/Scatter 算子优化](../../reference/op-class-gather_scatter.md) — 主题相关：共享 gather_scatter
- 相关主题: [Memory Access Optimization: vectorized 3d memory access](../memory-sram/exp_cache_store_001.md) — 主题相关：共享 indexing-operations
- 相关主题: [Memory Access Optimization: 2D Tiling Strategy](../memory-sram/exp_embedding_001.md) — 主题相关：共享 indexing-operations
- 相关主题: [Memory Access Optimization: vectorized indexing](../memory-sram/exp_gather_001.md) — 主题相关：共享 gather_scatter, indexing-operations
- 相关主题: [Memory Access Optimization: large block processing](../memory-sram/exp_gather_flat_001.md) — 主题相关：共享 gather_scatter, indexing-operations
- 相关主题: [Memory Access Optimization: block pointer utilization](../memory-sram/exp_idx_copy_001.md) — 主题相关：共享 indexing-operations
- 相关主题: [Memory Access Optimization: redundant access elimination](../memory-sram/exp_masked_fill_001.md) — 主题相关：共享 indexing-operations
- 相关主题: [Memory Access Optimization: register buffering](../memory-sram/exp_moe_binned_scatter_001.md) — 主题相关：共享 gather_scatter, indexing-operations
- 相关主题: [Memory Access Optimization: vectorized metadata loading](../memory-sram/exp_moe_gather_001.md) — 主题相关：共享 gather_scatter, indexing-operations
- 相关主题: [Memory Access Optimization: software pipelining](../memory-sram/exp_moe_scatter_001.md) — 主题相关：共享 gather_scatter, indexing-operations
- 相关主题: [Memory Access Optimization: buffered writes](../memory-sram/exp_padded_gather_001.md) — 主题相关：共享 gather_scatter, indexing-operations
- 相关主题: [Memory Access Optimization: head vectorization](../memory-sram/exp_paged_kv_cache_001.md) — 主题相关：共享 indexing-operations
- 相关主题: [Memory Access Optimization: vectorized memory access](../memory-sram/exp_scatter_001.md) — 主题相关：共享 gather_scatter, indexing-operations
- 相关主题: [Memory Access Optimization: sorted bin grouping with padded alignment](../memory-sram/exp_scatter_moe_padded_001.md) — 主题相关：共享 gather_scatter, indexing-operations
- 相关主题: [Memory Access Optimization: binning strategy](../memory-sram/exp_scatter_npu_001.md) — 主题相关：共享 gather_scatter, indexing-operations
- 相关主题: [Memory Access Optimization: padded binning for grouped scatter reduction](../memory-sram/exp_scatter_wgrad_moe_001.md) — 主题相关：共享 gather_scatter, indexing-operations
- 相关主题: [Memory Access Optimization: intermediate data residency](../memory-sram/exp_spec_dec_reject_001.md) — 主题相关：共享 indexing-operations
- 相关主题: [Memory Access Optimization: vectorized comparison](../memory-sram/exp_spec_dec_verify_001.md) — 主题相关：共享 indexing-operations
- 相关主题: [Memory Access Optimization: block ptr utilization](../memory-sram/exp_topk_pad_001.md) — 主题相关：共享 indexing-operations
- 相关主题: [并行轴选择必须与连续访问对齐，而非与最大 numel 对齐](../memory-sram/parallel-axis-align.md) — 主题相关：共享 gather_scatter
- 相关主题: [动态长度标量更新 kernel：do_not_specialize 防多版本编译，编译时间降 >80%](../meta-skills/exp_bonus_renew_001.md) — 主题相关：共享 indexing-operations
- 相关主题: [Scheduling Optimization: 2D Block Tiling](exp_embedding_backward_001.md) — 主题相关：共享 indexing-operations, 多核并行与负载均衡
- 相关主题: [Scheduling Optimization: compilation specialization control](exp_expand_serial_001.md) — 主题相关：共享 indexing-operations, 多核并行与负载均衡
- 相关主题: [Compute Optimization: vectorized binary search](exp_multinomial_001.md) — 主题相关：共享 indexing-operations
- 相关主题: [Compute Optimization: vectorized load](exp_reject_sampler_001.md) — 主题相关：共享 indexing-operations
- 相关主题: [Scheduling Optimization: vectorized global load](exp_rejection_sample_001.md) — 主题相关：共享 indexing-operations, 多核并行与负载均衡
- 相关主题: [Compute Optimization: batched reduction replacement](exp_scatter_wgrad_001.md) — 主题相关：共享 gather_scatter, indexing-operations
- 相关主题: [Scheduling Optimization: block local sorting](exp_shuffle_blockwise_001.md) — 主题相关：共享 indexing-operations, 多核并行与负载均衡
- 相关主题: [Compute Optimization: padding strategy](exp_sort_bitonic_001.md) — 主题相关：共享 indexing-operations
- 相关主题: [Compute Optimization: vectorized probability calculation](exp_top_p_sampling_001.md) — 主题相关：共享 indexing-operations

<!-- okf:related:end -->
