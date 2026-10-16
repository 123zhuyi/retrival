---
schema_version: okf.v1
kind: operator_optimization
type: optimization_runbook
source_family: runbooks
category: compute-pipeline
dimension: 2.3
title: 'Scheduling Optimization: 2D Block Tiling'
description: 'embedding 场景下，Scheduling Optimization: 2D Block Tiling，收益 35%'
tags: [多核并行与负载均衡, indexing-operations, embedding]
confidence: verified
status: verified
created_at: 2026-08-21T00:00:00Z
updated_at: 2026-08-21T00:00:00Z
---

## Scheduling Optimization: 2D Block Tiling

【优化维度】：多核并行与负载均衡
【泛化层级】：条件性（迁移条件：硬件 910b2, 910b3, 91093, 91095；失效边界：索引重复率高时原子加串行化，收益被竞争侵蚀；embedding_dim < 128 时 2D 块配置需重调；num_stages 流水深度受 UB 容量约束，过深反而溢出）

### 原则
- **2D Block Tiling**：The kernel tiles the gradient matrix into 2D blocks (BLOCK_SIZE_M x BLOCK_SIZE_N) rather than processing rows or columns individually. This improves memory access coalescing for loading `grad_output` and maximizes the utilization of memory bandwidth by processing larger contiguous chunks per transaction.
- **Software Pipelining**：Utilizes `tl.range` with the `num_stages` parameter to implement software pipelining. This allows the hardware to overlap memory loads (indices and gradients) for subsequent iterations with the atomic add operations of the current iteration, effectively hiding global memory latency.
- **Batched Atomic Accumulation**：Performs atomic updates on blocks of data rather than individual scalar elements. By accumulating a block of gradients before writing back, the kernel reduces the total number of atomic instructions issued and alleviates pressure on the memory subsystem, though contention remains dependent on index duplication.

### 代码/骨架出处
- `TritonPath`:
```python
raw_experiences/extracted_kernels/091-embedding_backward_kernel.py
```

**坏实践（反例）**：逐元素发原子加（每个标量一次 `tl.atomic_add`）→ 原子指令洪峰打满内存子系统，重复索引下严重串行化 → ✅ 正解即本条：块级梯度先累加再批量原子写回 + `num_stages` 软件流水掩盖 load 延迟。

**已知实例**：embedding §raw (`experiences/embedding/exp_embedding_backward_001.yaml`)
- 声称性能收益：35%
- 备注：This optimization is highly effective for large batch sizes and embedding dimensions. Performance may be impacted by high rates of duplicate indices due to serialization of atomic operations.

<!-- okf:related:start -->

# 相关

- 开发指南: [Embedding / 索引拷贝算子优化](../../reference/op-class-embedding.md) — 主题相关：共享 embedding
- 相关主题: [Memory Access Optimization: vectorized 3d memory access](../memory-sram/exp_cache_store_001.md) — 主题相关：共享 indexing-operations
- 相关主题: [Memory Access Optimization: 2D Tiling Strategy](../memory-sram/exp_embedding_001.md) — 主题相关：共享 embedding, indexing-operations
- 相关主题: [Memory Access Optimization: vectorized indexing](../memory-sram/exp_gather_001.md) — 主题相关：共享 indexing-operations
- 相关主题: [Memory Access Optimization: large block processing](../memory-sram/exp_gather_flat_001.md) — 主题相关：共享 indexing-operations
- 相关主题: [Memory Access Optimization: block pointer utilization](../memory-sram/exp_idx_copy_001.md) — 主题相关：共享 embedding, indexing-operations
- 相关主题: [Memory Access Optimization: redundant access elimination](../memory-sram/exp_masked_fill_001.md) — 主题相关：共享 indexing-operations
- 相关主题: [Memory Access Optimization: register buffering](../memory-sram/exp_moe_binned_scatter_001.md) — 主题相关：共享 indexing-operations
- 相关主题: [Memory Access Optimization: vectorized metadata loading](../memory-sram/exp_moe_gather_001.md) — 主题相关：共享 indexing-operations
- 相关主题: [Memory Access Optimization: software pipelining](../memory-sram/exp_moe_scatter_001.md) — 主题相关：共享 indexing-operations
- 相关主题: [Memory Access Optimization: buffered writes](../memory-sram/exp_padded_gather_001.md) — 主题相关：共享 indexing-operations
- 相关主题: [Memory Access Optimization: head vectorization](../memory-sram/exp_paged_kv_cache_001.md) — 主题相关：共享 indexing-operations
- 相关主题: [Memory Access Optimization: vectorized memory access](../memory-sram/exp_scatter_001.md) — 主题相关：共享 indexing-operations
- 相关主题: [Memory Access Optimization: sorted bin grouping with padded alignment](../memory-sram/exp_scatter_moe_padded_001.md) — 主题相关：共享 indexing-operations
- 相关主题: [Memory Access Optimization: binning strategy](../memory-sram/exp_scatter_npu_001.md) — 主题相关：共享 indexing-operations
- 相关主题: [Memory Access Optimization: padded binning for grouped scatter reduction](../memory-sram/exp_scatter_wgrad_moe_001.md) — 主题相关：共享 indexing-operations
- 相关主题: [Memory Access Optimization: intermediate data residency](../memory-sram/exp_spec_dec_reject_001.md) — 主题相关：共享 indexing-operations
- 相关主题: [Memory Access Optimization: vectorized comparison](../memory-sram/exp_spec_dec_verify_001.md) — 主题相关：共享 indexing-operations
- 相关主题: [Memory Access Optimization: block ptr utilization](../memory-sram/exp_topk_pad_001.md) — 主题相关：共享 indexing-operations
- 相关主题: [动态长度标量更新 kernel：do_not_specialize 防多版本编译，编译时间降 >80%](../meta-skills/exp_bonus_renew_001.md) — 主题相关：共享 indexing-operations
- 相关主题: [Scheduling Optimization: dynamic autotuning](exp_addmm_ascend_001.md) — 主题相关：共享 多核并行与负载均衡
- 相关主题: [Scheduling Optimization: pad slot early exit](exp_causal_conv1d_prefill_005.md) — 主题相关：共享 多核并行与负载均衡
- 相关主题: [Scheduling Optimization: bdt layout rationale](exp_causal_conv1d_update_007.md) — 主题相关：共享 多核并行与负载均衡
- 相关主题: [Scheduling Optimization: compilation specialization control](exp_expand_serial_001.md) — 主题相关：共享 indexing-operations, 多核并行与负载均衡
- 相关主题: [流程优化: 标准 score-归一化-加权三步流程](exp_fused_attention_001.md) — 主题相关：共享 多核并行与负载均衡
- 相关主题: [Scheduling Optimization: heterogeneous async scheduling](exp_lightning_attn_001.md) — 主题相关：共享 多核并行与负载均衡
- 相关主题: [Compute Optimization: vectorized binary search](exp_multinomial_001.md) — 主题相关：共享 indexing-operations
- 相关主题: [算子拆分优化: 按计算阶段拆分为简单 kernel](exp_perf_010.md) — 主题相关：共享 多核并行与负载均衡
- 相关主题: [Scheduling Optimization: lookback algorithm](exp_radix_sort_scatter_001.md) — 主题相关：共享 indexing-operations, 多核并行与负载均衡
- 相关主题: [Compute Optimization: vectorized load](exp_reject_sampler_001.md) — 主题相关：共享 indexing-operations
- 相关主题: [Scheduling Optimization: vectorized global load](exp_rejection_sample_001.md) — 主题相关：共享 indexing-operations, 多核并行与负载均衡
- 相关主题: [Compute Optimization: batched reduction replacement](exp_scatter_wgrad_001.md) — 主题相关：共享 indexing-operations
- 相关主题: [Scheduling Optimization: block local sorting](exp_shuffle_blockwise_001.md) — 主题相关：共享 indexing-operations, 多核并行与负载均衡
- 相关主题: [Scheduling Optimization: three tier hierarchical dispatch](exp_solve_tril_004.md) — 主题相关：共享 多核并行与负载均衡
- 相关主题: [Compute Optimization: padding strategy](exp_sort_bitonic_001.md) — 主题相关：共享 indexing-operations
- 相关主题: [Compute Optimization: vectorized probability calculation](exp_top_p_sampling_001.md) — 主题相关：共享 indexing-operations

<!-- okf:related:end -->
