---
schema_version: okf.v1
kind: operator_optimization
type: optimization_runbook
source_family: runbooks
category: memory-sram
dimension: 1.1
title: "Memory Access Optimization: 2D Tiling Strategy"
description: "embedding 场景下，Memory Access Optimization: 2D Tiling Strategy，收益 25%"
tags: [访存效率, indexing-operations, embedding]
confidence: verified
status: verified
created_at: 2026-08-21T00:00:00Z
updated_at: 2026-08-21T00:00:00Z
---

## Memory Access Optimization: 2D Tiling Strategy

【优化维度】：访存效率
【泛化层级】：条件性（迁移条件：硬件 910b2, 910b3, 91093, 91095，fp16/bf16/fp32，embedding_dim≥64、n_elements≥32；失效边界：embedding_dim 很小、访存非瓶颈时 2D tiling 收益稀释；BLOCK_SIZE_M×BLOCK_SIZE_N 超出 UB 预算时需降档；num_stages 流水深度同样受 UB 容量约束）

### 原则
- **2D Tiling Strategy**：The kernel tiles the output matrix into blocks of BLOCK_SIZE_M x BLOCK_SIZE_N. This allows for vectorized memory loads and stores along the embedding dimension (N), significantly improving memory bandwidth utilization compared to row-wise processing.
- **Software Pipelining**：Utilizes the num_stages parameter in tl.range to enable software pipelining. This overlaps memory access operations for subsequent blocks with the current block's execution, effectively hiding global memory latency.
- **Continuous Addressing**：The address calculation embedding_offsets = indices[:, None] * embedding_dim + offsets_n[None, :] ensures that 单个 program 内沿 embedding 维的访存地址连续，GM→UB 搬运可按向量指令批量发起，最大化访存事务效率。

### 代码/骨架出处
- `TritonPath`:
```python
raw_experiences/extracted_kernels/090-embedding_forward_kernel.py
```

**坏实践（反例）**：逐行（row-wise）处理查表结果
```python
for i in range(M):
    row = tl.load(table_ptr + idx[i] * embedding_dim + tl.arange(0, N))  # 每次只搬一行
    tl.store(out_ptr + i * embedding_dim + tl.arange(0, N), row)
```
后果：GM→UB 无法沿 embedding 维向量化，访存事务细碎，带宽利用率低。✅ 正解：BLOCK_SIZE_M×BLOCK_SIZE_N 2D tiling，`indices[:, None] * embedding_dim + offsets_n[None, :]` 一次构造整块连续地址，配合 tl.range 的 num_stages 软件流水隐藏 GM 时延，收益 25%。

**已知实例**：embedding §raw (`experiences/embedding/exp_embedding_001.yaml`)
- 声称性能收益：25%
- 备注：Best suited for scenarios with large embedding dimensions where memory bandwidth is the primary bottleneck.

<!-- okf:related:start -->

# 相关

- 开发指南: [Embedding / 索引拷贝算子优化](../../reference/op-class-embedding.md) — 主题相关：共享 embedding
- 相关主题: [Scheduling Optimization: 2D Block Tiling](../compute-pipeline/exp_embedding_backward_001.md) — 主题相关：共享 embedding, indexing-operations
- 相关主题: [Scheduling Optimization: compilation specialization control](../compute-pipeline/exp_expand_serial_001.md) — 主题相关：共享 indexing-operations
- 相关主题: [Compute Optimization: vectorized binary search](../compute-pipeline/exp_multinomial_001.md) — 主题相关：共享 indexing-operations
- 相关主题: [Scheduling Optimization: lookback algorithm](../compute-pipeline/exp_radix_sort_scatter_001.md) — 主题相关：共享 indexing-operations
- 相关主题: [Compute Optimization: vectorized load](../compute-pipeline/exp_reject_sampler_001.md) — 主题相关：共享 indexing-operations
- 相关主题: [Scheduling Optimization: vectorized global load](../compute-pipeline/exp_rejection_sample_001.md) — 主题相关：共享 indexing-operations
- 相关主题: [Compute Optimization: batched reduction replacement](../compute-pipeline/exp_scatter_wgrad_001.md) — 主题相关：共享 indexing-operations
- 相关主题: [Scheduling Optimization: block local sorting](../compute-pipeline/exp_shuffle_blockwise_001.md) — 主题相关：共享 indexing-operations
- 相关主题: [Compute Optimization: padding strategy](../compute-pipeline/exp_sort_bitonic_001.md) — 主题相关：共享 indexing-operations
- 相关主题: [Compute Optimization: vectorized probability calculation](../compute-pipeline/exp_top_p_sampling_001.md) — 主题相关：共享 indexing-operations
- 相关主题: [动态长度标量更新 kernel：do_not_specialize 防多版本编译，编译时间降 >80%](../meta-skills/exp_bonus_renew_001.md) — 主题相关：共享 indexing-operations
- 相关主题: [大 shape 张量连续分块访存（缩小 grid、单核处理连续段）](exp_ascend_large_shape_block_001.md) — 主题相关：共享 访存效率
- 相关主题: [Memory Access Optimization: vectorized 3d memory access](exp_cache_store_001.md) — 主题相关：共享 indexing-operations, 访存效率
- 相关主题: [Memory Access Optimization: flattened grid scheduling](exp_causal_conv1d_001.md) — 主题相关：共享 访存效率
- 相关主题: [Memory Access Optimization: compile time branching](exp_causal_conv1d_states_001.md) — 主题相关：共享 访存效率
- 相关主题: [Memory Access Optimization: fused state update](exp_causal_conv1d_update_001.md) — 主题相关：共享 访存效率
- 相关主题: [Memory Access Optimization: overlapping t tiling](exp_causal_conv1d_update_004.md) — 主题相关：共享 访存效率
- 相关主题: [Memory Access Optimization: simd index select](exp_flashattn_gqa_001.md) — 主题相关：共享 访存效率
- 相关主题: [Memory Access Optimization: vectorized indexing](exp_gather_001.md) — 主题相关：共享 indexing-operations, 访存效率
- 相关主题: [Memory Access Optimization: large block processing](exp_gather_flat_001.md) — 主题相关：共享 indexing-operations, 访存效率
- 相关主题: [Memory Access Optimization: block pointer utilization](exp_idx_copy_001.md) — 主题相关：共享 embedding, indexing-operations, 访存效率
- 相关主题: [内存访问优化：转置 load 行优先化与跨阶段复用](exp_kda_bwd_intra_memory_opt_001.md) — 主题相关：共享 访存效率
- 相关主题: [Memory Access Optimization: adaptive tiling strategy](exp_layernorm_bwd_001.md) — 主题相关：共享 访存效率
- 相关主题: [Memory Access Optimization: redundant access elimination](exp_masked_fill_001.md) — 主题相关：共享 indexing-operations, 访存效率
- 相关主题: [Memory Access Optimization: register buffering](exp_moe_binned_scatter_001.md) — 主题相关：共享 indexing-operations, 访存效率
- 相关主题: [Memory Access Optimization: vectorized metadata loading](exp_moe_gather_001.md) — 主题相关：共享 indexing-operations, 访存效率
- 相关主题: [Memory Access Optimization: software pipelining](exp_moe_scatter_001.md) — 主题相关：共享 indexing-operations, 访存效率
- 相关主题: [Memory Access Optimization: buffered writes](exp_padded_gather_001.md) — 主题相关：共享 indexing-operations, 访存效率
- 相关主题: [Memory Access Optimization: head vectorization](exp_paged_kv_cache_001.md) — 主题相关：共享 indexing-operations, 访存效率
- 相关主题: [Memory Access Optimization: vectorized memory access](exp_scatter_001.md) — 主题相关：共享 indexing-operations, 访存效率
- 相关主题: [Memory Access Optimization: sorted bin grouping with padded alignment](exp_scatter_moe_padded_001.md) — 主题相关：共享 indexing-operations, 访存效率
- 相关主题: [Memory Access Optimization: binning strategy](exp_scatter_npu_001.md) — 主题相关：共享 indexing-operations, 访存效率
- 相关主题: [Memory Access Optimization: padded binning for grouped scatter reduction](exp_scatter_wgrad_moe_001.md) — 主题相关：共享 indexing-operations, 访存效率
- 相关主题: [Memory Access Optimization: intermediate data residency](exp_spec_dec_reject_001.md) — 主题相关：共享 indexing-operations, 访存效率
- 相关主题: [Memory Access Optimization: vectorized comparison](exp_spec_dec_verify_001.md) — 主题相关：共享 indexing-operations, 访存效率
- 相关主题: [Memory Access Optimization: block ptr utilization](exp_topk_pad_001.md) — 主题相关：共享 indexing-operations, 访存效率

<!-- okf:related:end -->
