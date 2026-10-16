---
schema_version: okf.v1
kind: operator_optimization
type: optimization_runbook
source_family: runbooks
category: memory-sram
dimension: 1.1
title: "Memory Access Optimization: vectorized metadata loading"
description: "gather_scatter 场景下，Memory Access Optimization: vectorized metadata loading，收益 20%"
tags: [访存效率, indexing-operations, gather_scatter]
confidence: verified
status: verified
created_at: 2026-08-21T00:00:00Z
updated_at: 2026-08-21T00:00:00Z
---

## Memory Access Optimization: vectorized metadata loading

【优化维度】：访存效率
【泛化层级】：条件性（迁移条件：硬件 910b2/910b3/91093/91095，fp16/bf16，topk_num≤8 且为编译期常量，BLOCK_D≥128；失效边界：topk_num 非编译期常量或 >8 时循环展开与元数据向量化失效、寄存器压力上升；BLOCK_D<128（fp16 下不足 64 元素）时向量加载退化为部分访存事务）

### 原则
- **vectorized metadata loading**：The current implementation performs scalar loads for `expert_id`, `source_token_index`, and `recv_topk_weight` inside the inner loop. Since `topk_num` is a compile-time constant, these loads should be vectorized to fetch the entire array of indices/weights for the current token in a single instruction. This reduces the number of memory transactions and instruction overhead significantly.
- **loop unrolling**：The inner loop iterating over `topk_num` is typically small (e.g., 2 to 8). Unrolling this loop eliminates branch overhead and allows the compiler to schedule independent memory loads and floating-point operations more effectively, hiding latency.
- **memory alignment**：The vector loads of `input_tensor` and stores to `output_tensor` should be aligned to the hardware's memory transaction size (e.g., 128 bytes). Ensuring `BLOCK_D` is a multiple of the alignment size (e.g., 64 or 128 for fp16) prevents partial memory transactions and improves bandwidth utilization.

### 代码/骨架出处
- `TritonPath`:
```python
raw_experiences/extracted_kernels/062-_fwd_kernel_ep_gather.py
```

**坏实践（反例）**：内层循环逐标量加载路由元数据
```python
for k in range(topk_num):                    # 未展开的小循环，带分支开销
    eid = tl.load(expert_id_ptr + k)         # 标量 load
    src = tl.load(source_index_ptr + k)      # 标量 load
    w   = tl.load(recv_weight_ptr + k)       # 标量 load
    acc += w * tl.load(input_ptr + src * BLOCK_D + d_off)
```
后果：每个 token 触发 3×topk_num 次标量访存事务，指令与分支开销显著，20% 收益流失。✅ 正解：topk_num 为编译期常量时用 `tl.arange(0, topk_num)` 一次性向量取回全部索引/权重，并展开该小循环让编译器调度独立 load 掩盖延迟；同时保证 BLOCK_D 为对齐粒度倍数（fp16 下 64/128 元素）。

**已知实例**：gather_scatter §raw (`experiences/gather_scatter/exp_moe_gather_001.yaml`)
- 声称性能收益：20%
- 备注：该优化对推理阶段路由开销不可忽略的专家路由层尤为关键。假设 `topk_num` 较小且编译期已知。

<!-- okf:related:start -->

# 相关

- 开发指南: [Gather/Scatter 算子优化](../../reference/op-class-gather_scatter.md) — 主题相关：共享 gather_scatter
- 相关主题: [Scheduling Optimization: 2D Block Tiling](../compute-pipeline/exp_embedding_backward_001.md) — 主题相关：共享 indexing-operations
- 相关主题: [Scheduling Optimization: compilation specialization control](../compute-pipeline/exp_expand_serial_001.md) — 主题相关：共享 indexing-operations
- 相关主题: [Compute Optimization: vectorized binary search](../compute-pipeline/exp_multinomial_001.md) — 主题相关：共享 indexing-operations
- 相关主题: [Scheduling Optimization: lookback algorithm](../compute-pipeline/exp_radix_sort_scatter_001.md) — 主题相关：共享 gather_scatter, indexing-operations
- 相关主题: [Compute Optimization: vectorized load](../compute-pipeline/exp_reject_sampler_001.md) — 主题相关：共享 indexing-operations
- 相关主题: [Scheduling Optimization: vectorized global load](../compute-pipeline/exp_rejection_sample_001.md) — 主题相关：共享 indexing-operations
- 相关主题: [Compute Optimization: batched reduction replacement](../compute-pipeline/exp_scatter_wgrad_001.md) — 主题相关：共享 gather_scatter, indexing-operations
- 相关主题: [Scheduling Optimization: block local sorting](../compute-pipeline/exp_shuffle_blockwise_001.md) — 主题相关：共享 indexing-operations
- 相关主题: [Compute Optimization: padding strategy](../compute-pipeline/exp_sort_bitonic_001.md) — 主题相关：共享 indexing-operations
- 相关主题: [Compute Optimization: vectorized probability calculation](../compute-pipeline/exp_top_p_sampling_001.md) — 主题相关：共享 indexing-operations
- 相关主题: [动态长度标量更新 kernel：do_not_specialize 防多版本编译，编译时间降 >80%](../meta-skills/exp_bonus_renew_001.md) — 主题相关：共享 indexing-operations
- 相关主题: [Memory Access Optimization: vectorized 3d memory access](exp_cache_store_001.md) — 主题相关：共享 indexing-operations, 访存效率
- 相关主题: [Memory Access Optimization: 2D Tiling Strategy](exp_embedding_001.md) — 主题相关：共享 indexing-operations, 访存效率
- 相关主题: [Memory Access Optimization: vectorized indexing](exp_gather_001.md) — 主题相关：共享 gather_scatter, indexing-operations, 访存效率
- 相关主题: [Memory Access Optimization: large block processing](exp_gather_flat_001.md) — 主题相关：共享 gather_scatter, indexing-operations, 访存效率
- 相关主题: [Memory Access Optimization: block pointer utilization](exp_idx_copy_001.md) — 主题相关：共享 indexing-operations, 访存效率
- 相关主题: [Memory Access Optimization: redundant access elimination](exp_masked_fill_001.md) — 主题相关：共享 indexing-operations, 访存效率
- 相关主题: [Memory Access Optimization: register buffering](exp_moe_binned_scatter_001.md) — 主题相关：共享 gather_scatter, indexing-operations, 访存效率
- 相关主题: [Memory Access Optimization: software pipelining](exp_moe_scatter_001.md) — 主题相关：共享 gather_scatter, indexing-operations, 访存效率
- 相关主题: [Memory Access Optimization: buffered writes](exp_padded_gather_001.md) — 主题相关：共享 gather_scatter, indexing-operations, 访存效率
- 相关主题: [Memory Access Optimization: head vectorization](exp_paged_kv_cache_001.md) — 主题相关：共享 indexing-operations, 访存效率
- 相关主题: [Memory Access Optimization: 2D vector loading for small hidden dims](exp_rmsnorm_fused_001.md) — 主题相关：共享 访存效率
- 相关主题: [Memory Access Optimization: vectorized memory access](exp_scatter_001.md) — 主题相关：共享 gather_scatter, indexing-operations, 访存效率
- 相关主题: [Memory Access Optimization: sorted bin grouping with padded alignment](exp_scatter_moe_padded_001.md) — 主题相关：共享 gather_scatter, indexing-operations, 访存效率
- 相关主题: [Memory Access Optimization: binning strategy](exp_scatter_npu_001.md) — 主题相关：共享 gather_scatter, indexing-operations, 访存效率
- 相关主题: [Memory Access Optimization: padded binning for grouped scatter reduction](exp_scatter_wgrad_moe_001.md) — 主题相关：共享 gather_scatter, indexing-operations, 访存效率
- 相关主题: [Memory Access Optimization: intermediate data residency](exp_spec_dec_reject_001.md) — 主题相关：共享 indexing-operations, 访存效率
- 相关主题: [Memory Access Optimization: vectorized comparison](exp_spec_dec_verify_001.md) — 主题相关：共享 indexing-operations, 访存效率
- 相关主题: [Memory Access Optimization: block ptr utilization](exp_topk_pad_001.md) — 主题相关：共享 indexing-operations, 访存效率
- 相关主题: [并行轴选择必须与连续访问对齐，而非与最大 numel 对齐](parallel-axis-align.md) — 主题相关：共享 gather_scatter

<!-- okf:related:end -->
