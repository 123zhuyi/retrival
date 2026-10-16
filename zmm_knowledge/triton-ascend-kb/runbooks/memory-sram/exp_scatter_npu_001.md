---
schema_version: okf.v1
kind: operator_optimization
type: optimization_runbook
source_family: runbooks
category: memory-sram
dimension: 1.1
title: "Memory Access Optimization: binning strategy"
description: "gather_scatter 场景下，Memory Access Optimization: binning strategy，收益 35%"
tags: [访存效率, indexing-operations, gather_scatter]
confidence: verified
status: verified
created_at: 2026-08-21T00:00:00Z
updated_at: 2026-08-21T00:00:00Z
---

## Memory Access Optimization: binning strategy

【优化维度】：访存效率
【泛化层级】：条件性（迁移条件：硬件 910b2, 910b3；fp16/bf16；NUM_COLUMNS≥64、BLOCK_SIZE≥128、SUB_BLOCK_SIZE≥32；失效边界：无预排序 bins/bin_ids 的纯随机 scatter 场景连续性不成立，策略失效；SUB_BLOCK_SIZE<32 或 NUM_COLUMNS<64 时向量指令宽度不足，收益稀释）

### 原则
- **binning strategy**：The kernel leverages pre-computed bins and bin_ids to group indices. This allows the memory access pattern for the source tensor 'a' to be contiguous within the inner loop, significantly reducing memory latency compared to a purely random scatter.
- **vectorized block processing**：Data is loaded and stored in 2D tiles defined by SUB_BLOCK_SIZE and BLOCK_X. This enables the use of wide vector memory instructions (e.g., 128-bit or 256-bit loads/stores), maximizing the utilization of the memory bus bandwidth on the NPU.
- **register reuse scaling**：The scaling operation is fused into the scatter loop. By loading the weight, scaling it in registers using FP32 arithmetic, and then storing it, the kernel minimizes the number of times the data is moved between global memory and the compute unit.
- **loop tiling**：The nested loop structure (iterating over bins, then columns, then indices) optimizes for cache locality. It processes all columns for a given set of indices before moving to the next set, keeping the active working set small in the cache/SRAM.

### 代码/骨架出处
- `TritonPath`:
```python
raw_experiences/extracted_kernels/003-_binned_copy_scatter_npu.py
```

**坏实践（反例）**：不做 binning 的纯随机 scatter
```python
for i in range(N):                                # 索引乱序、未分组
    row = idx[i]
    val = tl.load(a_ptr + row*NUM_COLUMNS + offs_c, mask=...)  # 地址发散
    val = val * w[i]
    tl.store(out_ptr + row*NUM_COLUMNS + offs_c, val, mask=...)
```
后果：源张量访问地址发散，GM 访存延迟完全暴露，无法形成连续突发；binning 后实测收益 35%。✅ 正解：先用预计算的 bins/bin_ids 把同组索引聚到连续段，内层循环对源张量形成连续访问；数据按 (SUB_BLOCK_SIZE, BLOCK_X) 2D tile 用宽向量指令搬运；缩放用 FP32 寄存器运算融合进 scatter 循环，减少 GM 往返。

**已知实例**：gather_scatter §raw (`experiences/gather_scatter/exp_scatter_npu_001.yaml`)
- 声称性能收益：35%
- 备注：专为分组路由场景设计（行按分组键分派到不同分组）。要求输入为预排序索引或 bin 信息才能正确工作。

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
- 相关主题: [Memory Access Optimization: vectorized metadata loading](exp_moe_gather_001.md) — 主题相关：共享 gather_scatter, indexing-operations, 访存效率
- 相关主题: [Memory Access Optimization: software pipelining](exp_moe_scatter_001.md) — 主题相关：共享 gather_scatter, indexing-operations, 访存效率
- 相关主题: [Memory Access Optimization: buffered writes](exp_padded_gather_001.md) — 主题相关：共享 gather_scatter, indexing-operations, 访存效率
- 相关主题: [Memory Access Optimization: head vectorization](exp_paged_kv_cache_001.md) — 主题相关：共享 indexing-operations, 访存效率
- 相关主题: [Memory Access Optimization: vectorized memory access](exp_scatter_001.md) — 主题相关：共享 gather_scatter, indexing-operations, 访存效率
- 相关主题: [Memory Access Optimization: sorted bin grouping with padded alignment](exp_scatter_moe_padded_001.md) — 主题相关：共享 gather_scatter, indexing-operations, 访存效率
- 相关主题: [Memory Access Optimization: padded binning for grouped scatter reduction](exp_scatter_wgrad_moe_001.md) — 主题相关：共享 gather_scatter, indexing-operations, 访存效率
- 相关主题: [Memory Access Optimization: intermediate data residency](exp_spec_dec_reject_001.md) — 主题相关：共享 indexing-operations, 访存效率
- 相关主题: [Memory Access Optimization: vectorized comparison](exp_spec_dec_verify_001.md) — 主题相关：共享 indexing-operations, 访存效率
- 相关主题: [Memory Access Optimization: block ptr utilization](exp_topk_pad_001.md) — 主题相关：共享 indexing-operations, 访存效率
- 相关主题: [并行轴选择必须与连续访问对齐，而非与最大 numel 对齐](parallel-axis-align.md) — 主题相关：共享 gather_scatter

<!-- okf:related:end -->
