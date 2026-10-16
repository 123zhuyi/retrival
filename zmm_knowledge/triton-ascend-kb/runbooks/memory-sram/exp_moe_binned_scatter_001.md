---
schema_version: okf.v1
kind: operator_optimization
type: optimization_runbook
source_family: runbooks
category: memory-sram
dimension: 1.1
title: "Memory Access Optimization: register buffering"
description: "gather_scatter 场景下，Memory Access Optimization: register buffering，收益 30%"
tags: [访存效率, indexing-operations, gather_scatter]
confidence: verified
status: verified
created_at: 2026-08-21T00:00:00Z
updated_at: 2026-08-21T00:00:00Z
---

## Memory Access Optimization: register buffering

【优化维度】：访存效率
【泛化层级】：条件性（迁移条件：硬件 910b2/910b3，fp16/bf16，NUM_COLUMNS≥128、BLOCK_SIZE≥64、expert_capacity 足以容纳单桶 token；失效边界：expert_capacity 不足时越界写；单个 SUB_BLOCK_SIZE 迭代内桶切换极频繁时 register buffer 反复 flush，批量写收益稀释；NUM_COLUMNS<BLOCK_X 时列向量化退化）

### 原则
- **register buffering**：The kernel utilizes a register-level buffer (`tmp_buf`) sized `(SUB_BLOCK_SIZE, BLOCK_X)` to accumulate data. By gathering multiple elements from the source tensor `a` into this buffer before performing a global store to `b`, the number of high-latency global memory write operations is significantly reduced.
- **vectorized memory access**：The inner loop iterates over the feature dimension (`NUM_COLUMNS`) with a step size of `BLOCK_X`. This allows the kernel to load and store data using wide vector instructions (e.g., 128-bit or 256-bit), maximizing memory bandwidth utilization on the NPU.
- **dynamic bin handling**：The logic efficiently handles transitions between different expert bins within a single `SUB_BLOCK_SIZE` iteration. By checking `if bin_idx != base_bin_idx`, it flushes the current buffer and updates the target address dynamically, ensuring correct data placement without breaking the vectorized processing loop.
- **address calculation optimization**：The source address calculation `idx_a = (indices // TOP_K) * NUM_COLUMNS` is hoisted outside the column loop, and the destination address `idx_b` is updated incrementally. This minimizes redundant arithmetic instructions within the critical inner loops.

### 代码/骨架出处
- `TritonPath`:
```python
raw_experiences/extracted_kernels/002-_binned_copy_gather_npu.py
```

**坏实践（反例）**：每 gather 一个元素立即做一次全局 store，不做寄存器聚合
```python
for i in range(SUB_BLOCK_SIZE):
    idx_a = (indices[i] // TOP_K) * NUM_COLUMNS   # 循环内重复算地址
    val = tl.load(a_ptr + idx_a + col_off)        # 逐元素 gather
    tl.store(b_ptr + idx_b + col_off, val)        # 每元素一次高延迟 GM 写
```
后果：GM 写事务数等于元素数，高延迟写无法合并；地址算术散在关键内层循环里，指令开销叠加，30% 收益流失。✅ 正解：用 `(SUB_BLOCK_SIZE, BLOCK_X)` 的寄存器级 tmp_buf 聚合数据，仅在 bin 切换（`bin_idx != base_bin_idx`）时批量 flush；源地址 `idx_a` 提升到列循环外、目的地址 `idx_b` 增量更新。

**已知实例**：gather_scatter §raw (`experiences/gather_scatter/exp_moe_binned_scatter_001.yaml`)
- 声称性能收益：30%
- 备注：该优化专为专家路由场景设计——token 需散写到指定专家桶。假设 `expert_capacity` 足够容纳每个桶的 token。

<!-- okf:related:start -->

# 相关

- 开发指南: [Gather/Scatter 算子优化](../../reference/op-class-gather_scatter.md) — 主题相关：共享 gather_scatter
- 相关主题: [Scheduling Optimization: 2D Block Tiling](../compute-pipeline/exp_embedding_backward_001.md) — 主题相关：共享 indexing-operations
- 相关主题: [Scheduling Optimization: compilation specialization control](../compute-pipeline/exp_expand_serial_001.md) — 主题相关：共享 indexing-operations
- 相关主题: [Compute Optimization: register accumulation](../compute-pipeline/exp_hist_count_001.md) — 同属优化知识，主题相关
- 相关主题: [Compute Optimization: vectorized binary search](../compute-pipeline/exp_multinomial_001.md) — 主题相关：共享 indexing-operations
- 相关主题: [Scheduling Optimization: lookback algorithm](../compute-pipeline/exp_radix_sort_scatter_001.md) — 主题相关：共享 gather_scatter, indexing-operations
- 相关主题: [Compute Optimization: vectorized load](../compute-pipeline/exp_reject_sampler_001.md) — 主题相关：共享 indexing-operations
- 相关主题: [Scheduling Optimization: vectorized global load](../compute-pipeline/exp_rejection_sample_001.md) — 主题相关：共享 indexing-operations
- 相关主题: [Compute Optimization: batched reduction replacement](../compute-pipeline/exp_scatter_wgrad_001.md) — 主题相关：共享 gather_scatter, indexing-operations
- 相关主题: [Scheduling Optimization: block local sorting](../compute-pipeline/exp_shuffle_blockwise_001.md) — 主题相关：共享 indexing-operations
- 相关主题: [动态长度标量更新 kernel：do_not_specialize 防多版本编译，编译时间降 >80%](../meta-skills/exp_bonus_renew_001.md) — 主题相关：共享 indexing-operations
- 相关主题: [Memory Access Optimization: vectorized 3d memory access](exp_cache_store_001.md) — 主题相关：共享 indexing-operations, 访存效率
- 相关主题: [Memory Access Optimization: sliding window register reuse](exp_causal_conv1d_prefill_002.md) — 主题相关：共享 访存效率
- 相关主题: [Memory Access Optimization: 2D Tiling Strategy](exp_embedding_001.md) — 主题相关：共享 indexing-operations, 访存效率
- 相关主题: [Memory Access Optimization: vectorized indexing](exp_gather_001.md) — 主题相关：共享 gather_scatter, indexing-operations, 访存效率
- 相关主题: [Memory Access Optimization: large block processing](exp_gather_flat_001.md) — 主题相关：共享 gather_scatter, indexing-operations, 访存效率
- 相关主题: [Memory Access Optimization: block pointer utilization](exp_idx_copy_001.md) — 主题相关：共享 indexing-operations, 访存效率
- 相关主题: [Memory Access Optimization: redundant access elimination](exp_masked_fill_001.md) — 主题相关：共享 indexing-operations, 访存效率
- 相关主题: [Memory Access Optimization: vectorized metadata loading](exp_moe_gather_001.md) — 主题相关：共享 gather_scatter, indexing-operations, 访存效率
- 相关主题: [Memory Access Optimization: software pipelining](exp_moe_scatter_001.md) — 主题相关：共享 gather_scatter, indexing-operations, 访存效率
- 相关主题: [Memory Access Optimization: buffered writes](exp_padded_gather_001.md) — 主题相关：共享 gather_scatter, indexing-operations, 访存效率
- 相关主题: [Memory Access Optimization: head vectorization](exp_paged_kv_cache_001.md) — 主题相关：共享 indexing-operations, 访存效率
- 相关主题: [Memory Access Optimization: register reuse intermediate](exp_rmsnorm_add_001.md) — 主题相关：共享 访存效率
- 相关主题: [Memory Access Optimization: no-tiling register reuse with conflict-free dW accumulation](exp_rmsnorm_backward_001.md) — 主题相关：共享 访存效率
- 相关主题: [Memory Access Optimization: vectorized memory access](exp_scatter_001.md) — 主题相关：共享 gather_scatter, indexing-operations, 访存效率
- 相关主题: [Memory Access Optimization: sorted bin grouping with padded alignment](exp_scatter_moe_padded_001.md) — 主题相关：共享 gather_scatter, indexing-operations, 访存效率
- 相关主题: [Memory Access Optimization: binning strategy](exp_scatter_npu_001.md) — 主题相关：共享 gather_scatter, indexing-operations, 访存效率
- 相关主题: [Memory Access Optimization: padded binning for grouped scatter reduction](exp_scatter_wgrad_moe_001.md) — 主题相关：共享 gather_scatter, indexing-operations, 访存效率
- 相关主题: [Memory Access Optimization: intermediate data residency](exp_spec_dec_reject_001.md) — 主题相关：共享 indexing-operations, 访存效率
- 相关主题: [Memory Access Optimization: vectorized comparison](exp_spec_dec_verify_001.md) — 主题相关：共享 indexing-operations, 访存效率
- 相关主题: [Memory Access Optimization: block ptr utilization](exp_topk_pad_001.md) — 主题相关：共享 indexing-operations, 访存效率
- 相关主题: [并行轴选择必须与连续访问对齐，而非与最大 numel 对齐](parallel-axis-align.md) — 主题相关：共享 gather_scatter
- 相关主题: [寄存器移位复用滑动窗口，消除重复 HBM 读](register-shift-sliding-window.md) — 同属优化知识，主题相关

<!-- okf:related:end -->
