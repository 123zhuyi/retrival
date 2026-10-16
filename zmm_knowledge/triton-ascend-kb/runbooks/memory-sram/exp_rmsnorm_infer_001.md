---
schema_version: okf.v1
kind: operator_optimization
type: optimization_runbook
source_family: runbooks
category: memory-sram
dimension: 1.1
title: "Memory Access Optimization: grid-stride scheduling with two-pass reduction"
description: "normalization 场景下，Memory Access Optimization: grid-stride scheduling with two-pass reduction，收益 25%"
tags: [访存效率, normalization-operations, normalization]
confidence: verified
status: verified
created_at: 2026-08-21T00:00:00Z
updated_at: 2026-08-21T00:00:00Z
---

## Memory Access Optimization: grid-stride scheduling with two-pass reduction

【优化维度】：访存效率
【泛化层级】：条件性（迁移条件：硬件 910b2/910b3，fp16/bf16，n_cols >= 128 的推理归一化场景；失效边界：两遍法是用翻倍 GM 读带宽换低寄存器压力，n_cols 小、整行可驻留寄存器/UB 时单遍方案更优；UB 容量约束：BLOCK_SIZE_N × 4B（fp32 累加）× 驻留份数 ≤ UB 预算，升精度累加的预算要按 fp32 计）

### 原则
- **grid stride scheduling**：The kernel utilizes a grid-stride loop (`for row_task_id in range(pid, num_row_tasks, grid_size)`) to process rows. This allows a single kernel launch to handle an arbitrary number of rows exceeding the grid size, ensuring all compute units remain active even for large batch sizes or sequence lengths.
- **two pass algorithm**：The computation is split into two distinct passes over the input data. The first pass calculates the sum of squares (variance), and the second pass applies the normalization and scaling. This decouples the reduction dependency from the output calculation, significantly reducing register pressure by avoiding the need to store the entire input block in registers between the reduction and output stages.
- **high precision accumulation**：The kernel explicitly casts input data to `tl.float32` before accumulating the sum of squares (`ss_acc`) and performing the final arithmetic. This ensures numerical stability during the reduction, preventing overflow or underflow that can occur with low-precision floating-point types (fp16/bf16).
- **tiled memory access**：By iterating over columns in blocks of `BLOCK_SIZE_N`, the kernel ensures GM→UB 访存连续且对齐到有利边界, maximizing memory bandwidth utilization for both the input tensor `X` and the weight tensor `W`.

### 代码/骨架出处
- `TritonPath`:
```python
_rmsnorm_infer_kernel
```

**坏实践（反例）**：归约与输出单遍耦合、整行驻留寄存器
```python
x_blk = tl.load(X_ptr + row_off + tl.arange(0, n_cols))  # 整行寄存器驻留
ss = tl.sum(x_blk * x_blk)
y = x_blk * tl.rsqrt(ss / n_cols) * W
```
后果：n_cols 大时整行加中间量撑爆寄存器，触发 spill。✅ 正解：两遍法——第一遍 fp32 累加平方和（ss_acc），第二遍重载数据做归一化缩放；配合 grid-stride 循环让固定数量 program 处理任意行数（收益 25%）。

**已知实例**：normalization §raw (`experiences/normalization/exp_rmsnorm_infer_001.yaml`)
- 声称性能收益：25%
- 备注：The two-pass approach trades off increased memory bandwidth usage (loading X twice) for lower register pressure, which is highly effective for operations with large hidden dimensions (n_cols).

<!-- okf:related:start -->

# 相关

- 相关主题: [跨 tile 归约必须显式两阶段 reduce-then-apply](../algorithm-numerics/two-pass-reduce-then-apply.md) — 同属优化知识，主题相关
- 相关主题: [Scheduling Optimization: two dimensional grid strategy](../compute-pipeline/exp_causal_conv1d_prefill_003.md) — 同属优化知识，主题相关
- 相关主题: [Compute Optimization: two pass algorithm](../compute-pipeline/exp_celoss_bwd_001.md) — 同属优化知识，主题相关
- 相关主题: [Compute Optimization: two pass algorithm](../compute-pipeline/exp_celoss_smooth_bwd_001.md) — 同属优化知识，主题相关
- 相关主题: [Compute Optimization: npu simd vectorization](../compute-pipeline/exp_layernorm_npu_simd_001.md) — 主题相关：共享 normalization, normalization-operations
- 相关主题: [Compute Optimization: two pass algorithm](../compute-pipeline/exp_log_softmax_001.md) — 同属优化知识，主题相关
- 相关主题: [Compute Optimization: two pass algorithm](../compute-pipeline/exp_log_softmax_backward_001.md) — 同属优化知识，主题相关
- 相关主题: [Compute Optimization: two pass algorithm](../compute-pipeline/exp_rmsnorm_backward_npu_001.md) — 主题相关：共享 normalization
- 相关主题: [大 shape 张量连续分块访存（缩小 grid、单核处理连续段）](exp_ascend_large_shape_block_001.md) — 主题相关：共享 访存效率
- 相关主题: [Memory Access Optimization: kv tile reuse](exp_attn_bwd_hstu_001.md) — 主题相关：共享 访存效率
- 相关主题: [Memory Access Optimization: vectorized offset generation](exp_attn_causal_mask_001.md) — 主题相关：共享 访存效率
- 相关主题: [Memory Access Optimization: fp8 value casting](exp_attn_fp8_ub_slice_001.md) — 主题相关：共享 访存效率
- 相关主题: [Memory Access Optimization: block pointer abstraction](exp_attn_ptr_setup_001.md) — 主题相关：共享 访存效率
- 相关主题: [Memory Access Optimization: online softmax](exp_attn_sinks_001.md) — 主题相关：共享 访存效率
- 相关主题: [Memory Access Optimization: sparse gather optimization](exp_attn_sparse_bwd_001.md) — 主题相关：共享 访存效率
- 相关主题: [Memory Access Optimization: vectorized 3d memory access](exp_cache_store_001.md) — 主题相关：共享 访存效率
- 相关主题: [Memory Access Optimization: flattened grid scheduling](exp_causal_conv1d_001.md) — 主题相关：共享 访存效率
- 相关主题: [Memory Access Optimization: block ptr utilization](exp_causal_conv1d_bwd_001.md) — 主题相关：共享 访存效率
- 相关主题: [Memory Access Optimization: sliding window register reuse](exp_causal_conv1d_prefill_002.md) — 主题相关：共享 访存效率
- 相关主题: [Memory Access Optimization: compile time branching](exp_causal_conv1d_states_001.md) — 主题相关：共享 访存效率
- 相关主题: [Memory Access Optimization: fused state update](exp_causal_conv1d_update_001.md) — 主题相关：共享 访存效率
- 相关主题: [Memory Access Optimization: make block ptr hardware acceleration](exp_causal_conv1d_update_003.md) — 主题相关：共享 访存效率
- 相关主题: [Memory Access Optimization: fused split and dual-norm single-pass load](exp_fused_split_qk_norm_001.md) — 主题相关：共享 normalization, 访存效率
- 相关主题: [Memory Access Optimization: Tiled Two-Pass Reduction](exp_gn_backward_001.md) — 主题相关：共享 normalization, normalization-operations, 访存效率
- 相关主题: [Memory Access Optimization: dynamic vectorization tuning](exp_groupnorm_001.md) — 主题相关：共享 normalization, normalization-operations, 访存效率
- 相关主题: [Memory Access Optimization: adaptive tiling strategy](exp_layernorm_bwd_001.md) — 主题相关：共享 normalization, normalization-operations, 访存效率
- 相关主题: [Memory Access Optimization: two-pass reduction for large columns](exp_rmsnorm_bwd_large_cols_001.md) — 主题相关：共享 normalization, 访存效率
- 相关主题: [Memory Access Optimization: single-pass tiling via UB residency](exp_rmsnorm_grad_dx_001.md) — 主题相关：共享 normalization, normalization-operations, 访存效率
- 相关主题: [Memory Access Optimization: two pass reduction strategy](exp_softmax_backward_001.md) — 主题相关：共享 访存效率
- 相关主题: [Memory Access Optimization: two pass reduction strategy](exp_softmax_backward_non_inner_001.md) — 主题相关：共享 访存效率

<!-- okf:related:end -->
