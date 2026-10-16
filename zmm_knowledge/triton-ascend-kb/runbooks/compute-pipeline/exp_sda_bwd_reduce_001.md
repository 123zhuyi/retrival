---
schema_version: okf.v1
kind: operator_optimization
type: optimization_runbook
source_family: runbooks
category: compute-pipeline
dimension: 2.1
title: 'Compute Optimization: grid stride loop'
description: 'reduction 场景下，Compute Optimization: grid stride loop，收益 25%'
tags: [AICore 单元利用/Scalar 削减, reduction-operations, reduction]
confidence: verified
status: verified
created_at: 2026-08-21T00:00:00Z
updated_at: 2026-08-21T00:00:00Z
---

## Compute Optimization: grid stride loop

【优化维度】：AICore 单元利用/Scalar 削减
【泛化层级】：条件性（迁移条件：硬件 910b2, 910b3；失效边界：BLOCK_R 需按 H 维调整以撑满向量 load 宽度（备注自述）；S < 128 时分块收益有限；bf16→fp32 提升是精度前提，省略会在长序列梯度累加中漂移）

### 原则
- **grid stride loop**：The kernel utilizes a 1D launch grid with a stride loop (`for task_id in range(pid, total_tasks, tl.num_programs(axis=0))`). This scheduling strategy decouples the number of hardware threads from the total number of logical tasks (B * N * S_blocks), ensuring high utilization of all compute cores (SMs) even when the total workload is not perfectly divisible by the grid size.
- **mixed precision accumulation**：To ensure numerical stability during the backward pass, the kernel promotes input tensors (`LOW_TYPE`, typically bf16) to `HIGH_TYPE` (fp32) before performing the multiplication and summation (`tl.sum`). This prevents potential overflow or underflow errors that can occur when accumulating many small gradient values in lower precision.
- **vectorized block loading**：Memory pointers are constructed using broadcasting (`[:, None]` and `[None, :]`) to load 2D blocks of shape `(BLOCK_R, H)` in a single operation. This maximizes global memory bandwidth utilization by ensuring contiguous memory access patterns and efficient cache line usage.
- **sequence tiling**：The sequence dimension `S` is processed in fixed-size tiles defined by `BLOCK_R` (default 64). This tiling strategy balances register pressure and instruction-level parallelism, allowing the kernel to handle long sequences that would otherwise exceed the register file capacity if processed entirely at once.

### 代码/骨架出处
- `TritonPath`:
```python
raw_experiences/extracted_kernels/261-kernel_sda_bwd_d.py
```

**坏实践（反例）**：梯度归约直接用 bf16 乘加累加 → 长序列大量小梯度值累加溢出/下溢 → ✅ 正解即本条：升 fp32 做乘加归约（`tl.sum`）+ grid-stride 解耦任务数与核数。

**已知实例**：reduction §raw (`experiences/reduction/exp_sda_bwd_reduce_001.yaml`)
- 声称性能收益：25%
- 备注：This kernel is specifically designed for the backward pass of state-space or similar attention-like mechanisms, computing the gradient `d` with respect to the sequence dimension. The `BLOCK_R` should be tuned based on the specific H dimension to maximize vector load width.

<!-- okf:related:start -->

# 相关

- 相关主题: [Memory Access Optimization: fused residual-add RMSNorm with in-place writeback](../memory-sram/exp_fused_rms_norm_001.md) — 主题相关：共享 reduction-operations
- 相关主题: [Memory Access Optimization: fused split and dual-norm single-pass load](../memory-sram/exp_fused_split_qk_norm_001.md) — 主题相关：共享 reduction-operations
- 相关主题: [Memory Access Optimization: fused add-RMSNorm with weight hoisting](../memory-sram/exp_gemma_rmsnorm_001.md) — 主题相关：共享 reduction-operations
- 相关主题: [Memory Access Optimization: sub blocking strategy](../memory-sram/exp_gn_wb_backward_001.md) — 主题相关：共享 reduction-operations
- 相关主题: [Compute Optimization: online softmax algorithm](exp_attn_reduce_stage2_001.md) — 主题相关：共享 aicore 单元利用/scalar 削减, reduction-operations
- 相关主题: [Compute Optimization: vectorization alignment](exp_backward_sum_001.md) — 主题相关：共享 aicore 单元利用/scalar 削减, reduction, reduction-operations
- 相关主题: [Compute Optimization: numerical stability](exp_celoss_001.md) — 主题相关：共享 aicore 单元利用/scalar 削减, reduction-operations
- 相关主题: [Compute Optimization: two pass algorithm](exp_celoss_bwd_001.md) — 主题相关：共享 aicore 单元利用/scalar 削减, reduction-operations
- 相关主题: [Compute Optimization: numerical stability](exp_celoss_smooth_001.md) — 主题相关：共享 aicore 单元利用/scalar 削减, reduction-operations
- 相关主题: [Compute Optimization: two pass algorithm](exp_celoss_smooth_bwd_001.md) — 主题相关：共享 aicore 单元利用/scalar 削减, reduction-operations
- 相关主题: [Compute Optimization: chunked scan strategy](exp_chunk_cumsum_001.md) — 主题相关：共享 aicore 单元利用/scalar 削减, reduction-operations
- 相关主题: [Compute Optimization: online softmax algorithm](exp_crossentropy_001.md) — 主题相关：共享 aicore 单元利用/scalar 削减, reduction-operations
- 相关主题: [Compute Optimization: grid stride scheduling](exp_crossentropy_backward_001.md) — 主题相关：共享 aicore 单元利用/scalar 削减, reduction-operations
- 相关主题: [Compute Optimization: gemm based histogram](exp_digit_hist_001.md) — 主题相关：共享 aicore 单元利用/scalar 削减, reduction-operations
- 相关主题: [Compute Optimization: numerical stability reduction](exp_entropy_epilogue_001.md) — 主题相关：共享 aicore 单元利用/scalar 削减, reduction-operations
- 相关主题: [Compute Optimization: operator fusion](exp_fused_qkv_rmsnorm_rope_001.md) — 主题相关：共享 aicore 单元利用/scalar 削减, reduction-operations
- 相关主题: [Compute Optimization: operator fusion](exp_fused_rmsnorm_sigmoid_001.md) — 主题相关：共享 aicore 单元利用/scalar 削减, reduction-operations
- 相关主题: [Compute Optimization: numerical stability](exp_grpo_loss_001.md) — 主题相关：共享 aicore 单元利用/scalar 削减, reduction-operations
- 相关主题: [Compute Optimization: register accumulation](exp_hist_count_001.md) — 主题相关：共享 aicore 单元利用/scalar 削减, reduction-operations
- 相关主题: [Compute Optimization: Operator Fusion](exp_layernorm_npu_001.md) — 主题相关：共享 aicore 单元利用/scalar 削减, reduction-operations
- 相关主题: [Compute Optimization: two pass algorithm](exp_log_softmax_001.md) — 主题相关：共享 aicore 单元利用/scalar 削减, reduction-operations
- 相关主题: [Compute Optimization: memory alignment](exp_moe_wgrad_npu_001.md) — 主题相关：共享 aicore 单元利用/scalar 削减, reduction-operations
- 相关主题: [Compute Optimization: vectorized memory access](exp_reduce_split_001.md) — 主题相关：共享 aicore 单元利用/scalar 削减, reduction, reduction-operations
- 相关主题: [Compute Optimization: two pass algorithm](exp_rmsnorm_backward_npu_001.md) — 主题相关：共享 aicore 单元利用/scalar 削减, reduction-operations
- 相关主题: [Compute Optimization: dynamic tiling strategy](exp_rmsnorm_bwd_001.md) — 主题相关：共享 aicore 单元利用/scalar 削减, reduction-operations
- 相关主题: [Scheduling Optimization: grid stride scheduling](exp_rmsnorm_fwd_001.md) — 主题相关：共享 reduction-operations
- 相关主题: [Scheduling Optimization: grid stride scheduling](exp_rmsnorm_kv_001.md) — 主题相关：共享 reduction-operations
- 相关主题: [Compute Optimization: operator fusion](exp_rmsnorm_rope_001.md) — 主题相关：共享 aicore 单元利用/scalar 削减, reduction-operations
- 相关主题: [Compute Optimization: online softmax algorithm](exp_softmax_online_001.md) — 主题相关：共享 aicore 单元利用/scalar 削减, reduction-operations
- 相关主题: [Compute Optimization: kernel fusion](exp_topp_filter_001.md) — 主题相关：共享 aicore 单元利用/scalar 削减, reduction-operations

<!-- okf:related:end -->
