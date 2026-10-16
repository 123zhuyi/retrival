---
schema_version: okf.v1
kind: operator_optimization
type: optimization_runbook
source_family: runbooks
category: memory-sram
dimension: 1.1
title: "Memory Access Optimization: 2D block tiling with partial-sum reduction"
description: "normalization 场景下，Memory Access Optimization: 2D block tiling with partial-sum reduction，收益 35%"
tags: [访存效率, reduction-operations, normalization]
confidence: verified
status: verified
created_at: 2026-08-21T00:00:00Z
updated_at: 2026-08-21T00:00:00Z
---

## Memory Access Optimization: 2D block tiling with partial-sum reduction

【优化维度】：访存效率
【泛化层级】：条件性（迁移条件：硬件 910b2/910b3/91093/91095，fp16/bf16，M >= 128、N >= 512 的权重梯度归约场景；失效边界：M 很小、row_pid 数量少时部分和 buffer 的二次归约开销占比升高，直接原子加可能更划算；UB 容量约束：ROW_BLOCK_SIZE × COL_BLOCK_SIZE × dtype 字节数 × 驻留份数 ≤ UB 预算；非连续布局依赖显式 stride 参数，stride 无法预先确知时仍需数据拷贝/转置预处理）

### 原则
- **2D Block Tiling**：The kernel utilizes a 2D tiling strategy defined by ROW_BLOCK_SIZE and COL_BLOCK_SIZE. By processing the matrix in tiles rather than full rows or columns, it maximizes data reuse within the NPU's on-chip SRAM, significantly reducing the pressure on global memory bandwidth.
- **Partial Sum Reduction**：Instead of performing atomic additions directly to the global output tensor DW for every element (which causes severe contention), the kernel computes partial sums per tile. These partial sums are stored in a temporary buffer (indexed by row_pid), allowing for a more efficient final aggregation step (e.g., a separate reduction kernel or atomic add on a much smaller dataset).
- **Stride Aware Memory Access**：The kernel explicitly handles row and column strides (x_stride_r, x_stride_c). This optimization allows the kernel to efficiently process non-contiguous memory layouts (such as transposed or sliced tensors) without requiring a pre-processing data copy or transpose operation.

### 代码/骨架出处
- `TritonPath`:
```python
raw_experiences/extracted_kernels/132-rms_norm_grad_dw_kernel.py
```

**坏实践（反例）**：每个元素直接 `tl.atomic_add` 到全局输出 DW
```python
tl.atomic_add(DW_ptr + col_offsets, dy_val * x_val)  # 每个 tile 元素一次原子加
```
后果：大量 program 对同一 DW 行并发原子加，写冲突严重、访存串行化。✅ 正解：tile 内先算部分和，按 row_pid 写各自独立的临时 buffer 切片，最终由独立归约 kernel（或对小得多的部分和集合做原子加）聚合出 DW（收益 35%）。

**已知实例**：normalization §raw (`experiences/normalization/exp_rmsnorm_grad_dw_001.yaml`)
- 声称性能收益：35%
- 备注：This kernel outputs partial sums indexed by the row program ID. A subsequent reduction step (e.g., a second kernel or atomic add) is required to aggregate these partial sums into the final DW gradient.

<!-- okf:related:start -->

# 相关

- 开发指南: [归约算子优化](../../reference/op-class-reduction.md) — 同属优化知识，主题相关
- 相关主题: [Compute Optimization: online softmax algorithm](../compute-pipeline/exp_attn_reduce_stage2_001.md) — 主题相关：共享 reduction-operations
- 相关主题: [Compute Optimization: vectorization alignment](../compute-pipeline/exp_backward_sum_001.md) — 主题相关：共享 reduction-operations
- 相关主题: [Compute Optimization: numerical stability](../compute-pipeline/exp_celoss_001.md) — 主题相关：共享 reduction-operations
- 相关主题: [Compute Optimization: two pass algorithm](../compute-pipeline/exp_celoss_bwd_001.md) — 主题相关：共享 reduction-operations
- 相关主题: [Compute Optimization: numerical stability](../compute-pipeline/exp_celoss_smooth_001.md) — 主题相关：共享 reduction-operations
- 相关主题: [Compute Optimization: two pass algorithm](../compute-pipeline/exp_celoss_smooth_bwd_001.md) — 主题相关：共享 reduction-operations
- 相关主题: [Memory Access Optimization: fused residual-add RMSNorm with in-place writeback](exp_fused_rms_norm_001.md) — 主题相关：共享 normalization, reduction-operations, 访存效率
- 相关主题: [Memory Access Optimization: fused split and dual-norm single-pass load](exp_fused_split_qk_norm_001.md) — 主题相关：共享 normalization, reduction-operations, 访存效率
- 相关主题: [Memory Access Optimization: fused add-RMSNorm with weight hoisting](exp_gemma_rmsnorm_001.md) — 主题相关：共享 reduction-operations, 访存效率
- 相关主题: [Memory Access Optimization: sub blocking strategy](exp_gn_wb_backward_001.md) — 主题相关：共享 reduction-operations, 访存效率
- 相关主题: [Memory Access Optimization: fused lse reduction](exp_grpo_bwd_001.md) — 主题相关：共享 reduction-operations, 访存效率
- 相关主题: [Memory Access Optimization: dynamic workload tiling](exp_layernorm_bwd_fused_001.md) — 主题相关：共享 normalization, reduction-operations, 访存效率
- 相关主题: [Memory Access Optimization: persistent kernel strategy](exp_layernorm_fwd_001.md) — 主题相关：共享 normalization, reduction-operations, 访存效率
- 相关主题: [Memory Access Optimization: grid stride scheduling](exp_rmsnorm_001.md) — 主题相关：共享 normalization, reduction-operations, 访存效率
- 相关主题: [Memory Access Optimization: register reuse intermediate](exp_rmsnorm_add_001.md) — 主题相关：共享 normalization, reduction-operations, 访存效率
- 相关主题: [Memory Access Optimization: no-tiling register reuse with conflict-free dW accumulation](exp_rmsnorm_backward_001.md) — 主题相关：共享 normalization, reduction-operations, 访存效率
- 相关主题: [Memory Access Optimization: backward column tiling with conflict-free dW accumulation](exp_rmsnorm_backward_tiled_001.md) — 主题相关：共享 normalization, reduction-operations, 访存效率
- 相关主题: [Memory Access Optimization: dynamic block sizing with fused backward+add](exp_rmsnorm_bwd_fused_001.md) — 主题相关：共享 normalization, reduction-operations, 访存效率
- 相关主题: [Memory Access Optimization: two-pass reduction for large columns](exp_rmsnorm_bwd_large_cols_001.md) — 主题相关：共享 normalization, reduction-operations, 访存效率
- 相关主题: [Memory Access Optimization: 2D vector loading for small hidden dims](exp_rmsnorm_fused_001.md) — 主题相关：共享 normalization, reduction-operations, 访存效率
- 相关主题: [Memory Access Optimization: fusing add/norm/bias/quant into one kernel](exp_rmsnorm_fused_quant_001.md) — 主题相关：共享 normalization, reduction-operations, 访存效率
- 相关主题: [Memory Access Optimization: no-tiling single-load for small rows](exp_rmsnorm_notiling_001.md) — 主题相关：共享 normalization, reduction-operations, 访存效率
- 相关主题: [Memory Access Optimization: column blocking for UB management](exp_rmsnorm_npu_001.md) — 主题相关：共享 normalization, reduction-operations, 访存效率
- 相关主题: [Memory Access Optimization: column tiling for wide rows](exp_rmsnorm_tiled_001.md) — 主题相关：共享 normalization, reduction-operations, 访存效率
- 相关主题: [Memory Access Optimization: multibuffer software pipelining](exp_sdpa_bwd_reduce_001.md) — 主题相关：共享 reduction-operations, 访存效率
- 相关主题: [Memory Access Optimization: precision control](exp_skip_layernorm_001.md) — 主题相关：共享 normalization, reduction-operations, 访存效率
- 相关主题: [Memory Access Optimization: vectorized chunking](exp_speculative_recovery_001.md) — 主题相关：共享 reduction-operations, 访存效率
- 相关主题: [Memory Access Optimization: block size tuning](exp_sum_scale_001.md) — 主题相关：共享 reduction-operations, 访存效率
- 相关主题: [Memory Access Optimization: software pipelining](exp_tv_distance_001.md) — 主题相关：共享 reduction-operations, 访存效率

<!-- okf:related:end -->
