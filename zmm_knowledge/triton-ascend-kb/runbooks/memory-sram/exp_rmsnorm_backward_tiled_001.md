---
schema_version: okf.v1
kind: operator_optimization
type: optimization_runbook
source_family: runbooks
category: memory-sram
dimension: 1.1
title: "Memory Access Optimization: backward column tiling with conflict-free dW accumulation"
description: "normalization 场景下，Memory Access Optimization: backward column tiling with conflict-free dW accumulation，收益 25%"
tags: [访存效率, reduction-operations, normalization]
confidence: verified
status: verified
created_at: 2026-08-21T00:00:00Z
updated_at: 2026-08-21T00:00:00Z
---

## Memory Access Optimization: backward column tiling with conflict-free dW accumulation

【优化维度】：访存效率
【泛化层级】：条件性（迁移条件：硬件 910b2/910b3，fp16/bf16，n_cols >= 2048 的归一化反向场景；失效边界：n_cols < 2048 时单行可整块驻留 UB，tiling 的循环与边界开销抵消收益，宜改用非 tiling 整行方案；UB 容量约束：BLOCK_SIZE × dtype 字节数 × 驻留份数（输入/梯度/掩码等）≤ UB 预算，超出需下调 BLOCK_SIZE；另需 GM 侧预分配 (num_programs, n_cols) 的部分和 buffer，GM 紧张时不适用）

### 原则
- **tiling strategy**：Implements column-wise tiling using a configurable blockSize to process large feature dimensions (n_cols >= 2048) in chunks. This prevents Unified Buffer (UB) overflow on NPU hardware and ensures data remains in high-speed memory during the two-pass computation.
- **two pass algorithm**：Decouples the gradient computation into two distinct phases a reduction pass to calculate the sum(m * X) statistic required for the correction factor, and a subsequent pass to compute and store the actual dX and dW gradients. This minimizes register pressure and simplifies control flow.
- **software pipelining**：Utilizes the `num_stages` parameter within the grid-stride loop (`tl.range`) to enable software pipelining. This overlaps memory load operations with arithmetic computations, effectively hiding memory latency and keeping the compute pipeline saturated.
- **conflict free accumulation**：Avoids expensive atomic operations for weight gradient (dW) accumulation by allocating a per-program ID (pid) buffer. Each program instance writes to a unique slice of memory (`dW_ptr + pid * stride`), eliminating write conflicts and serialization, deferring the final reduction to a separate, highly efficient step.

### 代码/骨架出处
- `TritonPath`:
```python
raw_experiences/extracted_kernels/103-_rms_norm_backward_kernel_tiled.py
```

**坏实践（反例）**：dW 用 `tl.atomic_add` 逐行直接累加到全局输出
```python
for row in rows_of_this_program:
    dw_partial = compute_dw(row)
    tl.atomic_add(dW_ptr + col_offsets, dw_partial)  # 每行一次原子加
```
后果：多个 program 对同一 dW 行并发原子加，产生严重写冲突与串行化，核间相互等待。✅ 正解：按 pid 分配互不重叠的切片 `dW_ptr + pid * n_cols`，各 program 无冲突写部分和，最终由单独的归约步骤聚合成 dW（收益 25%）。

**已知实例**：normalization §raw (`experiences/normalization/exp_rmsnorm_backward_tiled_001.yaml`)
- 声称性能收益：25%
- 备注：The kernel requires `dW_ptr` to be pre-allocated with a shape of `(num_programs, n_cols)` to store intermediate partial gradients. A host-side or subsequent kernel reduction is necessary to aggregate these partials into the final dW. 支持多种模型特定的精度转换（cast）模式，用于归约阶段升 fp32、存储阶段降回原精度。

<!-- okf:related:start -->

# 相关

- 实践案例: [BlockSparseAttentionGrad (block-sparse backward) on Ascend NPU](../../ops/block-sparse-attention-grad.md) — 同属优化知识，主题相关
- 开发指南: [Embedding / 索引拷贝算子优化](../../reference/op-class-embedding.md) — 同属优化知识，主题相关
- 开发指南: [RoPE（旋转位置编码）算子优化](../../reference/op-class-rope.md) — 同属优化知识，主题相关
- 相关主题: [大矩阵 matmul 用对角 core mapping 消除 L2 bank conflict](../compute-pipeline/diagonal-core-mapping-l2-bank-conflict.md) — 同属优化知识，主题相关
- 相关主题: [Compute Optimization: register accumulation](../compute-pipeline/exp_hist_count_001.md) — 主题相关：共享 reduction-operations
- 相关主题: [Compute Optimization: high precision accumulation](../compute-pipeline/exp_recompute_w_u_001.md) — 同属优化知识，主题相关
- 相关主题: [Compute Optimization: vectorized accumulation strategy](../compute-pipeline/exp_softmax_inner_002.md) — 同属优化知识，主题相关
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
- 相关主题: [Memory Access Optimization: dynamic block sizing with fused backward+add](exp_rmsnorm_bwd_fused_001.md) — 主题相关：共享 normalization, reduction-operations, 访存效率
- 相关主题: [Memory Access Optimization: two-pass reduction for large columns](exp_rmsnorm_bwd_large_cols_001.md) — 主题相关：共享 normalization, reduction-operations, 访存效率
- 相关主题: [Memory Access Optimization: 2D vector loading for small hidden dims](exp_rmsnorm_fused_001.md) — 主题相关：共享 normalization, reduction-operations, 访存效率
- 相关主题: [Memory Access Optimization: fusing add/norm/bias/quant into one kernel](exp_rmsnorm_fused_quant_001.md) — 主题相关：共享 normalization, reduction-operations, 访存效率
- 相关主题: [Memory Access Optimization: 2D block tiling with partial-sum reduction](exp_rmsnorm_grad_dw_001.md) — 主题相关：共享 normalization, reduction-operations, 访存效率
- 相关主题: [Memory Access Optimization: no-tiling single-load for small rows](exp_rmsnorm_notiling_001.md) — 主题相关：共享 normalization, reduction-operations, 访存效率
- 相关主题: [Memory Access Optimization: column blocking for UB management](exp_rmsnorm_npu_001.md) — 主题相关：共享 normalization, reduction-operations, 访存效率
- 相关主题: [Memory Access Optimization: column tiling for wide rows](exp_rmsnorm_tiled_001.md) — 主题相关：共享 normalization, reduction-operations, 访存效率
- 相关主题: [Memory Access Optimization: multibuffer software pipelining](exp_sdpa_bwd_reduce_001.md) — 主题相关：共享 reduction-operations, 访存效率
- 相关主题: [Memory Access Optimization: precision control](exp_skip_layernorm_001.md) — 主题相关：共享 normalization, reduction-operations, 访存效率
- 相关主题: [Memory Access Optimization: vectorized chunking](exp_speculative_recovery_001.md) — 主题相关：共享 reduction-operations, 访存效率
- 相关主题: [Memory Access Optimization: block size tuning](exp_sum_scale_001.md) — 主题相关：共享 reduction-operations, 访存效率
- 相关主题: [Memory Access Optimization: software pipelining](exp_tv_distance_001.md) — 主题相关：共享 reduction-operations, 访存效率

<!-- okf:related:end -->
