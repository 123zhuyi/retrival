---
schema_version: okf.v1
kind: operator_optimization
type: optimization_runbook
source_family: runbooks
category: memory-sram
dimension: 1.1
title: "Memory Access Optimization: dynamic workload tiling"
description: "normalization 场景下，Memory Access Optimization: dynamic workload tiling，收益 35%"
tags: [访存效率, reduction-operations, normalization]
confidence: verified
status: verified
created_at: 2026-08-21T00:00:00Z
updated_at: 2026-08-21T00:00:00Z
---

## Memory Access Optimization: dynamic workload tiling

【优化维度】：访存效率
【泛化层级】：条件性（迁移条件：硬件 910b2, 910b3，fp16/bf16/fp32，n_rows/n_cols ≥1；失效边界：grid 很小时 dW/dB 原子加争用成为瓶颈；单列数据可全量驻留寄存器时，两遍列循环重读输入的带宽开销不划算，应改单遍融合实现）

### 原则
- **dynamic workload tiling**：The heuristic `BLOCK_SIZE_M = ceil_div(4096, n_cols)` dynamically adjusts the row block size based on the hidden dimension (n_cols). This ensures that each program instance processes a consistent amount of data (~4096 elements), which optimizes hardware utilization and load balancing regardless of the matrix aspect ratio (e.g., long sequences vs. large hidden sizes).
- **in loop atomic reduction**：The kernel employs `tl.atomic_add` to accumulate gradients for weights (`dW`) and biases (`dB`) directly within the column iteration loop. This strategy avoids the need for a separate global reduction kernel or storing large intermediate partial sums, thereby reducing global memory pressure and kernel launch overhead.
- **register efficient reduction passes**：The computation is split into two distinct passes over the column dimension. The first pass computes row-wise reduction statistics (`ds_dx_hat`, `ds`) and weight gradients. The second pass computes the final input gradients (`dX`) using the statistics from the first pass. This design minimizes register pressure by avoiding the need to keep the entire intermediate `dY_W` matrix in registers, trading off slightly increased memory bandwidth (reloading inputs) for better occupancy.
- **fused residual backpropagation**：The kernel conditionally adds the residual gradient (`dS_out`) directly during the `dX` computation loop. This fuses the backward pass of the residual addition operation with the normalization backward, eliminating a separate kernel launch and reducing global memory read/write operations for the gradient tensors.

### 代码/骨架出处
- `TritonPath`:
```python
raw_experiences/extracted_kernels/251-_fused_add_layernorm_bwd_kernel.py
```

**坏实践（反例）**：单遍实现把整个中间矩阵留在寄存器跨循环复用；残差梯度另起一个 add kernel 回写
```python
dy_w = dy * w                      # 整个中间矩阵常驻寄存器，等待 dX 阶段复用
# ……期间寄存器压力撑满，触发 spill
res_add_kernel(dS_out, dX)         # 残差梯度单独 kernel 再读写一轮 GM
```
后果：中间矩阵占满寄存器导致 spill、占用率下降；残差 add 单独 kernel 多一轮 GM 读写与 launch 开销。✅ 正解：拆成两遍列循环——第一遍算行统计量与权重梯度，第二遍用统计量算 dX，寄存器只保留统计量；残差梯度在 dX 循环内按条件直接融合，省掉单独 kernel。

**已知实例**：normalization §raw (`experiences/normalization/exp_layernorm_bwd_fused_001.yaml`)
- 声称性能收益：35%
- 备注：列向双循环增加了带宽用量（重读输入与上游梯度），但这是把归约逻辑塞进寄存器限制内的必要代价。grid 很小时 dW/dB 原子操作可能争用。

<!-- okf:related:start -->

# 相关

- 相关主题: [Compute Optimization: online softmax algorithm](../compute-pipeline/exp_attn_reduce_stage2_001.md) — 主题相关：共享 reduction-operations
- 相关主题: [Compute Optimization: vectorization alignment](../compute-pipeline/exp_backward_sum_001.md) — 主题相关：共享 reduction-operations
- 相关主题: [Compute Optimization: numerical stability](../compute-pipeline/exp_celoss_001.md) — 主题相关：共享 reduction-operations
- 相关主题: [Compute Optimization: two pass algorithm](../compute-pipeline/exp_celoss_bwd_001.md) — 主题相关：共享 reduction-operations
- 相关主题: [Compute Optimization: numerical stability](../compute-pipeline/exp_celoss_smooth_001.md) — 主题相关：共享 reduction-operations
- 相关主题: [Compute Optimization: two pass algorithm](../compute-pipeline/exp_celoss_smooth_bwd_001.md) — 主题相关：共享 reduction-operations
- 相关主题: [Compute Optimization: chunked scan strategy](../compute-pipeline/exp_chunk_cumsum_001.md) — 主题相关：共享 reduction-operations
- 相关主题: [Memory Access Optimization: fused residual-add RMSNorm with in-place writeback](exp_fused_rms_norm_001.md) — 主题相关：共享 normalization, reduction-operations, 访存效率
- 相关主题: [Memory Access Optimization: fused split and dual-norm single-pass load](exp_fused_split_qk_norm_001.md) — 主题相关：共享 normalization, reduction-operations, 访存效率
- 相关主题: [Memory Access Optimization: fused add-RMSNorm with weight hoisting](exp_gemma_rmsnorm_001.md) — 主题相关：共享 reduction-operations, 访存效率
- 相关主题: [Memory Access Optimization: sub blocking strategy](exp_gn_wb_backward_001.md) — 主题相关：共享 reduction-operations, 访存效率
- 相关主题: [Memory Access Optimization: fused lse reduction](exp_grpo_bwd_001.md) — 主题相关：共享 reduction-operations, 访存效率
- 相关主题: [Memory Access Optimization: persistent kernel strategy](exp_layernorm_fwd_001.md) — 主题相关：共享 normalization, reduction-operations, 访存效率
- 相关主题: [Memory Access Optimization: grid stride scheduling](exp_rmsnorm_001.md) — 主题相关：共享 normalization, reduction-operations, 访存效率
- 相关主题: [Memory Access Optimization: register reuse intermediate](exp_rmsnorm_add_001.md) — 主题相关：共享 normalization, reduction-operations, 访存效率
- 相关主题: [Memory Access Optimization: no-tiling register reuse with conflict-free dW accumulation](exp_rmsnorm_backward_001.md) — 主题相关：共享 normalization, reduction-operations, 访存效率
- 相关主题: [Memory Access Optimization: backward column tiling with conflict-free dW accumulation](exp_rmsnorm_backward_tiled_001.md) — 主题相关：共享 normalization, reduction-operations, 访存效率
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
