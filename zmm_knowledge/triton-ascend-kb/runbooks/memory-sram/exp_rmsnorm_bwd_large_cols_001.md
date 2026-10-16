---
schema_version: okf.v1
kind: operator_optimization
type: optimization_runbook
source_family: runbooks
category: memory-sram
dimension: 1.1
title: "Memory Access Optimization: two-pass reduction for large columns"
description: "normalization 场景下，Memory Access Optimization: two-pass reduction for large columns，收益 35%"
tags: [访存效率, reduction-operations, normalization]
confidence: verified
status: verified
created_at: 2026-08-21T00:00:00Z
updated_at: 2026-08-21T00:00:00Z
---

## Memory Access Optimization: two-pass reduction for large columns

【优化维度】：访存效率
【泛化层级】：条件性（迁移条件：硬件 910b2/910b3/91093/91095，fp16/bf16/fp32，n_cols >= 4096 的归一化反向场景；失效边界：n_cols < 4096 时单 pass 寄存器压力可承受，两遍读取的额外 GM 带宽反而拖累，宜用单 pass；UB 容量约束：BLOCK_SIZE_M × BLOCK_SIZE_N × dtype 字节数 × 驻留份数 ≤ UB 预算；IS_DETERMINISTIC=False 的 atomic 路径以牺牲严格可复现性换速度，要求结果可复现的场景必须走 pid 索引 buffer 路径）

### 原则
- **two pass reduction strategy**：The kernel implements a two-pass loop over the column dimension. The first pass computes the dot product reduction (sum of dY * W * X) required for the dX calculation, storing it in `dot_product_acc`. The second pass performs the actual gradient calculation using this pre-computed value. This decouples the reduction dependency, significantly reducing register pressure and avoiding the need to keep the entire row in registers, which is critical for large column dimensions.
- **block tiling optimization**：Utilizes 2D tiling with `BLOCK_SIZE_M` and `BLOCK_SIZE_N` to process the input matrix in chunks. This ensures that data loads fit within the NPU on-chip cache hierarchy and aligns with memory transaction sizes, maximizing bandwidth utilization for the large column scenario.
- **deterministic gradient accumulation**：Provides a configurable accumulation strategy for the weight gradient (`dW`). When `IS_DETERMINISTIC` is True, it uses a program-id indexed buffer to accumulate partial sums, ensuring reproducible results at the cost of a final reduction step (not shown). When False, it uses `tl.atomic_add` for maximum performance, trading off strict reproducibility for speed.
- **architecture specific casting**：Includes specialized casting paths to handle data type casting (e.g., fp16 to fp32) at intermediate stages. This optimizes the balance between numerical stability (using higher precision for accumulation) and computational throughput (using lower precision for storage).

### 代码/骨架出处
- `TritonPath`:
```python
raw_experiences/extracted_kernels/234-_rmsnorm_bwd_large_cols_kernel.py
```

**坏实践（反例）**：naive 单 pass——整行（n_cols >= 4096）数据与归约中间量同时驻留寄存器
```python
x_row = load_full_row()                 # n_cols=4096+ 整行驻留
dot = sum(dY * W * x_row)               # 归约与行数据同驻寄存器
dX = f(dY, W, x_row, dot)               # 寄存器压力拉满
```
后果：整行加中间量超出寄存器容量，触发 register spilling，性能崩塌。✅ 正解：两遍法——第一遍只算 `dot_product_acc`，第二遍重载数据算梯度，寄存器任意时刻只驻留 BLOCK_SIZE_M × BLOCK_SIZE_N 的 tile（收益 35%）。

**已知实例**：normalization §raw (`experiences/normalization/exp_rmsnorm_bwd_large_cols_001.yaml`)
- 声称性能收益：35%
- 备注：This kernel is specifically designed for normalization layers where the hidden dimension (n_cols) is large (e.g., 4096+). The two-pass approach prevents register spilling that would occur in a naive single-pass implementation.

<!-- okf:related:start -->

# 相关

- 相关主题: [跨 tile 归约必须显式两阶段 reduce-then-apply](../algorithm-numerics/two-pass-reduce-then-apply.md) — 同属优化知识，主题相关
- 相关主题: [Scheduling Optimization: two dimensional grid strategy](../compute-pipeline/exp_causal_conv1d_prefill_003.md) — 同属优化知识，主题相关
- 相关主题: [Compute Optimization: two pass algorithm](../compute-pipeline/exp_celoss_bwd_001.md) — 主题相关：共享 reduction-operations
- 相关主题: [Compute Optimization: two pass algorithm](../compute-pipeline/exp_celoss_smooth_bwd_001.md) — 主题相关：共享 reduction-operations
- 相关主题: [Compute Optimization: two pass algorithm](../compute-pipeline/exp_log_softmax_001.md) — 主题相关：共享 reduction-operations
- 相关主题: [Compute Optimization: two pass algorithm](../compute-pipeline/exp_log_softmax_backward_001.md) — 同属优化知识，主题相关
- 相关主题: [Compute Optimization: two pass algorithm](../compute-pipeline/exp_rmsnorm_backward_npu_001.md) — 主题相关：共享 normalization, reduction-operations
- 相关主题: [M>=256 时 BLOCK_M=256（总 block-iter 减半）](blockm-256-large-m.md) — 同属优化知识，主题相关
- 相关主题: [Memory Access Optimization: fused residual-add RMSNorm with in-place writeback](exp_fused_rms_norm_001.md) — 主题相关：共享 normalization, reduction-operations, 访存效率
- 相关主题: [Memory Access Optimization: fused split and dual-norm single-pass load](exp_fused_split_qk_norm_001.md) — 主题相关：共享 normalization, reduction-operations, 访存效率
- 相关主题: [Memory Access Optimization: large block processing](exp_gather_flat_001.md) — 主题相关：共享 访存效率
- 相关主题: [Memory Access Optimization: fused add-RMSNorm with weight hoisting](exp_gemma_rmsnorm_001.md) — 主题相关：共享 reduction-operations, 访存效率
- 相关主题: [Memory Access Optimization: Tiled Two-Pass Reduction](exp_gn_backward_001.md) — 主题相关：共享 normalization, 访存效率
- 相关主题: [Memory Access Optimization: sub blocking strategy](exp_gn_wb_backward_001.md) — 主题相关：共享 reduction-operations, 访存效率
- 相关主题: [Memory Access Optimization: fused lse reduction](exp_grpo_bwd_001.md) — 主题相关：共享 reduction-operations, 访存效率
- 相关主题: [Memory Access Optimization: dynamic workload tiling](exp_layernorm_bwd_fused_001.md) — 主题相关：共享 normalization, reduction-operations, 访存效率
- 相关主题: [Memory Access Optimization: persistent kernel strategy](exp_layernorm_fwd_001.md) — 主题相关：共享 normalization, reduction-operations, 访存效率
- 相关主题: [Memory Access Optimization: grid stride scheduling](exp_rmsnorm_001.md) — 主题相关：共享 normalization, reduction-operations, 访存效率
- 相关主题: [Memory Access Optimization: register reuse intermediate](exp_rmsnorm_add_001.md) — 主题相关：共享 normalization, reduction-operations, 访存效率
- 相关主题: [Memory Access Optimization: no-tiling register reuse with conflict-free dW accumulation](exp_rmsnorm_backward_001.md) — 主题相关：共享 normalization, reduction-operations, 访存效率
- 相关主题: [Memory Access Optimization: backward column tiling with conflict-free dW accumulation](exp_rmsnorm_backward_tiled_001.md) — 主题相关：共享 normalization, reduction-operations, 访存效率
- 相关主题: [Memory Access Optimization: dynamic block sizing with fused backward+add](exp_rmsnorm_bwd_fused_001.md) — 主题相关：共享 normalization, reduction-operations, 访存效率
- 相关主题: [Memory Access Optimization: 2D vector loading for small hidden dims](exp_rmsnorm_fused_001.md) — 主题相关：共享 normalization, reduction-operations, 访存效率
- 相关主题: [Memory Access Optimization: fusing add/norm/bias/quant into one kernel](exp_rmsnorm_fused_quant_001.md) — 主题相关：共享 normalization, reduction-operations, 访存效率
- 相关主题: [Memory Access Optimization: 2D block tiling with partial-sum reduction](exp_rmsnorm_grad_dw_001.md) — 主题相关：共享 normalization, reduction-operations, 访存效率
- 相关主题: [Memory Access Optimization: single-pass tiling via UB residency](exp_rmsnorm_grad_dx_001.md) — 主题相关：共享 normalization, 访存效率
- 相关主题: [Memory Access Optimization: grid-stride scheduling with two-pass reduction](exp_rmsnorm_infer_001.md) — 主题相关：共享 normalization, 访存效率
- 相关主题: [Memory Access Optimization: no-tiling single-load for small rows](exp_rmsnorm_notiling_001.md) — 主题相关：共享 normalization, reduction-operations, 访存效率
- 相关主题: [Memory Access Optimization: column blocking for UB management](exp_rmsnorm_npu_001.md) — 主题相关：共享 normalization, reduction-operations, 访存效率
- 相关主题: [Memory Access Optimization: column tiling for wide rows](exp_rmsnorm_tiled_001.md) — 主题相关：共享 normalization, reduction-operations, 访存效率
- 相关主题: [Memory Access Optimization: multibuffer software pipelining](exp_sdpa_bwd_reduce_001.md) — 主题相关：共享 reduction-operations, 访存效率
- 相关主题: [Memory Access Optimization: precision control](exp_skip_layernorm_001.md) — 主题相关：共享 normalization, reduction-operations, 访存效率
- 相关主题: [Memory Access Optimization: two pass reduction strategy](exp_softmax_backward_001.md) — 主题相关：共享 访存效率
- 相关主题: [Memory Access Optimization: two pass reduction strategy](exp_softmax_backward_non_inner_001.md) — 主题相关：共享 访存效率
- 相关主题: [Memory Access Optimization: vectorized chunking](exp_speculative_recovery_001.md) — 主题相关：共享 reduction-operations, 访存效率
- 相关主题: [Memory Access Optimization: block size tuning](exp_sum_scale_001.md) — 主题相关：共享 reduction-operations, 访存效率

<!-- okf:related:end -->
