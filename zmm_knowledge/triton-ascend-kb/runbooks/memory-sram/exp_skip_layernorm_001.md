---
schema_version: okf.v1
kind: operator_optimization
type: optimization_runbook
source_family: runbooks
category: memory-sram
dimension: 1.1
title: "Memory Access Optimization: precision control"
description: "normalization 场景下，Memory Access Optimization: precision control，收益 30%"
tags: [访存效率, reduction-operations, normalization]
confidence: verified
status: verified
created_at: 2026-08-21T00:00:00Z
updated_at: 2026-08-21T00:00:00Z
---

## Memory Access Optimization: precision control

【优化维度】：访存效率
【泛化层级】：条件性（迁移条件：硬件 910b2/910b3/91093/91095，fp16/bf16/fp32，N ≥1024；失效边界：N 小于片上寄存器/UB 容量时单遍融合更优，两遍法多一轮 X/R 重读反而吃亏；blockSize 未对齐 L1/UB 容量时收益回落，需重新调参）

### 原则
- **precision control**：Accumulators for the sum (sum_x) and sum of squares (sum_sq) are explicitly computed in float32. This prevents overflow and ensures numerical stability during the variance calculation, which is critical when inputs are fp16 or bf16.
- **kernel fusion**：The residual connection (X + R) and the affine transformation (W * x_hat + B) are fused directly into the LayerNorm kernel. This eliminates the need for separate kernel launches and intermediate global memory storage, significantly reducing memory bandwidth pressure.
- **two pass algorithm**：The kernel utilizes a two-pass strategy the first pass iterates over the data in blocks to compute mean and variance (reduction), and the second pass applies the normalization and affine transform. This approach decouples the reduction phase from the element-wise phase, allowing the kernel to handle hidden dimensions (N) larger than the register file capacity by processing data in blockSize chunks.

### 代码/骨架出处
- `TritonPath`:
```python
raw_experiences/extracted_kernels/118-skip_layer_norm_kernel.py
```

**坏实践（反例）**：fp16/bf16 输入直接用输入精度累加 sum 与 sum_sq
```python
sum_x  += x_blk          # fp16 累加，方差计算溢出/精度崩塌
sum_sq += x_blk * x_blk
```
后果：大方差输入下 sum_sq 溢出或精度损失，归一化结果数值错误。✅ 正解：累加器显式声明 fp32，仅在归一化输出的 element-wise 阶段回到输入精度。另外残差相加与仿射变换若拆成独立 kernel，中间结果会多一轮 GM 往返——应融合进归一化 kernel。

**已知实例**：normalization §raw (`experiences/normalization/exp_skip_layernorm_001.yaml`)
- 声称性能收益：30%
- 备注：The two-pass approach trades off some memory bandwidth (reloading X and R in the second pass) for flexibility in handling large N dimensions. Ensure blockSize is tuned to the L1 cache size for optimal performance.

<!-- okf:related:start -->

# 相关

- 相关主题: [fp32 精度契约（长链复合算子低精度 verify 不可达的破解模式）](../algorithm-numerics/fp32-precision-contract.md) — 同属优化知识，主题相关
- 相关主题: [Compute Optimization: mixed precision compute](../compute-pipeline/exp_attn_bwd_q_001.md) — 同属优化知识，主题相关
- 相关主题: [Scheduling Optimization: compilation specialization control](../compute-pipeline/exp_expand_serial_001.md) — 同属优化知识，主题相关
- 相关主题: [Compute Optimization: mixed precision compute](../compute-pipeline/exp_flashattn_bwd_kv_001.md) — 同属优化知识，主题相关
- 相关主题: [Compute Optimization: high precision accumulation](../compute-pipeline/exp_recompute_w_u_001.md) — 同属优化知识，主题相关
- 相关主题: [Scheduling Optimization: mixed precision computation](../compute-pipeline/exp_swiglu_fwd_001.md) — 同属优化知识，主题相关
- 相关主题: [tl.dot vs torch.matmul 精度差异诊断](../meta-skills/dot-vs-matmul-precision-diagnosis.md) — 同属优化知识，主题相关
- 相关主题: [量化融合算子 int8 输出精度不一致的逐层下钻诊断链](../meta-skills/quant-fusion-int8-precision-diagnosis.md) — 同属优化知识，主题相关
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
- 相关主题: [Memory Access Optimization: 2D block tiling with partial-sum reduction](exp_rmsnorm_grad_dw_001.md) — 主题相关：共享 normalization, reduction-operations, 访存效率
- 相关主题: [Memory Access Optimization: no-tiling single-load for small rows](exp_rmsnorm_notiling_001.md) — 主题相关：共享 normalization, reduction-operations, 访存效率
- 相关主题: [Memory Access Optimization: column blocking for UB management](exp_rmsnorm_npu_001.md) — 主题相关：共享 normalization, reduction-operations, 访存效率
- 相关主题: [Memory Access Optimization: column tiling for wide rows](exp_rmsnorm_tiled_001.md) — 主题相关：共享 normalization, reduction-operations, 访存效率
- 相关主题: [Memory Access Optimization: multibuffer software pipelining](exp_sdpa_bwd_reduce_001.md) — 主题相关：共享 reduction-operations, 访存效率
- 相关主题: [Memory Access Optimization: vectorized chunking](exp_speculative_recovery_001.md) — 主题相关：共享 reduction-operations, 访存效率
- 相关主题: [Memory Access Optimization: block size tuning](exp_sum_scale_001.md) — 主题相关：共享 reduction-operations, 访存效率

<!-- okf:related:end -->
