---
schema_version: okf.v1
kind: operator_optimization
type: optimization_runbook
source_family: runbooks
category: compute-pipeline
dimension: 2.1
title: 'Compute Optimization: npu simd vectorization'
description: 'normalization 场景下，Compute Optimization: npu simd vectorization，收益 35%'
tags: [AICore 单元利用/Scalar 削减, normalization-operations, normalization]
confidence: verified
status: verified
created_at: 2026-08-21T00:00:00Z
updated_at: 2026-08-21T00:00:00Z
---

## Compute Optimization: npu simd vectorization

【优化维度】：AICore 单元利用/Scalar 削减
【泛化层级】：条件性（迁移条件：硬件 910b2, 910b3；失效边界：N < 128 时向量化块配置需重调；gating 与 norm 的顺序（NORM_BEFORE_GATE）必须与模型语义一致，顺序错即结果错；可选输入的 heuristics 特化组合过多时编译产物膨胀）

### 原则
- **npu simd vectorization**：The kernel is explicitly optimized for NPU SIMD units. By utilizing vectorized loads and arithmetic operations across the blockN dimension, it maximizes the throughput of the vector processing pipeline on the Ascend hardware.
- **operator fusion**：Fuses the Gating (Swish z * sigmoid(z)) operation with the LayerNorm/RMSNorm computation. This is performed either before or after normalization based on the NORM_BEFORE_GATE flag, eliminating the need to write intermediate results to global memory (HBM), thus significantly reducing memory bandwidth consumption.
- **heuristic specialization**：Utilizes @triton.heuristics to generate specialized code paths for optional inputs (Bias B and residual Z). This ensures that the compiled kernel contains only the necessary instructions for the specific configuration, avoiding runtime branching overhead for unused features.
- **float32 accumulation**：Performs statistical calculations (mean and variance) in float32 precision before casting back to the input type. This ensures numerical stability during the reduction operations, which is critical for training convergence.

### 代码/骨架出处
- `TritonPath`:
```python
raw_experiences/extracted_kernels/180-_layer_norm_fwd_1pass_kernel_npu_simd.py
```

**坏实践（反例）**：门控（z*sigmoid(z)）与归一化拆成两个 kernel → 中间结果落 HBM 再读回，带宽翻倍 → ✅ 正解即本条：融合 + `@triton.heuristics` 按 Bias/Residual 有无裁剪代码路径 + fp32 统计量。

**已知实例**：normalization §raw (`experiences/normalization/exp_layernorm_npu_simd_001.yaml`)
- 声称性能收益：35%
- 备注：Supports both LayerNorm and RMSNorm, as well as flexible gating mechanisms (pre- or post-normalization). Best suited for transformer architectures requiring fused normalization and activation steps.

<!-- okf:related:start -->

# 相关

- 开发指南: [Elementwise 算子优化](../../reference/op-class-elementwise.md) — 同属优化知识，主题相关
- 相关主题: [Memory Access Optimization: Tiled Two-Pass Reduction](../memory-sram/exp_gn_backward_001.md) — 主题相关：共享 normalization, normalization-operations
- 相关主题: [Memory Access Optimization: dynamic vectorization tuning](../memory-sram/exp_groupnorm_001.md) — 主题相关：共享 normalization, normalization-operations
- 相关主题: [Memory Access Optimization: adaptive tiling strategy](../memory-sram/exp_layernorm_bwd_001.md) — 主题相关：共享 normalization, normalization-operations
- 相关主题: [Memory Access Optimization: head vectorization](../memory-sram/exp_paged_kv_cache_001.md) — 同属优化知识，主题相关
- 相关主题: [Memory Access Optimization: single-pass tiling via UB residency](../memory-sram/exp_rmsnorm_grad_dx_001.md) — 主题相关：共享 normalization, normalization-operations
- 相关主题: [Memory Access Optimization: grid-stride scheduling with two-pass reduction](../memory-sram/exp_rmsnorm_infer_001.md) — 主题相关：共享 normalization, normalization-operations
- 相关主题: [Compute Optimization: mixed precision compute](exp_attn_bwd_q_001.md) — 主题相关：共享 aicore 单元利用/scalar 削减
- 相关主题: [Compute Optimization: blocking strategy](exp_attn_qk_cube_001.md) — 主题相关：共享 aicore 单元利用/scalar 削减
- 相关主题: [Compute Optimization: online softmax algorithm](exp_attn_reduce_stage2_001.md) — 主题相关：共享 aicore 单元利用/scalar 削减
- 相关主题: [Compute Optimization: attention sinks mechanism](exp_attn_sinks_prefill_001.md) — 主题相关：共享 aicore 单元利用/scalar 削减
- 相关主题: [Compute Optimization: vectorization alignment](exp_backward_sum_001.md) — 主题相关：共享 aicore 单元利用/scalar 削减
- 相关主题: [Compute Optimization: block ptr tiling](exp_block_inverse_001.md) — 主题相关：共享 aicore 单元利用/scalar 削减
- 相关主题: [Compute Optimization: block ptr utilization](exp_block_matmul_inv_001.md) — 主题相关：共享 aicore 单元利用/scalar 削减
- 相关主题: [Compute Optimization: pointer base precompute](exp_causal_conv1d_prefill_004.md) — 主题相关：共享 aicore 单元利用/scalar 削减
- 相关主题: [Compute Optimization: conv bias silu triple fusion](exp_causal_conv1d_prefill_006.md) — 主题相关：共享 aicore 单元利用/scalar 削减
- 相关主题: [Compute Optimization: dim is multiple mask elision](exp_causal_conv1d_prefill_007.md) — 主题相关：共享 aicore 单元利用/scalar 削减
- 相关主题: [Compute Optimization: w loop unrolling with threshold](exp_causal_conv1d_update_002.md) — 主题相关：共享 aicore 单元利用/scalar 削减
- 相关主题: [Compute Optimization: dim is multiple mask elimination](exp_causal_conv1d_update_005.md) — 主题相关：共享 aicore 单元利用/scalar 削减
- 相关主题: [Compute Optimization: numerical stability](exp_celoss_001.md) — 主题相关：共享 aicore 单元利用/scalar 削减
- 相关主题: [Compute Optimization: two pass algorithm](exp_celoss_bwd_001.md) — 主题相关：共享 aicore 单元利用/scalar 削减
- 相关主题: [Compute Optimization: numerical stability](exp_celoss_smooth_001.md) — 主题相关：共享 aicore 单元利用/scalar 削减
- 相关主题: [Compute Optimization: two pass algorithm](exp_celoss_smooth_bwd_001.md) — 主题相关：共享 aicore 单元利用/scalar 削减
- 相关主题: [Compute Optimization: chunked scan strategy](exp_chunk_cumsum_001.md) — 主题相关：共享 aicore 单元利用/scalar 削减
- 相关主题: [Compute Optimization: block ptr tiling](exp_chunk_fwd_001.md) — 主题相关：共享 aicore 单元利用/scalar 削减
- 相关主题: [Compute Optimization: block ptr abstraction](exp_chunk_kkt_001.md) — 主题相关：共享 aicore 单元利用/scalar 削减
- 相关主题: [Compute Optimization: online softmax algorithm](exp_crossentropy_001.md) — 主题相关：共享 aicore 单元利用/scalar 削减
- 相关主题: [Compute Optimization: grid stride scheduling](exp_crossentropy_backward_001.md) — 主题相关：共享 aicore 单元利用/scalar 削减
- 相关主题: [Compute Optimization: scalar loop vectorization](exp_spec_decode_verify_scalar.md) — 主题相关：共享 aicore 单元利用/scalar 削减
- 相关主题: [Compute Optimization: interleaved vectorization](exp_swiglu_oai_001.md) — 主题相关：共享 aicore 单元利用/scalar 削减

<!-- okf:related:end -->
