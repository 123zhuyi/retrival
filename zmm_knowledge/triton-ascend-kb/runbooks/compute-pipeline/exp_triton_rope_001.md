---
schema_version: okf.v1
kind: operator_optimization
type: optimization_runbook
source_family: runbooks
category: compute-pipeline
dimension: 2.1
title: 'Compute Optimization: vectorized chunked processing'
description: 'rope 场景下，Compute Optimization: vectorized chunked processing，收益 25%'
tags: [AICore 单元利用/Scalar 削减, activation-operations, rope]
confidence: verified
status: verified
created_at: 2026-08-21T00:00:00Z
updated_at: 2026-08-21T00:00:00Z
---

## Compute Optimization: vectorized chunked processing

【优化维度】：AICore 单元利用/Scalar 削减
【泛化层级】：条件性（迁移条件：硬件 910b2, 910b3, 91093, 91095；失效边界：布局风格由 IS_NEOX_STYLE 编译期决定，混用需两个版本；padded 对齐偏移 + 运行时 mask 的组合要求 pad 参数与真实分配一致；rope_dim < head_dim 的 partial 场景偏移计算是正确性关键）

### 原则
- **vectorized chunked processing**：The kernel explicitly calculates offsets for the first and second halves of the head dimension (`first_half_q_offsets`, `second_half_q_offsets`) based on the `IS_NEOX_STYLE` flag. This enables loading contiguous blocks of memory for the rotation calculation, maximizing memory bandwidth utilization and allowing the compiler to generate efficient SIMD instructions for the `x1*cos - x2*sin` operations.
- **padded access with masking**：The kernel utilizes padding parameters (`pad_n_qh`, `pad_rope_dim`) to calculate static, aligned memory offsets for vectorized loads. It simultaneously applies runtime masks (`first_q_mask`, `cos_mask`) to handle cases where the actual dimensions are smaller than the padded block size. This strategy ensures memory access alignment for performance while maintaining correctness for arbitrary tensor shapes without expensive global memory re-padding.
- **loop tiling over tokens**：The kernel employs a grid-stride loop (`tl.range`) over the `num_tokens` dimension, allowing a single kernel launch to process the entire sequence. This significantly reduces kernel launch overhead compared to launching a separate kernel instance for each token or head, and improves data locality by keeping cos/sin values in registers/cache across head iterations.

### 代码/骨架出处
- `TritonPath`:
```python
raw_experiences/extracted_kernels/172-_triton_rope.py
```

**坏实践（反例）**：每个 token/head 单独 launch kernel，且用非对齐的动态偏移做向量 load → launch 开销 + 访存未对齐惩罚 → ✅ 正解即本条：`tl.range` grid-stride 覆盖整个序列 + padded 静态对齐偏移 + 运行时 mask 保正确性。

**已知实例**：rope §raw (`experiences/rope/exp_triton_rope_001.yaml`)
- 声称性能收益：25%
- 备注：Supports both chunked-style and interleaved-style RoPE layouts, as well as partial RoPE application where rope_dim < head_dim. It also handles both pre-computed cos/sin tensors and position-indexed lookups.

<!-- okf:related:start -->

# 相关

- 实践案例: [GDN 线性注意力三阶段优化案例（FLA 移植 → Triton 体系化 → AscendC 重写）](../../ops/gdn-chunked-linear-attention.md) — 同属优化知识，主题相关
- 相关主题: [Memory Access Optimization: chunked recurrence strategy](../memory-sram/exp_chunk_gated_delta_001.md) — 同属优化知识，主题相关
- 相关主题: [Memory Access Optimization: operator fusion](../memory-sram/exp_fused_add_layernorm_001.md) — 主题相关：共享 activation-operations
- 相关主题: [Memory Access Optimization: operator fusion](../memory-sram/exp_fused_norm_rope_001.md) — 主题相关：共享 activation-operations
- 相关主题: [Memory Access Optimization: large block processing](../memory-sram/exp_gather_flat_001.md) — 同属优化知识，主题相关
- 相关主题: [Memory Access Optimization: grid stride loop](../memory-sram/exp_geglu_bwd_001.md) — 主题相关：共享 activation-operations
- 相关主题: [Memory Access Optimization: cos sin reuse](../memory-sram/exp_rope_npu_001.md) — 主题相关：共享 activation-operations, rope
- 相关主题: [Memory Access Optimization: two pass reduction strategy](../memory-sram/exp_softmax_backward_001.md) — 主题相关：共享 activation-operations
- 相关主题: [Memory Access Optimization: grid stride loop](../memory-sram/exp_swiglu_flat_001.md) — 主题相关：共享 activation-operations
- 相关主题: [Compute Optimization: chunked scan strategy](exp_chunk_cumsum_001.md) — 主题相关：共享 aicore 单元利用/scalar 削减
- 相关主题: [Compute Optimization: operator fusion](exp_fused_gating_001.md) — 主题相关：共享 activation-operations, aicore 单元利用/scalar 削减
- 相关主题: [Scheduling Optimization: grid stride loop](exp_geglu_flat_001.md) — 主题相关：共享 activation-operations
- 相关主题: [Scheduling Optimization: autotune block m](exp_gelu_bwd_001.md) — 主题相关：共享 activation-operations
- 相关主题: [Scheduling Optimization: grid stride loop](exp_gelu_fwd_001.md) — 主题相关：共享 activation-operations
- 相关主题: [Compute Optimization: two pass algorithm](exp_log_softmax_backward_001.md) — 主题相关：共享 activation-operations, aicore 单元利用/scalar 削减
- 相关主题: [Compute Optimization: vectorized memory access](exp_partial_rope_001.md) — 主题相关：共享 activation-operations, aicore 单元利用/scalar 削减, rope
- 相关主题: [Compute Optimization: fused mrope logic](exp_qwen2vl_mrope_001.md) — 主题相关：共享 activation-operations, aicore 单元利用/scalar 削减, rope
- 相关主题: [Compute Optimization: cos sin reuse](exp_rope_001.md) — 主题相关：共享 activation-operations, aicore 单元利用/scalar 削减, rope
- 相关主题: [Scheduling Optimization: grid stride loop](exp_rope_backward_001.md) — 主题相关：共享 activation-operations, rope
- 相关主题: [Compute Optimization: kernel fusion](exp_rope_mqa_001.md) — 主题相关：共享 activation-operations, aicore 单元利用/scalar 削减
- 相关主题: [Compute Optimization: vectorized memory access](exp_rope_triton_001.md) — 主题相关：共享 activation-operations, aicore 单元利用/scalar 削减, rope
- 相关主题: [Scheduling Optimization: grid stride scheduling](exp_sigmoid_grid_stride_001.md) — 主题相关：共享 activation-operations
- 相关主题: [Scheduling Optimization: autotuning strategy](exp_silu_bwd_001.md) — 主题相关：共享 activation-operations
- 相关主题: [Scheduling Optimization: grid stride loop](exp_silu_fwd_001.md) — 主题相关：共享 activation-operations
- 相关主题: [Compute Optimization: vectorized accumulation strategy](exp_softmax_inner_002.md) — 主题相关：共享 activation-operations, aicore 单元利用/scalar 削减
- 相关主题: [Compute Optimization: online softmax algorithm](exp_softmax_non_inner_001.md) — 主题相关：共享 activation-operations, aicore 单元利用/scalar 削减
- 相关主题: [Compute Optimization: activation recomputation](exp_swiglu_bwd_001.md) — 主题相关：共享 activation-operations, aicore 单元利用/scalar 削减
- 相关主题: [Compute Optimization: software pipelining](exp_swiglu_bwd_flat_001.md) — 主题相关：共享 activation-operations, aicore 单元利用/scalar 削减
- 相关主题: [Scheduling Optimization: mixed precision computation](exp_swiglu_fwd_001.md) — 主题相关：共享 activation-operations
- 相关主题: [Compute Optimization: interleaved vectorization](exp_swiglu_oai_001.md) — 主题相关：共享 activation-operations, aicore 单元利用/scalar 削减
- 相关主题: [Compute Optimization: operator fusion](exp_swiglu_quant_001.md) — 主题相关：共享 activation-operations, aicore 单元利用/scalar 削减

<!-- okf:related:end -->
