---
schema_version: okf.v1
kind: operator_optimization
type: optimization_runbook
source_family: runbooks
category: compute-pipeline
dimension: 2.1
title: 'Compute Optimization: software pipelining'
description: 'activation 场景下，Compute Optimization: software pipelining，收益 35%'
tags: [AICore 单元利用/Scalar 削减, activation-operations, activation]
confidence: verified
status: verified
created_at: 2026-08-21T00:00:00Z
updated_at: 2026-08-21T00:00:00Z
---

## Compute Optimization: software pipelining

【优化维度】：AICore 单元利用/Scalar 削减
【泛化层级】：条件性（迁移条件：硬件 910b2, 910b3, 91093, 91095；失效边界：假设输入为 flat 1D 连续布局（备注自述），非连续不适用；num_stages 过深超出 UB 预算；memory-bound 场景收益最大，compute-bound 时流水掩盖无空间）

### 原则
- **software pipelining**：Utilizes `tl.range` with the `num_stages` parameter to implement software pipelining. This overlaps memory load operations (fetching dc, a, b) with arithmetic computations (sigmoid, multiplication), effectively hiding global memory latency.
- **kernel fusion**：Fuses the entire backward pass logic for the SwiGLU activation into a single kernel. It computes gradients `da` and `db` directly without writing intermediate results (like `sig_a` or `silu_a`) back to global memory, significantly reducing memory bandwidth pressure.
- **precision management**：Casts input tensors to `tl.float32` before performing the derivative calculations. This ensures numerical stability during the computation of the SiLU derivative term (`silu_a * (1.0 - sig_a) + sig_a`), which can be sensitive to precision loss in lower precision formats.

### 代码/骨架出处
- `TritonPath`:
```python
raw_experiences/extracted_kernels/106-_swiglu_backward_kernel_flat.py
```

**坏实践（反例）**：反向中间量（`sig_a`/`silu_a`）写回 HBM，da/db 另起 kernel → 中间结果往返 → ✅ 正解即本条：单 kernel 融合全反向逻辑 + `tl.range(num_stages=...)` 流水掩盖 load + fp32 算导数。

**已知实例**：activation §raw (`experiences/activation/exp_swiglu_bwd_flat_001.yaml`)
- 声称性能收益：35%
- 备注：This kernel assumes a flat 1D memory layout for the input tensors. It is optimized for scenarios where the activation layer is memory-bound.

<!-- okf:related:start -->

# 相关

- 实践案例: [FusedBiasLeakyRelu（偏置+LeakyReLU+缩放融合）Ascend C → Triton 语义转化卡](../../ops/fused-bias-leaky-relu.md) — 主题相关：共享 activation
- 相关主题: [Memory Access Optimization: operator fusion](../memory-sram/exp_fused_add_layernorm_001.md) — 主题相关：共享 activation-operations
- 相关主题: [Memory Access Optimization: operator fusion](../memory-sram/exp_fused_norm_rope_001.md) — 主题相关：共享 activation-operations
- 相关主题: [Memory Access Optimization: grid stride loop](../memory-sram/exp_geglu_bwd_001.md) — 主题相关：共享 activation, activation-operations
- 相关主题: [Memory Access Optimization: software pipelining](../memory-sram/exp_moe_scatter_001.md) — 同属优化知识，主题相关
- 相关主题: [Memory Access Optimization: cos sin reuse](../memory-sram/exp_rope_npu_001.md) — 主题相关：共享 activation-operations
- 相关主题: [Memory Access Optimization: multibuffer software pipelining](../memory-sram/exp_sdpa_bwd_reduce_001.md) — 同属优化知识，主题相关
- 相关主题: [Memory Access Optimization: grid stride loop](../memory-sram/exp_swiglu_flat_001.md) — 主题相关：共享 activation, activation-operations
- 相关主题: [Memory Access Optimization: software pipelining](../memory-sram/exp_tv_distance_001.md) — 同属优化知识，主题相关
- 相关主题: [Scheduling Optimization: dual task pipelining](exp_attn_ascend_dual_pipeline_001.md) — 同属优化知识，主题相关
- 相关主题: [Compute Optimization: operator fusion](exp_fused_gating_001.md) — 主题相关：共享 activation, activation-operations, aicore 单元利用/scalar 削减
- 相关主题: [Scheduling Optimization: grid stride loop](exp_geglu_flat_001.md) — 主题相关：共享 activation, activation-operations
- 相关主题: [Scheduling Optimization: autotune block m](exp_gelu_bwd_001.md) — 主题相关：共享 activation, activation-operations
- 相关主题: [Scheduling Optimization: grid stride loop](exp_gelu_fwd_001.md) — 主题相关：共享 activation, activation-operations
- 相关主题: [Compute Optimization: two pass algorithm](exp_log_softmax_backward_001.md) — 主题相关：共享 activation-operations, aicore 单元利用/scalar 削减
- 相关主题: [Compute Optimization: vectorized memory access](exp_partial_rope_001.md) — 主题相关：共享 activation-operations, aicore 单元利用/scalar 削减
- 相关主题: [Compute Optimization: fused mrope logic](exp_qwen2vl_mrope_001.md) — 主题相关：共享 activation-operations, aicore 单元利用/scalar 削减
- 相关主题: [Compute Optimization: cos sin reuse](exp_rope_001.md) — 主题相关：共享 activation-operations, aicore 单元利用/scalar 削减
- 相关主题: [Scheduling Optimization: grid stride loop](exp_rope_backward_001.md) — 主题相关：共享 activation-operations
- 相关主题: [Compute Optimization: kernel fusion](exp_rope_mqa_001.md) — 主题相关：共享 activation-operations, aicore 单元利用/scalar 削减
- 相关主题: [Compute Optimization: vectorized memory access](exp_rope_triton_001.md) — 主题相关：共享 activation-operations, aicore 单元利用/scalar 削减
- 相关主题: [Scheduling Optimization: grid stride scheduling](exp_sigmoid_grid_stride_001.md) — 主题相关：共享 activation, activation-operations
- 相关主题: [Scheduling Optimization: autotuning strategy](exp_silu_bwd_001.md) — 主题相关：共享 activation, activation-operations
- 相关主题: [Scheduling Optimization: grid stride loop](exp_silu_fwd_001.md) — 主题相关：共享 activation, activation-operations
- 相关主题: [Compute Optimization: vectorized accumulation strategy](exp_softmax_inner_002.md) — 主题相关：共享 activation-operations, aicore 单元利用/scalar 削减
- 相关主题: [Compute Optimization: online softmax algorithm](exp_softmax_non_inner_001.md) — 主题相关：共享 activation-operations, aicore 单元利用/scalar 削减
- 相关主题: [Compute Optimization: activation recomputation](exp_swiglu_bwd_001.md) — 主题相关：共享 activation, activation-operations, aicore 单元利用/scalar 削减
- 相关主题: [Scheduling Optimization: mixed precision computation](exp_swiglu_fwd_001.md) — 主题相关：共享 activation, activation-operations
- 相关主题: [Compute Optimization: interleaved vectorization](exp_swiglu_oai_001.md) — 主题相关：共享 activation, activation-operations, aicore 单元利用/scalar 削减
- 相关主题: [Compute Optimization: operator fusion](exp_swiglu_quant_001.md) — 主题相关：共享 activation, activation-operations, aicore 单元利用/scalar 削减
- 相关主题: [Compute Optimization: vectorized chunked processing](exp_triton_rope_001.md) — 主题相关：共享 activation-operations, aicore 单元利用/scalar 削减
- 相关主题: [算子融合模式与约束决策](operator-fusion-patterns.md) — 主题相关：共享 activation

<!-- okf:related:end -->
