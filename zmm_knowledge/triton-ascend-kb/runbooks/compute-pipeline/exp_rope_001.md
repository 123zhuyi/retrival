---
schema_version: okf.v1
kind: operator_optimization
type: optimization_runbook
source_family: runbooks
category: compute-pipeline
dimension: 2.1
title: 'Compute Optimization: cos sin reuse'
description: 'rope 场景下，Compute Optimization: cos sin reuse，收益 15%'
tags: [AICore 单元利用/Scalar 削减, activation-operations, rope]
confidence: verified
status: verified
created_at: 2026-08-21T00:00:00Z
updated_at: 2026-08-21T00:00:00Z
---

## Compute Optimization: cos sin reuse

【优化维度】：AICore 单元利用/Scalar 削减
【泛化层级】：条件性（迁移条件：硬件 910b2, 910b3, 91093, 91095；失效边界：本条是计算 helper，假设数据已载入 SRAM/寄存器（备注自述），收益依赖调用方 kernel 的加载策略；inverse constexpr 分支使编译产物翻倍，需确认正反两个 pass 都真实需要）

### 原则
- **cos sin reuse**：The COS and SIN tensors are loaded once per sequence block outside the NUM_HEADS loop. Since positional embeddings are identical across all attention heads for a given position, this strategy significantly reduces global memory bandwidth consumption by avoiding redundant loads for each head iteration.
- **persistent scheduling**：The kernel utilizes a grid-stride loop pattern (range(pid, NUM_BLOCKS, NUM_CORES)). This acts as a persistent kernel, allowing a single program instance to process multiple sequence blocks. This improves load balancing across cores and reduces kernel launch overhead for variable sequence lengths.
- **vectorized access**：The COS and SIN values are loaded in contiguous chunks (split into low/high dimensions) corresponding to the BLOCK size. This maximizes memory coalescing and utilizes the memory bus width efficiently on the NPU.
- **vectorized complex decomposition**：The complex number multiplication required for RoPE is decomposed into separate real (x1) and imaginary (x2) floating-point operations. This mapping allows the underlying hardware to utilize standard SIMD vector units efficiently, avoiding potential overhead or latency associated with native complex data type handling.
- **compile time branch pruning**：The 'inverse' flag is defined as a tl.constexpr. This forces the Triton compiler to generate specialized kernels for the forward and inverse (backward) passes during compilation. It completely eliminates runtime branch divergence within the thread blocks, ensuring all threads in a warp execute the same instruction stream.
- **fused arithmetic expression**：The computation combines multiplication and addition/subtraction into single expressions (e.g., x1 * cos_tile - x2 * sin_tile). This fusion minimizes register pressure by avoiding the storage of intermediate products, allowing for higher instruction throughput and better utilization of the NPU's compute pipelines.

### 代码/骨架出处
- `TritonPath`:
```python
_compute_rope_separated
```

**坏实践（反例）**：每个 head 迭代都重新 load 一次 cos/sin（各 head 在同一位置的值其实相同）→ 位置编码表被重复读 NUM_HEADS 次 → ✅ 正解即本条：cos/sin 在头循环外按块一次加载复用 + 复数乘拆成实/虚 SIMD 运算 + inverse 编译期裁剪。

**已知实例**：rope §raw (`experiences/rope/exp_rope_001.yaml`)
- 声称性能收益：15%
- 备注：This kernel is a compute helper that assumes data (x1, x2, sin, cos) is already loaded into SRAM/Registers. Performance gains are realized when combined with an optimized memory loading strategy in the parent kernel.

<!-- okf:related:start -->

# 相关

- 相关主题: [Memory Access Optimization: kv tile reuse](../memory-sram/exp_attn_bwd_hstu_001.md) — 同属优化知识，主题相关
- 相关主题: [Memory Access Optimization: sliding window register reuse](../memory-sram/exp_causal_conv1d_prefill_002.md) — 同属优化知识，主题相关
- 相关主题: [Memory Access Optimization: operator fusion](../memory-sram/exp_fused_add_layernorm_001.md) — 主题相关：共享 activation-operations
- 相关主题: [Memory Access Optimization: operator fusion](../memory-sram/exp_fused_norm_rope_001.md) — 主题相关：共享 activation-operations
- 相关主题: [Memory Access Optimization: grid stride loop](../memory-sram/exp_geglu_bwd_001.md) — 主题相关：共享 activation-operations
- 相关主题: [Memory Access Optimization: register reuse intermediate](../memory-sram/exp_rmsnorm_add_001.md) — 同属优化知识，主题相关
- 相关主题: [Memory Access Optimization: no-tiling register reuse with conflict-free dW accumulation](../memory-sram/exp_rmsnorm_backward_001.md) — 同属优化知识，主题相关
- 相关主题: [Memory Access Optimization: cos sin reuse](../memory-sram/exp_rope_npu_001.md) — 主题相关：共享 activation-operations, rope
- 相关主题: [Memory Access Optimization: score-tile reuse with online softmax](../memory-sram/exp_sdpa_flash_001.md) — 同属优化知识，主题相关
- 相关主题: [Memory Access Optimization: grid stride loop](../memory-sram/exp_swiglu_flat_001.md) — 主题相关：共享 activation-operations
- 相关主题: [Compute Optimization: operator fusion](exp_fused_gating_001.md) — 主题相关：共享 activation-operations, aicore 单元利用/scalar 削减
- 相关主题: [Scheduling Optimization: grid stride loop](exp_geglu_flat_001.md) — 主题相关：共享 activation-operations
- 相关主题: [Scheduling Optimization: autotune block m](exp_gelu_bwd_001.md) — 主题相关：共享 activation-operations
- 相关主题: [Scheduling Optimization: grid stride loop](exp_gelu_fwd_001.md) — 主题相关：共享 activation-operations
- 相关主题: [Compute Optimization: two pass algorithm](exp_log_softmax_backward_001.md) — 主题相关：共享 activation-operations, aicore 单元利用/scalar 削减
- 相关主题: [Compute Optimization: vectorized memory access](exp_partial_rope_001.md) — 主题相关：共享 activation-operations, aicore 单元利用/scalar 削减, rope
- 相关主题: [Compute Optimization: fused mrope logic](exp_qwen2vl_mrope_001.md) — 主题相关：共享 activation-operations, aicore 单元利用/scalar 削减, rope
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
- 相关主题: [Compute Optimization: vectorized chunked processing](exp_triton_rope_001.md) — 主题相关：共享 activation-operations, aicore 单元利用/scalar 削减, rope
- 相关主题: [循环不变量按复用组外提（cos/sin 类小表每组一次，实测单点 +100%）](loop-invariant-group-hoisting.md) — 同属优化知识，主题相关

<!-- okf:related:end -->
