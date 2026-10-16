---
schema_version: okf.v1
kind: operator_optimization
type: optimization_runbook
source_family: runbooks
category: compute-pipeline
dimension: 2.1
title: 'Compute Optimization: activation recomputation'
description: 'activation 场景下，Compute Optimization: activation recomputation，收益 25%'
tags: [AICore 单元利用/Scalar 削减, activation-operations, activation]
confidence: verified
status: verified
created_at: 2026-08-21T00:00:00Z
updated_at: 2026-08-21T00:00:00Z
---

## Compute Optimization: activation recomputation

【优化维度】：AICore 单元利用/Scalar 削减
【泛化层级】：条件性（迁移条件：硬件 910b2, 910b3；失效边界：重算以算换存，前向已存激活时重算无收益；BLOCK_SIZE_M 1-32 区间适配元素级反向；a/b 梯度在同一块迭代内分别计算以复用已载入块（备注自述），拆分会失去该复用）

### 原则
- **activation recomputation**：The kernel employs a recomputation strategy for the sigmoid and SiLU functions during the backward pass. By recalculating these values locally instead of storing them in global memory during the forward pass, it significantly reduces memory bandwidth consumption and memory footprint, which is critical for large batch sizes.
- **mixed precision computation**：To ensure numerical stability, the input tensor 'a' is explicitly cast to float32 before the sigmoid operation. This prevents potential underflow or overflow issues inherent to half-precision formats (fp16/bf16) when calculating exponential functions, while the rest of the arithmetic can leverage hardware acceleration for lower precision.
- **vectorized block processing**：The kernel loads and stores data in chunks defined by blockSize. This maximizes global memory throughput by ensuring memory accesses are coalesced and aligned with the hardware's cache line size, reducing the number of memory transactions.
- **Optimization Point 1**：Dynamic Block Size Tuning. The kernel utilizes `@triton.autotune` to search for the optimal `BLOCK_SIZE_M` (ranging from 1 to 32) at runtime. This is critical for element-wise operations on matrices with varying row counts, ensuring that the thread block size aligns well with the hardware's memory access granularity and minimizing tail effects.
- **Optimization Point 2**：FP32 Intermediate Precision. To guarantee numerical stability during the backward pass, the kernel casts input tensors to `float32` (`a_f32`) before computing the sigmoid and SiLU derivatives. This prevents precision loss and potential underflow/overflow that can occur when computing non-linear activation gradients directly in `fp16` or `bf16`.
- **Optimization Point 3**：Efficient Loop Tiling. The implementation uses a nested loop structure where the outer loop handles rows (tuned by `BLOCK_SIZE_M`) and the inner loop handles columns (`BLOCK_SIZE_N`). This approach maximizes data locality and allows the kernel to process matrices of arbitrary dimensions that exceed the register file capacity by iterating over tiles.

### 代码/骨架出处
- `TritonPath`:
```python
raw_experiences/extracted_kernels/212-_swiglu_bwd_kernel.py
```

**坏实践（反例）**：前向把 sigmoid/silu 中间激活存到 HBM 供反向读取 → 大 batch 下带宽与显存双高 → ✅ 正解即本条：反向就地重算 sigmoid/silu（fp32 下）+ 块式合并访存。

**已知实例**：activation §raw (`experiences/activation/exp_swiglu_bwd_001.yaml`)
- 声称性能收益：25%
- 备注：This kernel is specifically optimized for the SwiGLU backward pass. The separation of gradient calculations for 'a' and 'b' allows for efficient memory reuse of the loaded input chunks within the same block iteration.

<!-- okf:related:start -->

# 相关

- 实践案例: [FusedBiasLeakyRelu（偏置+LeakyReLU+缩放融合）Ascend C → Triton 语义转化卡](../../ops/fused-bias-leaky-relu.md) — 主题相关：共享 activation
- 相关主题: [Memory Access Optimization: operator fusion](../memory-sram/exp_fused_add_layernorm_001.md) — 主题相关：共享 activation-operations
- 相关主题: [Memory Access Optimization: operator fusion](../memory-sram/exp_fused_norm_rope_001.md) — 主题相关：共享 activation-operations
- 相关主题: [Memory Access Optimization: grid stride loop](../memory-sram/exp_geglu_bwd_001.md) — 主题相关：共享 activation, activation-operations
- 相关主题: [Memory Access Optimization: cos sin reuse](../memory-sram/exp_rope_npu_001.md) — 主题相关：共享 activation-operations
- 相关主题: [Memory Access Optimization: grid stride loop](../memory-sram/exp_swiglu_flat_001.md) — 主题相关：共享 activation, activation-operations
- 相关主题: [Scheduling Optimization: kernel fusion and recomputation](exp_entropy_backward_001.md) — 同属优化知识，主题相关
- 相关主题: [Compute Optimization: operator fusion](exp_fused_gating_001.md) — 主题相关：共享 activation, activation-operations, aicore 单元利用/scalar 削减
- 相关主题: [Scheduling Optimization: grid stride loop](exp_geglu_flat_001.md) — 主题相关：共享 activation, activation-operations
- 相关主题: [Scheduling Optimization: autotune block m](exp_gelu_bwd_001.md) — 主题相关：共享 activation, activation-operations
- 相关主题: [Scheduling Optimization: grid stride loop](exp_gelu_fwd_001.md) — 主题相关：共享 activation, activation-operations
- 相关主题: [Scheduling Optimization: activation recomputation](exp_hstu_attn_bwd_001.md) — 同属优化知识，主题相关
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
- 相关主题: [Compute Optimization: software pipelining](exp_swiglu_bwd_flat_001.md) — 主题相关：共享 activation, activation-operations, aicore 单元利用/scalar 削减
- 相关主题: [Scheduling Optimization: mixed precision computation](exp_swiglu_fwd_001.md) — 主题相关：共享 activation, activation-operations
- 相关主题: [Compute Optimization: interleaved vectorization](exp_swiglu_oai_001.md) — 主题相关：共享 activation, activation-operations, aicore 单元利用/scalar 削减
- 相关主题: [Compute Optimization: operator fusion](exp_swiglu_quant_001.md) — 主题相关：共享 activation, activation-operations, aicore 单元利用/scalar 削减
- 相关主题: [Compute Optimization: vectorized chunked processing](exp_triton_rope_001.md) — 主题相关：共享 activation-operations, aicore 单元利用/scalar 削减
- 相关主题: [算子融合模式与约束决策](operator-fusion-patterns.md) — 主题相关：共享 activation

<!-- okf:related:end -->
