---
schema_version: okf.v1
kind: operator_optimization
type: optimization_runbook
source_family: runbooks
category: compute-pipeline
dimension: 2.1
title: 'Compute Optimization: operator fusion'
description: 'activation 场景下，Compute Optimization: operator fusion，收益 25%'
tags: [AICore 单元利用/Scalar 削减, activation-operations, activation]
confidence: verified
status: verified
created_at: 2026-08-21T00:00:00Z
updated_at: 2026-08-21T00:00:00Z
---

## Compute Optimization: operator fusion

【优化维度】：AICore 单元利用/Scalar 削减
【泛化层级】：条件性（迁移条件：硬件 910b2, 910b3, 91093, 91095；失效边界：softplus 阈值需按数据分布调整，大输入段走线性近似会改变函数行为，精度敏感场景需评估；NUM_HEADS/NUM_BATCHES 小于 BLK 配置时嵌套块循环退化；无融合需求（只要单个输出）时多读输入的假设不成立）

### 原则
- **operator fusion**：Fuses the computation of the gating parameter 'g' (involving Softplus, Exp, and multiplication) and 'beta_output' (Sigmoid) into a single kernel. This significantly reduces global memory bandwidth pressure by reading inputs 'a', 'b', 'A_log', and 'dt_bias' only once and writing outputs 'g' and 'beta_output' in a single pass.
- **numerical stability**：Implements a numerically stable Softplus approximation using `tl.where`. It switches to a linear approximation (x) when `beta * x` exceeds a defined threshold, preventing overflow in the `tl.exp` operation which is a common issue with standard Softplus implementations on lower precision hardware.
- **block tiling strategy**：Utilizes nested loops (`ROW_ITER`, `COL_ITER`) over fixed block sizes (`BLK_BATCHES`, `BLK_HEADS`). This strategy allows the kernel to handle tensor dimensions larger than the hardware's maximum block size constraints or optimal tiling limits, ensuring high occupancy and reducing kernel launch overhead for large batch/head dimensions.
- **precision upcasting**：Explicitly casts intermediate computations (addition, exponential, logarithm, and sigmoid) to `tl.float32`. This preserves numerical accuracy during the accumulation of non-linear operations which are sensitive to the reduced dynamic range and precision of fp16/bf16, before casting the final result back to the original dtype.

### 代码/骨架出处
- `TritonPath`:
```python
raw_experiences/extracted_kernels/163-fused_gdn_gating_kernel.py
```

**坏实践（反例）**：两个输出各起一个 kernel 分别计算 → 输入 `a`/`b`/`A_log`/`dt_bias` 被读两遍，launch 开销翻倍 → ✅ 正解即本条：单 kernel 单 pass 读一次输入、写两个输出，中间非线性全部 fp32 计算。

**已知实例**：activation §raw (`experiences/activation/exp_fused_gating_001.yaml`)
- 声称性能收益：25%
- 备注：This kernel is optimized for Gated Linear Unit (GLU) or GDN variants. Ensure that the threshold parameter is tuned appropriately for the specific data distribution to maximize the effectiveness of the Softplus approximation.

<!-- okf:related:start -->

# 相关

- 实践案例: [FusedBiasLeakyRelu（偏置+LeakyReLU+缩放融合）Ascend C → Triton 语义转化卡](../../ops/fused-bias-leaky-relu.md) — 主题相关：共享 activation
- 相关主题: [Memory Access Optimization: operator fusion](../memory-sram/exp_fused_add_layernorm_001.md) — 主题相关：共享 activation-operations
- 相关主题: [Memory Access Optimization: operator fusion](../memory-sram/exp_fused_norm_rope_001.md) — 主题相关：共享 activation-operations
- 相关主题: [Memory Access Optimization: grid stride loop](../memory-sram/exp_geglu_bwd_001.md) — 主题相关：共享 activation, activation-operations
- 相关主题: [Memory Access Optimization: cos sin reuse](../memory-sram/exp_rope_npu_001.md) — 主题相关：共享 activation-operations
- 相关主题: [Memory Access Optimization: two pass reduction strategy](../memory-sram/exp_softmax_backward_001.md) — 主题相关：共享 activation-operations
- 相关主题: [Memory Access Optimization: two pass reduction strategy](../memory-sram/exp_softmax_backward_non_inner_001.md) — 主题相关：共享 activation-operations
- 相关主题: [Memory Access Optimization: grid stride loop](../memory-sram/exp_swiglu_flat_001.md) — 主题相关：共享 activation, activation-operations
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
- 相关主题: [Compute Optimization: software pipelining](exp_swiglu_bwd_flat_001.md) — 主题相关：共享 activation, activation-operations, aicore 单元利用/scalar 削减
- 相关主题: [Scheduling Optimization: mixed precision computation](exp_swiglu_fwd_001.md) — 主题相关：共享 activation, activation-operations
- 相关主题: [Compute Optimization: interleaved vectorization](exp_swiglu_oai_001.md) — 主题相关：共享 activation, activation-operations, aicore 单元利用/scalar 削减
- 相关主题: [Compute Optimization: operator fusion](exp_swiglu_quant_001.md) — 主题相关：共享 activation, activation-operations, aicore 单元利用/scalar 削减
- 相关主题: [Compute Optimization: vectorized chunked processing](exp_triton_rope_001.md) — 主题相关：共享 activation-operations, aicore 单元利用/scalar 削减
- 相关主题: [算子融合模式与约束决策](operator-fusion-patterns.md) — 主题相关：共享 activation

<!-- okf:related:end -->
