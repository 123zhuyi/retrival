---
schema_version: okf.v1
kind: operator_optimization
type: optimization_runbook
source_family: runbooks
category: compute-pipeline
dimension: 2.3
title: 'Scheduling Optimization: grid stride loop'
description: 'activation 场景下，Scheduling Optimization: grid stride loop，收益 25%'
tags: [多核并行与负载均衡, activation-operations, activation]
confidence: verified
status: verified
created_at: 2026-08-21T00:00:00Z
updated_at: 2026-08-21T00:00:00Z
---

## Scheduling Optimization: grid stride loop

【优化维度】：多核并行与负载均衡
【泛化层级】：条件性（迁移条件：硬件 910b2, 910b3, 91093, 91095；失效边界：flat 1D 化以输入连续为前提，非连续 layout 不适用；num_stages 过深超出 UB 预算反而降速；total_elements < 1024 时 grid-stride 无摊薄收益）

### 原则
- **grid stride loop**：The kernel utilizes a grid-stride loop pattern via `tl.range(start_idx, total_elements, stride, num_stages=NUM_STAGES)`. This allows a single program to process multiple non-contiguous blocks of data, ensuring high utilization of all NPU cores even when the total workload is not perfectly aligned with the grid size or significantly larger than the grid capacity.
- **memory flattening**：By treating the input tensors as a flat 1D array (`total_elements`), the kernel simplifies memory addressing logic. This eliminates the overhead of calculating multi-dimensional indices (e.g., row/col strides) within the hot loop, reducing instruction count and improving memory throughput.
- **precision stabilization**：The input `a_val` is explicitly cast to `float32` before the GELU calculation. This ensures numerical stability during the cubic power operation (`a_val * a_val * a_val`) and the `tanh` approximation, preventing potential overflow or underflow issues common in lower precision floating-point types (fp16/bf16) before casting back to the original dtype.
- **software pipelining**：The `num_stages` parameter is passed to `tl.range`, enabling software pipelining. This overlaps memory load operations with the arithmetic computations of the GELU activation and multiplication, effectively hiding memory latency and keeping the compute units busy.

### 代码/骨架出处
- `TritonPath`:
```python
_geglu_forward_kernel_flat
```

**坏实践（反例）**：热循环内逐元素计算多维索引（row/col stride），且激活的三次方与 tanh 直接在 fp16/bf16 下算 → 标量指令膨胀 + 低精度溢出风险 → ✅ 正解即本条：flat 1D 化消除多维寻址 + grid-stride + 中间升 fp32 再降回。

**已知实例**：activation §raw (`experiences/activation/exp_geglu_flat_001.yaml`)
- 声称性能收益：25%
- 备注：This implementation is particularly effective for GEGLU layers in Transformer models where the activation and gate tensors are contiguous in memory. The tanh approximation of GELU is used for hardware compatibility.

- GELU fwd（activation，Ascend910_9372）案例：flat 1D 展平 + fp32 中间精度采纳并 50/50 通过精度验证（fp32 rtol 9.77e-4 生态标准）；出处：`output/iter_0/verify/verify_result.json`，轨迹 file:///home/w00836812/shixi_test02/cannbot-skills-new/plugins-official/triton-op-generator/triton_ascend_output/op_1_GELU_20260917_1827_8773/report.md

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
- 相关主题: [流程优化: 标准 score-归一化-加权三步流程](exp_fused_attention_001.md) — 主题相关：共享 多核并行与负载均衡
- 相关主题: [Compute Optimization: operator fusion](exp_fused_gating_001.md) — 主题相关：共享 activation, activation-operations
- 相关主题: [Scheduling Optimization: autotune block m](exp_gelu_bwd_001.md) — 主题相关：共享 activation, activation-operations, 多核并行与负载均衡
- 相关主题: [Scheduling Optimization: grid stride loop](exp_gelu_fwd_001.md) — 主题相关：共享 activation, activation-operations, 多核并行与负载均衡
- 相关主题: [Compute Optimization: two pass algorithm](exp_log_softmax_backward_001.md) — 主题相关：共享 activation-operations
- 相关主题: [Compute Optimization: vectorized memory access](exp_partial_rope_001.md) — 主题相关：共享 activation-operations
- 相关主题: [Compute Optimization: fused mrope logic](exp_qwen2vl_mrope_001.md) — 主题相关：共享 activation-operations
- 相关主题: [Compute Optimization: cos sin reuse](exp_rope_001.md) — 主题相关：共享 activation-operations
- 相关主题: [Scheduling Optimization: grid stride loop](exp_rope_backward_001.md) — 主题相关：共享 activation-operations, 多核并行与负载均衡
- 相关主题: [Compute Optimization: kernel fusion](exp_rope_mqa_001.md) — 主题相关：共享 activation-operations
- 相关主题: [Compute Optimization: vectorized memory access](exp_rope_triton_001.md) — 主题相关：共享 activation-operations
- 相关主题: [Scheduling Optimization: grid stride scheduling](exp_sigmoid_grid_stride_001.md) — 主题相关：共享 activation, activation-operations, 多核并行与负载均衡
- 相关主题: [Scheduling Optimization: autotuning strategy](exp_silu_bwd_001.md) — 主题相关：共享 activation, activation-operations, 多核并行与负载均衡
- 相关主题: [Scheduling Optimization: grid stride loop](exp_silu_fwd_001.md) — 主题相关：共享 activation, activation-operations, 多核并行与负载均衡
- 相关主题: [Compute Optimization: vectorized accumulation strategy](exp_softmax_inner_002.md) — 主题相关：共享 activation-operations
- 相关主题: [Compute Optimization: online softmax algorithm](exp_softmax_non_inner_001.md) — 主题相关：共享 activation-operations
- 相关主题: [Compute Optimization: activation recomputation](exp_swiglu_bwd_001.md) — 主题相关：共享 activation, activation-operations
- 相关主题: [Compute Optimization: software pipelining](exp_swiglu_bwd_flat_001.md) — 主题相关：共享 activation, activation-operations
- 相关主题: [Scheduling Optimization: mixed precision computation](exp_swiglu_fwd_001.md) — 主题相关：共享 activation, activation-operations, 多核并行与负载均衡
- 相关主题: [Compute Optimization: interleaved vectorization](exp_swiglu_oai_001.md) — 主题相关：共享 activation, activation-operations
- 相关主题: [Compute Optimization: operator fusion](exp_swiglu_quant_001.md) — 主题相关：共享 activation, activation-operations
- 相关主题: [Compute Optimization: vectorized chunked processing](exp_triton_rope_001.md) — 主题相关：共享 activation-operations
- 相关主题: [算子融合模式与约束决策](operator-fusion-patterns.md) — 主题相关：共享 activation

<!-- okf:related:end -->
