---
schema_version: okf.v1
kind: operator_optimization
type: optimization_runbook
source_family: runbooks
category: memory-sram
dimension: 1.1
title: "Memory Access Optimization: grid stride loop"
description: "activation 场景下，Memory Access Optimization: grid stride loop，收益 25%"
tags: [访存效率, activation-operations, activation]
confidence: verified
status: verified
created_at: 2026-08-21T00:00:00Z
updated_at: 2026-08-21T00:00:00Z
---

## Memory Access Optimization: grid stride loop

【优化维度】：访存效率
【泛化层级】：条件性（迁移条件：硬件 910b2, 910b3, 91093, 91095，fp16/bf16，total_elements≥4096、BLOCK_SIZE≥128，访存瓶颈的逐元素反向场景；失效边界：tensor 总量小于 grid×BLOCK_SIZE 时 grid-stride 退化为单次迭代，无收益；算力瓶颈场景下重计算策略反而增加 ALU 压力，应改回前向激活落盘）

### 原则
- **grid stride loop**：Implements a grid-stride loop pattern using `tl.range` with a stride equal to `num_progs * blockSize`. This ensures that all NPU cores are fully utilized even when the tensor size is not an exact multiple of the grid size, effectively handling arbitrary tensor dimensions without idle cores.
- **activation recomputation**：Adopts a recompute strategy for the gated-activation intermediates (`tanh_result`, `geglu_a`) during the backward pass. Instead of storing the forward activations in high-bandwidth memory (HBM), the kernel recalculates them. This trades increased ALU usage for significantly reduced memory bandwidth and storage requirements, which is critical for memory-bound operations on NPUs.
- **software pipelining**：Utilizes the `NUM_STAGES` parameter in the `tl.range` loop to enable software pipelining. This allows the kernel to overlap memory load/store operations with the arithmetic computations of the activation derivative, effectively hiding memory access latency.
- **precision preservation**：Performs intermediate accumulation in `float32` (e.g., `a.to(tl.float32)`) before casting back to the input precision. This ensures numerical stability during the derivative calculation, which involves cubic terms (`a_cubed`) and hyperbolic tangents that are sensitive to precision loss in lower precision formats.

### 代码/骨架出处
- `TritonPath`:
```python
raw_experiences/extracted_kernels/097-_geglu_backward_kernel_flat.py
```

**坏实践（反例）**：反向 kernel 把前向存下的激活中间值从 GM 读回
```python
tanh_res = tl.load(act_cache_ptr + offs)  # 前向激活缓存整体落盘再读回
a  = tl.load(a_ptr + offs)
da = dy * act_grad(tanh_res, a)
```
后果：前向需额外写出整块激活缓存、反向再读回，memory-bound 场景下 GM 流量与显存占用显著增加。✅ 正解：反向 kernel 内由 a 重计算 `tanh_result`、`geglu_a` 等中间量，以少量 ALU 换取访存与存储的大幅下降；中间累加用 fp32 保精度，并用 `NUM_STAGES` 软件流水让 load/store 与导数算术交叠。

**已知实例**：activation §raw (`experiences/activation/exp_geglu_bwd_001.yaml`)
- 声称性能收益：25%
- 备注：扁平化 1D 寻址简化了地址计算，对门控激活层非常有效。当 batch 或序列长度较大、内存带宽成为主要瓶颈时，重计算策略收益最明显。

<!-- okf:related:start -->

# 相关

- 实践案例: [FusedBiasLeakyRelu（偏置+LeakyReLU+缩放融合）Ascend C → Triton 语义转化卡](../../ops/fused-bias-leaky-relu.md) — 主题相关：共享 activation
- 相关主题: [Compute Optimization: operator fusion](../compute-pipeline/exp_fused_gating_001.md) — 主题相关：共享 activation, activation-operations
- 相关主题: [Scheduling Optimization: grid stride loop](../compute-pipeline/exp_geglu_flat_001.md) — 主题相关：共享 activation, activation-operations
- 相关主题: [Scheduling Optimization: autotune block m](../compute-pipeline/exp_gelu_bwd_001.md) — 主题相关：共享 activation, activation-operations
- 相关主题: [Scheduling Optimization: grid stride loop](../compute-pipeline/exp_gelu_fwd_001.md) — 主题相关：共享 activation, activation-operations
- 相关主题: [Compute Optimization: two pass algorithm](../compute-pipeline/exp_log_softmax_backward_001.md) — 主题相关：共享 activation-operations
- 相关主题: [Compute Optimization: vectorized memory access](../compute-pipeline/exp_partial_rope_001.md) — 主题相关：共享 activation-operations
- 相关主题: [Compute Optimization: fused mrope logic](../compute-pipeline/exp_qwen2vl_mrope_001.md) — 主题相关：共享 activation-operations
- 相关主题: [Compute Optimization: cos sin reuse](../compute-pipeline/exp_rope_001.md) — 主题相关：共享 activation-operations
- 相关主题: [Scheduling Optimization: grid stride loop](../compute-pipeline/exp_rope_backward_001.md) — 主题相关：共享 activation-operations
- 相关主题: [Compute Optimization: kernel fusion](../compute-pipeline/exp_rope_mqa_001.md) — 主题相关：共享 activation-operations
- 相关主题: [Compute Optimization: vectorized memory access](../compute-pipeline/exp_rope_triton_001.md) — 主题相关：共享 activation-operations
- 相关主题: [Scheduling Optimization: grid stride scheduling](../compute-pipeline/exp_sigmoid_grid_stride_001.md) — 主题相关：共享 activation, activation-operations
- 相关主题: [Scheduling Optimization: autotuning strategy](../compute-pipeline/exp_silu_bwd_001.md) — 主题相关：共享 activation, activation-operations
- 相关主题: [Scheduling Optimization: grid stride loop](../compute-pipeline/exp_silu_fwd_001.md) — 主题相关：共享 activation, activation-operations
- 相关主题: [Compute Optimization: vectorized accumulation strategy](../compute-pipeline/exp_softmax_inner_002.md) — 主题相关：共享 activation-operations
- 相关主题: [Compute Optimization: online softmax algorithm](../compute-pipeline/exp_softmax_non_inner_001.md) — 主题相关：共享 activation-operations
- 相关主题: [Compute Optimization: activation recomputation](../compute-pipeline/exp_swiglu_bwd_001.md) — 主题相关：共享 activation, activation-operations
- 相关主题: [Compute Optimization: software pipelining](../compute-pipeline/exp_swiglu_bwd_flat_001.md) — 主题相关：共享 activation, activation-operations
- 相关主题: [Scheduling Optimization: mixed precision computation](../compute-pipeline/exp_swiglu_fwd_001.md) — 主题相关：共享 activation, activation-operations
- 相关主题: [Compute Optimization: interleaved vectorization](../compute-pipeline/exp_swiglu_oai_001.md) — 主题相关：共享 activation, activation-operations
- 相关主题: [Compute Optimization: operator fusion](../compute-pipeline/exp_swiglu_quant_001.md) — 主题相关：共享 activation, activation-operations
- 相关主题: [Compute Optimization: vectorized chunked processing](../compute-pipeline/exp_triton_rope_001.md) — 主题相关：共享 activation-operations
- 相关主题: [算子融合模式与约束决策](../compute-pipeline/operator-fusion-patterns.md) — 主题相关：共享 activation
- 相关主题: [Memory Access Optimization: operator fusion](exp_fused_add_layernorm_001.md) — 主题相关：共享 activation-operations, 访存效率
- 相关主题: [Memory Access Optimization: operator fusion](exp_fused_norm_rope_001.md) — 主题相关：共享 activation-operations, 访存效率
- 相关主题: [Memory Access Optimization: cos sin reuse](exp_rope_npu_001.md) — 主题相关：共享 activation-operations, 访存效率
- 相关主题: [Memory Access Optimization: two pass reduction strategy](exp_softmax_backward_001.md) — 主题相关：共享 activation-operations, 访存效率
- 相关主题: [Memory Access Optimization: two pass reduction strategy](exp_softmax_backward_non_inner_001.md) — 主题相关：共享 activation-operations, 访存效率
- 相关主题: [Memory Access Optimization: grid stride loop](exp_swiglu_flat_001.md) — 主题相关：共享 activation, activation-operations, 访存效率

<!-- okf:related:end -->
