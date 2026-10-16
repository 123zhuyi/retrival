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
【泛化层级】：条件性（迁移条件：硬件 910b2, 910b3, 91093, 91095，fp16/bf16/fp32，BLOCK_SIZE ≥128 且为缓存行（如 128B）整数倍；失效边界：输入非连续/未预展平的布局不适用（需先做 contiguous）；小张量时 grid-stride 复用与软件流水收益稀释；BLOCK_SIZE 不对齐缓存行时访存对齐收益失效）

### 原则
- **grid stride loop**：Implements a grid-stride loop pattern using `tl.range` with a stride equal to `num_progs * blockSize`. This decouples the total data size from the grid size, allowing the kernel to process arbitrarily large tensors while keeping NPU core occupancy high by reusing program instances.
- **software pipelining**：Utilizes the `num_stages` parameter within the `tl.range` loop to enable software pipelining. This allows the kernel to prefetch memory loads for the next iteration while executing computations for the current iteration, effectively hiding memory latency.
- **vectorized memory access**：Loads and stores data in chunks defined by `blockSize` using `tl.arange`. This ensures GM→UB contiguous memory access and maximizes bandwidth utilization by touching contiguous memory segments.
- **precision safety**：Explicitly casts input tensors `a_val` and `b_val` to `tl.float32` before performing the sigmoid and multiplication operations. This prevents potential overflow or underflow issues associated with low-precision (fp16/bf16) exponential calculations, ensuring numerical stability.

### 代码/骨架出处
- `TritonPath`:
```python
raw_experiences/extracted_kernels/105-_swiglu_forward_kernel_flat.py
```

**坏实践（反例）**：grid 与数据量硬绑定、无 stride 复用，且低精度直接算指数
```python
grid = (triton.cdiv(total_elements, BLOCK_SIZE),)   # 数据越大 program 越多
def kernel(...):
    off = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)   # 单趟无复用、无预取
    a = tl.load(a_ptr + off)
    b = tl.load(b_ptr + off)
    r = tl.sigmoid(a) * b                           # fp16/bf16 直接做 sigmoid
```
后果：超大张量时 program 实例数随数据量线性膨胀；load 与计算串行，访存延迟无遮挡；fp16/bf16 下指数计算存在溢出/下溢风险。✅ 正解：grid-stride 循环（stride = `num_progs * BLOCK_SIZE`）解耦数据量与 grid 规模，`tl.range(..., num_stages=k)` 软件流水预取下一迭代的数据；输入先转 `tl.float32` 再做 sigmoid 与乘法。

**已知实例**：activation §raw (`experiences/activation/exp_swiglu_flat_001.yaml`)
- 声称性能收益：25%
- 备注：This kernel is optimized for flat (1D) memory layouts. Ensure input tensors are contiguous or pre-flattened for best performance. The blockSize should be tuned to be a multiple of the cache line size (e.g., 128 bytes) for optimal memory alignment.

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
- 相关主题: [Memory Access Optimization: grid stride loop](exp_geglu_bwd_001.md) — 主题相关：共享 activation, activation-operations, 访存效率
- 相关主题: [Memory Access Optimization: cos sin reuse](exp_rope_npu_001.md) — 主题相关：共享 activation-operations, 访存效率
- 相关主题: [Memory Access Optimization: two pass reduction strategy](exp_softmax_backward_001.md) — 主题相关：共享 activation-operations, 访存效率
- 相关主题: [Memory Access Optimization: two pass reduction strategy](exp_softmax_backward_non_inner_001.md) — 主题相关：共享 activation-operations, 访存效率

<!-- okf:related:end -->
