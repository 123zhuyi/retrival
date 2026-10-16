---
schema_version: okf.v1
kind: operator_optimization
type: optimization_runbook
source_family: runbooks
category: compute-pipeline
dimension: 2.3
title: 'Scheduling Optimization: grid stride scheduling'
description: 'activation 场景下，Scheduling Optimization: grid stride scheduling，收益 15%'
tags: [多核并行与负载均衡, activation-operations, activation]
confidence: verified
status: verified
created_at: 2026-08-21T00:00:00Z
updated_at: 2026-08-21T00:00:00Z
---

## Scheduling Optimization: grid stride scheduling

【优化维度】：多核并行与负载均衡
【泛化层级】：条件性（迁移条件：硬件 910b2, 910b3, 91093, 91095；失效边界：NUM_CORES 必须等于实际发射的 program 数，否则跨步行为错误（备注自述）；SIGMOID_DIM 需对齐 cache line 才能发挥 2D 块访存；n_rows < BLOCK 时 grid-stride 退化）

### 原则
- **grid stride scheduling**：Implements a grid-stride loop using `range(pid, NUM_BLOCKS, NUM_CORES)`. This pattern allows a fixed number of thread blocks (defined by `NUM_CORES`) to process an arbitrary number of row blocks (`NUM_BLOCKS`), ensuring efficient load balancing and hardware utilization even when the total workload far exceeds the grid size.
- **precision stability**：Explicitly casts input tensors to `float32` (`q.to(tl.float32)`) before computing the sigmoid function. This prevents numerical overflow or underflow issues that frequently occur during exponential calculations in low-precision formats (fp16/bf16), while casting back to the original dtype for storage to preserve memory footprint.
- **vectorized block access**：Utilizes 2D block loading and storing (`BLOCK x SIGMOID_DIM`) via broadcasting offsets. This approach maximizes memory bandwidth throughput by coalescing memory accesses, which is significantly more efficient than scalar operations, especially when `SIGMOID_DIM` aligns with the hardware's cache line size.

### 代码/骨架出处
- `TritonPath`:
```python
sigmoid_kernel
```

**坏实践（反例）**：fp16/bf16 下直接算 sigmoid 的 exp → 大负值下溢/大正值溢出 → ✅ 正解即本条：升 fp32 算指数再降回存储 + grid-stride + 2D 块合并访存。

**已知实例**：activation §raw (`experiences/activation/exp_sigmoid_grid_stride_001.yaml`)
- 声称性能收益：15%
- 备注：The `NUM_CORES` parameter must accurately reflect the number of thread blocks launched to ensure correct stride behavior. This kernel is optimized for scenarios where the activation dimension is contiguous or accessed via a constant stride.

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
- 相关主题: [Compute Optimization: operator fusion](exp_fused_gating_001.md) — 主题相关：共享 activation, activation-operations
- 相关主题: [Scheduling Optimization: grid stride loop](exp_geglu_flat_001.md) — 主题相关：共享 activation, activation-operations, 多核并行与负载均衡
- 相关主题: [Scheduling Optimization: autotune block m](exp_gelu_bwd_001.md) — 主题相关：共享 activation, activation-operations, 多核并行与负载均衡
- 相关主题: [Scheduling Optimization: grid stride loop](exp_gelu_fwd_001.md) — 主题相关：共享 activation, activation-operations, 多核并行与负载均衡
- 相关主题: [Compute Optimization: two pass algorithm](exp_log_softmax_backward_001.md) — 主题相关：共享 activation-operations
- 相关主题: [Compute Optimization: vectorized memory access](exp_partial_rope_001.md) — 主题相关：共享 activation-operations
- 相关主题: [Compute Optimization: fused mrope logic](exp_qwen2vl_mrope_001.md) — 主题相关：共享 activation-operations
- 相关主题: [Compute Optimization: cos sin reuse](exp_rope_001.md) — 主题相关：共享 activation-operations
- 相关主题: [Scheduling Optimization: grid stride loop](exp_rope_backward_001.md) — 主题相关：共享 activation-operations, 多核并行与负载均衡
- 相关主题: [Compute Optimization: kernel fusion](exp_rope_mqa_001.md) — 主题相关：共享 activation-operations
- 相关主题: [Compute Optimization: vectorized memory access](exp_rope_triton_001.md) — 主题相关：共享 activation-operations
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
