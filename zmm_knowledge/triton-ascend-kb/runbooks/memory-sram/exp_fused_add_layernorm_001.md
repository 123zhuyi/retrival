---
schema_version: okf.v1
kind: operator_optimization
type: optimization_runbook
source_family: runbooks
category: memory-sram
dimension: 1.1
title: "Memory Access Optimization: operator fusion"
description: "normalization 场景下，Memory Access Optimization: operator fusion，收益 30%"
tags: [访存效率, activation-operations, normalization]
confidence: verified
status: verified
created_at: 2026-08-21T00:00:00Z
updated_at: 2026-08-21T00:00:00Z
---

## Memory Access Optimization: operator fusion

【优化维度】：访存效率
【泛化层级】：条件性（迁移条件：硬件 910b2, 910b3, 91093, 91095，fp16/bf16/fp32；失效边界：n_cols 较小、中间和 S 可驻留寄存器时，两遍 GM 流式方案反而更慢，应改用寄存器 tiling；fp32 统计量累加抬高 UB 占用，超大 n_cols 需重算 BLOCK_SIZE_N）

### 原则
- **operator fusion**：Fuses the element-wise addition (X + R), Layer Normalization (mean/variance calculation), and the affine transformation (scaling and shifting) into a single kernel launch. This significantly reduces the overhead of writing and reading intermediate tensors to and from global memory (HBM).
- **precision management**：Utilizes single-precision floating-point (float32) for the accumulation of mean and variance statistics to ensure numerical stability and prevent overflow/underflow, while performing the final arithmetic and storage in the input data type (e.g., fp16/bf16) to maximize throughput.
- **loop tiling strategy**：Implements a 2D tiling strategy using `BLOCK_SIZE_M` for rows and `BLOCK_SIZE_N` for columns. This allows the kernel to process data in chunks that fit efficiently into 片上 SRAM, reducing global memory access frequency and improving data reuse.
- **heuristic block sizing**：Uses `@triton.heuristics` to dynamically configure `BLOCK_SIZE_M` based on the input tensor dimensions. This ensures that the number of launched programs is optimal for the specific hardware, maximizing occupancy and resource utilization.
- **intermediate streaming**：The kernel explicitly stores the intermediate sum (S) to global memory in the first pass and reloads it in the second pass. This design choice trades off some memory bandwidth for reduced register pressure, allowing the kernel to handle larger hidden dimensions (n_cols) that exceed register capacity.

### 代码/骨架出处
- `TritonPath`:
```python
raw_experiences/extracted_kernels/250-_fused_add_layernorm_fwd_kernel.py
```

**坏实践（反例）**：加法、归一化统计、仿射变换拆成独立 kernel
```python
s = add_kernel(x, r)        # 中间和 S 写 GM
m, v = stats_kernel(s)      # S 读回算均值/方差，统计量再写读
y = affine_kernel(s, m, v)  # 又一轮 GM 读写
```
后果：中间张量反复 HBM 写读，三次 kernel 启动与带宽开销叠加。✅ 正解：单 kernel 融合加+归一化+仿射，float32 累加统计量保证数值稳定；n_cols 超寄存器容量时采用中间流式（S 落 GM、二遍读回）换寄存器压力，收益 30%。

**已知实例**：normalization §raw (`experiences/normalization/exp_fused_add_layernorm_001.yaml`)
- 声称性能收益：30%
- 备注：This kernel is optimized for scenarios where the intermediate result S is required to be materialized in global memory or when the hidden dimension is too large for register tiling. For smaller dimensions where S fits in registers, a register-tiling approach would be more performant.

<!-- okf:related:start -->

# 相关

- 相关主题: [Compute Optimization: operator fusion](../compute-pipeline/exp_fused_gating_001.md) — 主题相关：共享 activation-operations
- 相关主题: [Scheduling Optimization: grid stride loop](../compute-pipeline/exp_geglu_flat_001.md) — 主题相关：共享 activation-operations
- 相关主题: [Scheduling Optimization: autotune block m](../compute-pipeline/exp_gelu_bwd_001.md) — 主题相关：共享 activation-operations
- 相关主题: [Scheduling Optimization: grid stride loop](../compute-pipeline/exp_gelu_fwd_001.md) — 主题相关：共享 activation-operations
- 相关主题: [Compute Optimization: two pass algorithm](../compute-pipeline/exp_log_softmax_backward_001.md) — 主题相关：共享 activation-operations
- 相关主题: [Compute Optimization: vectorized memory access](../compute-pipeline/exp_partial_rope_001.md) — 主题相关：共享 activation-operations
- 相关主题: [Compute Optimization: fused mrope logic](../compute-pipeline/exp_qwen2vl_mrope_001.md) — 主题相关：共享 activation-operations
- 相关主题: [Compute Optimization: cos sin reuse](../compute-pipeline/exp_rope_001.md) — 主题相关：共享 activation-operations
- 相关主题: [Scheduling Optimization: grid stride loop](../compute-pipeline/exp_rope_backward_001.md) — 主题相关：共享 activation-operations
- 相关主题: [Compute Optimization: kernel fusion](../compute-pipeline/exp_rope_mqa_001.md) — 主题相关：共享 activation-operations
- 相关主题: [Compute Optimization: vectorized memory access](../compute-pipeline/exp_rope_triton_001.md) — 主题相关：共享 activation-operations
- 相关主题: [Scheduling Optimization: grid stride scheduling](../compute-pipeline/exp_sigmoid_grid_stride_001.md) — 主题相关：共享 activation-operations
- 相关主题: [Scheduling Optimization: autotuning strategy](../compute-pipeline/exp_silu_bwd_001.md) — 主题相关：共享 activation-operations
- 相关主题: [Scheduling Optimization: grid stride loop](../compute-pipeline/exp_silu_fwd_001.md) — 主题相关：共享 activation-operations
- 相关主题: [Compute Optimization: vectorized accumulation strategy](../compute-pipeline/exp_softmax_inner_002.md) — 主题相关：共享 activation-operations
- 相关主题: [Compute Optimization: online softmax algorithm](../compute-pipeline/exp_softmax_non_inner_001.md) — 主题相关：共享 activation-operations
- 相关主题: [Compute Optimization: activation recomputation](../compute-pipeline/exp_swiglu_bwd_001.md) — 主题相关：共享 activation-operations
- 相关主题: [Compute Optimization: software pipelining](../compute-pipeline/exp_swiglu_bwd_flat_001.md) — 主题相关：共享 activation-operations
- 相关主题: [Scheduling Optimization: mixed precision computation](../compute-pipeline/exp_swiglu_fwd_001.md) — 主题相关：共享 activation-operations
- 相关主题: [Compute Optimization: interleaved vectorization](../compute-pipeline/exp_swiglu_oai_001.md) — 主题相关：共享 activation-operations
- 相关主题: [Compute Optimization: operator fusion](../compute-pipeline/exp_swiglu_quant_001.md) — 主题相关：共享 activation-operations
- 相关主题: [Compute Optimization: vectorized chunked processing](../compute-pipeline/exp_triton_rope_001.md) — 主题相关：共享 activation-operations
- 相关主题: [大 shape 张量连续分块访存（缩小 grid、单核处理连续段）](exp_ascend_large_shape_block_001.md) — 主题相关：共享 访存效率
- 相关主题: [Memory Access Optimization: kv tile reuse](exp_attn_bwd_hstu_001.md) — 主题相关：共享 访存效率
- 相关主题: [Memory Access Optimization: operator fusion](exp_fused_norm_rope_001.md) — 主题相关：共享 activation-operations, normalization, 访存效率
- 相关主题: [Memory Access Optimization: grid stride loop](exp_geglu_bwd_001.md) — 主题相关：共享 activation-operations, 访存效率
- 相关主题: [内存访问优化：转置 load 行优先化与跨阶段复用](exp_kda_bwd_intra_memory_opt_001.md) — 主题相关：共享 访存效率
- 相关主题: [Memory Access Optimization: cos sin reuse](exp_rope_npu_001.md) — 主题相关：共享 activation-operations, 访存效率
- 相关主题: [Memory Access Optimization: two pass reduction strategy](exp_softmax_backward_001.md) — 主题相关：共享 activation-operations, 访存效率
- 相关主题: [Memory Access Optimization: two pass reduction strategy](exp_softmax_backward_non_inner_001.md) — 主题相关：共享 activation-operations, 访存效率
- 相关主题: [Memory Access Optimization: grid stride loop](exp_swiglu_flat_001.md) — 主题相关：共享 activation-operations, 访存效率

<!-- okf:related:end -->
