---
schema_version: okf.v1
kind: operator_optimization
type: optimization_runbook
source_family: runbooks
category: compute-pipeline
dimension: 2.3
title: 'Scheduling Optimization: grid stride loop'
description: 'activation 场景下，Scheduling Optimization: grid stride loop，收益 15%'
tags: [多核并行与负载均衡, activation-operations, activation]
confidence: verified
status: verified
created_at: 2026-08-21T00:00:00Z
updated_at: 2026-08-21T00:00:00Z
---

## Scheduling Optimization: grid stride loop

【优化维度】：多核并行与负载均衡
【泛化层级】：条件性（迁移条件：硬件 910b2, 910b3, 91093, 91095；失效边界：tanh 近似与精确 erf 版存在精度差，精度敏感场景需确认可接受；行数小于核数时 grid-stride 无摊薄收益；BLOCK_SIZE_M 搜索区间（1-32）针对元素级算子，不适用于计算密度高的场景）

### 原则
- **grid stride loop**：The kernel utilizes a grid-stride loop pattern (`for row_task_id in range(pid, num_row_tasks, grid_size)`) to process the row dimension. This allows a fixed number of thread blocks to process an arbitrarily large number of rows, maximizing hardware occupancy and reducing kernel launch overhead compared to a 1:1 block-to-row mapping.
- **autotuning block m**：The `@triton.autotune` decorator is configured to test `BLOCK_SIZE_M` values from 1 to 32. This dynamic tuning is critical for element-wise operations on 2D tensors, as the optimal tile size for the row dimension often varies based on the specific shape of `n_rows` and `n_cols` (e.g., handling very small batch sizes or sequence lengths efficiently).
- **precision preservation**：The computation explicitly casts the input chunk to `float32` (`x_f32 = x_chunk.to(tl.float32)`) before applying the GELU approximation. This ensures numerical stability for the `tanh` and exponential operations, preventing overflow or underflow that can occur when computing GELU directly in `fp16` or `bf16`.
- **vectorized memory access**：The inner loop iterates over columns using `BLOCK_SIZE_N` chunks with masked loads/stores. This ensures memory accesses are coalesced and vectorized, optimizing global memory bandwidth usage while correctly handling tensor boundaries where dimensions are not multiples of the block size.

### 代码/骨架出处
- `TritonPath`:
```python
raw_experiences/extracted_kernels/243-_gelu_fwd_kernel.py
```

**坏实践（反例）**：按行 1:1 发射 grid → 行数大时 launch 开销显著、行数不整除 grid 时部分核空转；且 GELU 直接在 fp16 下算 tanh 溢出 → ✅ 正解即本条：grid-stride 行循环 + autotune BLOCK_SIZE_M（1-32）+ fp32 中间精度。

**已知实例**：activation §raw (`experiences/activation/exp_gelu_fwd_001.yaml`)
- 声称性能收益：15%
- 备注：This implementation uses the tanh approximation for GELU. The grid-stride scheduling makes it particularly robust for tensors with a large number of rows (e.g., large batch sizes) where the grid size might otherwise be insufficient to cover the dimension.

- GELU fwd（activation，Ascend910_9372）案例：grid-stride 1D 循环 + 动态 grid 钳制 num_vectorcore 设计采纳，50 case 全过、geomean 0.7293（vs torch eager）；erf 路径计算 bound 限制整体水平；出处：`output/perf_result.json.speedup_vs_torch`，轨迹 file:///home/w00836812/shixi_test02/cannbot-skills-new/plugins-official/triton-op-generator/triton_ascend_output/op_1_GELU_20260917_1827_8773/report.md

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
- 相关主题: [Scheduling Optimization: grid stride loop](exp_geglu_flat_001.md) — 主题相关：共享 activation, activation-operations, 多核并行与负载均衡
- 相关主题: [Scheduling Optimization: autotune block m](exp_gelu_bwd_001.md) — 主题相关：共享 activation, activation-operations, 多核并行与负载均衡
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
