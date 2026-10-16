---
schema_version: okf.v1
kind: operator_optimization
type: optimization_runbook
source_family: runbooks
category: compute-pipeline
dimension: 2.3
title: 'Scheduling Optimization: autotune block m'
description: 'activation 场景下，Scheduling Optimization: autotune block m，收益 20%'
tags: [多核并行与负载均衡, activation-operations, activation]
confidence: verified
status: verified
created_at: 2026-08-21T00:00:00Z
updated_at: 2026-08-21T00:00:00Z
---

## Scheduling Optimization: autotune block m

【优化维度】：多核并行与负载均衡
【泛化层级】：条件性（迁移条件：硬件 910b2, 910b3；失效边界：假设行主序 ND layout，其他 layout 需重设计访存；BLOCK_SIZE_M 的 1-32 搜索区间是元素级算子的保守配置，计算密度更高的算子需另调；寄存器压力极低时 autotune 上限可放宽）

### 原则
- **autotune block m**：The kernel utilizes triton.autotune to dynamically select the optimal BLOCK_SIZE_M (ranging from 1 to 32). This balances register pressure and instruction-level parallelism (ILP) specifically for the row dimension, ensuring optimal performance across varying tensor shapes.
- **grid stride loop**：Implements a grid-stride loop over the row dimension (`for row_task_id in range(pid, num_row_tasks, grid_size)`). This ensures that all available compute cores are utilized even if the number of rows is not perfectly divisible by the grid size, significantly improving load balancing and hardware occupancy.
- **vectorized memory access**：The inner loop iterates over columns using BLOCK_SIZE_N, allowing for contiguous memory loads and stores. This maximizes memory bandwidth utilization and coalescing efficiency.
- **precision upcast**：Intermediate calculations (tanh, cubic terms) are explicitly upcasted to float32 (`x_f32 = x_chunk.to(tl.float32)`). This preserves numerical stability during the derivative computation before casting back to the input data type for the final store.

### 代码/骨架出处
- `TritonPath`:
```python
raw_experiences/extracted_kernels/244-_gelu_bwd_kernel.py
```

**坏实践（反例）**：反向导数的 tanh 与三次项直接在 fp16/bf16 下计算 → 低精度下上溢/下溢，梯度不准 → ✅ 正解即本条：`x_f32 = x_chunk.to(tl.float32)` 中间升 fp32 算完再降回，配合 grid-stride 行循环。

**已知实例**：activation §raw (`experiences/activation/exp_gelu_bwd_001.yaml`)
- 声称性能收益：20%
- 备注：The kernel is optimized for row-major (ND) layouts. The autotuning range for BLOCK_SIZE_M is conservative (up to 32), which is suitable for element-wise operations where compute intensity is lower than matrix multiplications, preventing register spilling.

- GELU fwd（activation，Ascend910_9372）案例：autotune BLOCK ∈ {512..32768} key=[numel, APPROX_TANH] 采纳；向上扩 65536 档实测无效（见 autotuning/exp_perf_001 证伪实例）；轨迹 file:///home/w00836812/shixi_test02/cannbot-skills-new/plugins-official/triton-op-generator/triton_ascend_output/op_1_GELU_20260917_1827_8773/report.md

<!-- okf:related:start -->

# 相关

- 实践案例: [FusedBiasLeakyRelu（偏置+LeakyReLU+缩放融合）Ascend C → Triton 语义转化卡](../../ops/fused-bias-leaky-relu.md) — 主题相关：共享 activation
- 相关主题: [Triton API Quick Reference for Ascend Development](../../reference/triton-api-cheatsheet.md) — 同属优化知识，主题相关
- 相关主题: [累加/原地修改算子 autotune 必须每次重置输出 buffer](../autotuning/autotune-debug.md) — 同属优化知识，主题相关
- 相关主题: [autotune 必须探索 num_warps 与 num_stages](../autotuning/autotune-num-warps-stages.md) — 同属优化知识，主题相关
- 相关主题: [autotune 配置空间须覆盖 num_warps(2/4/8)×num_stages(2/3/4) 与 tile 组合](../autotuning/exp_perf_017.md) — 同属优化知识，主题相关
- 相关主题: [µs 级 kernel 禁用 autotune；num_stages/num_warps 在 Triton-Ascend 不生效](../autotuning/triton-ascend-microkernel-autotune-noise.md) — 同属优化知识，主题相关
- 相关主题: [Memory Access Optimization: operator fusion](../memory-sram/exp_fused_add_layernorm_001.md) — 主题相关：共享 activation-operations
- 相关主题: [Memory Access Optimization: operator fusion](../memory-sram/exp_fused_norm_rope_001.md) — 主题相关：共享 activation-operations
- 相关主题: [Memory Access Optimization: grid stride loop](../memory-sram/exp_geglu_bwd_001.md) — 主题相关：共享 activation, activation-operations
- 相关主题: [Memory Access Optimization: cos sin reuse](../memory-sram/exp_rope_npu_001.md) — 主题相关：共享 activation-operations
- 相关主题: [Memory Access Optimization: grid stride loop](../memory-sram/exp_swiglu_flat_001.md) — 主题相关：共享 activation, activation-operations
- 相关主题: [Tiling 预算与布局驱动的块大小选择](../memory-sram/tiling-budget-and-layout-driven-selection.md) — 同属优化知识，主题相关
- 相关主题: [量化 matmul autotune 配置策略诊断--按 shape 范围特化](../meta-skills/quant-matmul-autotune-strategy.md) — 同属优化知识，主题相关
- 相关主题: [波前调度 MatMul：对角遍历 + dot_pad_only_k + autotune tile 长宽比](../portability/exp_matmul_ascend_wavefront_001.md) — 同属优化知识，主题相关
- 相关主题: [流程优化: 标准 score-归一化-加权三步流程](exp_fused_attention_001.md) — 主题相关：共享 多核并行与负载均衡
- 相关主题: [Compute Optimization: operator fusion](exp_fused_gating_001.md) — 主题相关：共享 activation, activation-operations
- 相关主题: [Scheduling Optimization: grid stride loop](exp_geglu_flat_001.md) — 主题相关：共享 activation, activation-operations, 多核并行与负载均衡
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
- 相关主题: [Compute Optimization: activation recomputation](exp_swiglu_bwd_001.md) — 主题相关：共享 activation, activation-operations
- 相关主题: [Compute Optimization: software pipelining](exp_swiglu_bwd_flat_001.md) — 主题相关：共享 activation, activation-operations
- 相关主题: [Scheduling Optimization: mixed precision computation](exp_swiglu_fwd_001.md) — 主题相关：共享 activation, activation-operations, 多核并行与负载均衡
- 相关主题: [Compute Optimization: interleaved vectorization](exp_swiglu_oai_001.md) — 主题相关：共享 activation, activation-operations
- 相关主题: [Compute Optimization: operator fusion](exp_swiglu_quant_001.md) — 主题相关：共享 activation, activation-operations
- 相关主题: [Compute Optimization: vectorized chunked processing](exp_triton_rope_001.md) — 主题相关：共享 activation-operations
- 相关主题: [算子融合模式与约束决策](operator-fusion-patterns.md) — 主题相关：共享 activation

<!-- okf:related:end -->
