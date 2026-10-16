---
schema_version: okf.v1
kind: operator_optimization
type: optimization_runbook
source_family: runbooks
category: compute-pipeline
dimension: 2.3
title: 'Scheduling Optimization: grid stride loop'
description: 'attention 场景下，Scheduling Optimization: grid stride loop，收益 35%'
tags: [多核并行与负载均衡, matrix-operations, attention]
confidence: verified
status: verified
created_at: 2026-08-21T00:00:00Z
updated_at: 2026-08-21T00:00:00Z
---

## Scheduling Optimization: grid stride loop

【优化维度】：多核并行与负载均衡
【泛化层级】：条件性（迁移条件：硬件 910b2, 910b3, 91093, 91095；失效边界：grid stride 因子（20）需按具体核数与序列长调优，取值不当导致负载不均；head_dim ≥ 256 时累加器回落 HBM，流水掩盖收益部分丧失；STAGE 位掩码多阶段调度依赖编译器正确划分阶段）

### 原则
- **grid stride loop**：Implements a grid stride loop mechanism (`for block_idx in range(pid, NUM_BLOCKS, 20)`) where a single program ID processes multiple sequence blocks. This significantly increases hardware occupancy and amortizes kernel launch overhead, which is critical for batch processing or long sequences.
- **software pipelining**：Utilizes a multi-stage execution strategy (`STAGE & 1` for off-band and `STAGE & 2` for on-band). This decouples memory loading of inputB and inputC tiles from the computation of score 矩阵乘（inputA·inputB^T）, effectively hiding memory latency and keeping the compute units busy.
- **adaptive accumulator storage**：Dynamically selects the storage location for the accumulator based on the `headDim`. For dimensions < 256, it uses fast SRAM/Registers (`tl.zeros`); for larger dimensions, it falls back to global memory (`acc` pointer) to prevent expensive register spilling.
- **block ptr abstraction**：Leverages `tl.make_block_ptr` to handle complex tensor strides and offsets efficiently, allowing the compiler to generate optimized vectorized load/store instructions for the NPU.

### 代码/骨架出处
- `TritonPath`:
```python
raw_experiences/extracted_kernels/012-_attn_fwd.py
```

**坏实践（反例）**：head_dim 很大（≥256）时仍把累加器放寄存器/SRAM → 寄存器 spill 到 local memory，性能雪崩 → ✅ 正解即本条：按 headDim 自适应选择累加器存储位置（<256 用 `tl.zeros` 驻留片上，否则回落 GM 指针），配合 grid-stride 摊薄 launch 开销。

**已知实例**：attention §raw (`experiences/attention/exp_flash_attn_gridstride_001.yaml`)
- 声称性能收益：35%
- 备注：The grid stride factor (20) should be tuned based on the specific NPU's AICore count and the specific sequence length to maximize occupancy without causing resource contention.

<!-- okf:related:start -->

# 相关

- 相关主题: [Scheduling Optimization: dynamic autotuning](exp_addmm_ascend_001.md) — 主题相关：共享 matrix-operations, 多核并行与负载均衡
- 相关主题: [Scheduling Optimization: dual task pipelining](exp_attn_ascend_dual_pipeline_001.md) — 主题相关：共享 attention, matrix-operations, 多核并行与负载均衡
- 相关主题: [Scheduling Optimization: grouped scheduling](exp_bmm_grouped_001.md) — 主题相关：共享 matrix-operations, 多核并行与负载均衡
- 相关主题: [Scheduling Optimization: two dimensional grid strategy](exp_causal_conv1d_prefill_003.md) — 主题相关：共享 多核并行与负载均衡
- 相关主题: [Scheduling Optimization: pad slot early exit](exp_causal_conv1d_prefill_005.md) — 主题相关：共享 多核并行与负载均衡
- 相关主题: [Scheduling Optimization: bdt layout rationale](exp_causal_conv1d_update_007.md) — 主题相关：共享 多核并行与负载均衡
- 相关主题: [Scheduling Optimization: grid stride loop](exp_element_mul_001.md) — 主题相关：共享 matrix-operations, 多核并行与负载均衡
- 相关主题: [Scheduling Optimization: kernel fusion and recomputation](exp_entropy_backward_001.md) — 主题相关：共享 matrix-operations, 多核并行与负载均衡
- 相关主题: [Scheduling Optimization: persistent kernel grid stride](exp_entropy_backward_d_weight_001.md) — 主题相关：共享 matrix-operations, 多核并行与负载均衡
- 相关主题: [Scheduling Optimization: grid stride scheduling](exp_entropy_backward_fused_001.md) — 主题相关：共享 matrix-operations, 多核并行与负载均衡
- 相关主题: [流程优化: 标准 score-归一化-加权三步流程](exp_fused_attention_001.md) — 主题相关：共享 attention, 多核并行与负载均衡
- 相关主题: [Scheduling Optimization: operator fusion](exp_fused_attn_preprocess_001.md) — 主题相关：共享 attention, matrix-operations, 多核并行与负载均衡
- 相关主题: [Scheduling Optimization: async pipeline scheduling](exp_fused_matmul_norm_sigmoid_001.md) — 主题相关：共享 matrix-operations, 多核并行与负载均衡
- 相关主题: [Scheduling Optimization: operator fusion](exp_fused_matmul_rms_001.md) — 主题相关：共享 matrix-operations, 多核并行与负载均衡
- 相关主题: [Scheduling Optimization: operator fusion](exp_fused_matmul_rmsnorm_001.md) — 主题相关：共享 matrix-operations, 多核并行与负载均衡
- 相关主题: [Scheduling Optimization: operator fusion](exp_fused_matmul_rope_001.md) — 主题相关：共享 matrix-operations, 多核并行与负载均衡
- 相关主题: [Scheduling Optimization: dynamic work stealing](exp_grouped_gemm_001.md) — 主题相关：共享 matrix-operations, 多核并行与负载均衡
- 相关主题: [Scheduling Optimization: dynamic load balancing](exp_grouped_gemm_bkmajor_001.md) — 主题相关：共享 matrix-operations, 多核并行与负载均衡
- 相关主题: [Scheduling Optimization: dynamic load balancing](exp_grouped_gemm_bnmajor_001.md) — 主题相关：共享 matrix-operations, 多核并行与负载均衡
- 相关主题: [Scheduling Optimization: dynamic load balancing](exp_grouped_matmul_001.md) — 主题相关：共享 matrix-operations, 多核并行与负载均衡
- 相关主题: [Scheduling Optimization: dynamic load balancing](exp_grouped_matmul_bkmajor_001.md) — 主题相关：共享 matrix-operations, 多核并行与负载均衡
- 相关主题: [Scheduling Optimization: activation recomputation](exp_hstu_attn_bwd_001.md) — 主题相关：共享 attention, matrix-operations, 多核并行与负载均衡
- 相关主题: [Scheduling Optimization: static task scheduling](exp_hstu_attn_sched_001.md) — 主题相关：共享 attention, matrix-operations, 多核并行与负载均衡
- 相关主题: [Scheduling Optimization: diagonal traversal pattern](exp_launch_diagonal_001.md) — 主题相关：共享 matrix-operations, 多核并行与负载均衡
- 相关主题: [Scheduling Optimization: heterogeneous async scheduling](exp_lightning_attn_001.md) — 主题相关：共享 attention, 多核并行与负载均衡
- 相关主题: [Scheduling Optimization: persistent kernel grid stride loop](exp_linear_persistent_001.md) — 主题相关：共享 matrix-operations, 多核并行与负载均衡
- 相关主题: [Scheduling Optimization: diagonal partitioning](exp_matmul_ascend_diag_001.md) — 主题相关：共享 matrix-operations, 多核并行与负载均衡
- 相关主题: [Scheduling Optimization: diagonal tiling strategy](exp_matmul_diag_tiling_001.md) — 主题相关：共享 matrix-operations, 多核并行与负载均衡
- 相关主题: [Scheduling Optimization: hardware specific instruction](exp_matmul_qn_001.md) — 主题相关：共享 matrix-operations, 多核并行与负载均衡
- 相关主题: [Scheduling Optimization: micro tiling strategy](exp_matmul_small_tile_001.md) — 主题相关：共享 matrix-operations, 多核并行与负载均衡
- 相关主题: [Scheduling Optimization: block swizzling](exp_matmul_swizzle_hardware_hint_001.md) — 主题相关：共享 matrix-operations, 多核并行与负载均衡
- 相关主题: [Scheduling Optimization: pid grouping strategy](exp_mm_grouped_reorder_001.md) — 主题相关：共享 matrix-operations, 多核并行与负载均衡
- 相关主题: [Scheduling Optimization: dynamic persistent scheduling](exp_partial_matmul_dynamic_001.md) — 主题相关：共享 matrix-operations, 多核并行与负载均衡
- 相关主题: [算子拆分优化: 按计算阶段拆分为简单 kernel](exp_perf_010.md) — 主题相关：共享 多核并行与负载均衡
- 相关主题: [Scheduling Optimization: block tiling strategy](exp_sdpa_bwd_kv_001.md) — 主题相关：共享 attention, matrix-operations, 多核并行与负载均衡
- 相关主题: [Scheduling Optimization: three tier hierarchical dispatch](exp_solve_tril_004.md) — 主题相关：共享 多核并行与负载均衡

<!-- okf:related:end -->
