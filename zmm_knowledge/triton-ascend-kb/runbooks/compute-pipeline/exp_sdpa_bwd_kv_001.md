---
schema_version: okf.v1
kind: operator_optimization
type: optimization_runbook
source_family: runbooks
category: compute-pipeline
dimension: 2.3
title: 'Scheduling Optimization: block tiling strategy'
description: 'attention 场景下，Scheduling Optimization: block tiling strategy，收益 25%'
tags: [多核并行与负载均衡, matrix-operations, attention]
confidence: verified
status: verified
created_at: 2026-08-21T00:00:00Z
updated_at: 2026-08-21T00:00:00Z
---

## Scheduling Optimization: block tiling strategy

【优化维度】：多核并行与负载均衡
【泛化层级】：条件性（迁移条件：硬件 910b2, 910b3, 91093, 91095；失效边界：假设输入标准 ND layout（备注自述）；S < 128 时 SxS 分块无收益；multibuffer 流水深度受 UB 容量约束；dk/dv 片上累加要求外层按 inputB/inputC 块正确组织循环）

### 原则
- **block tiling strategy**：The kernel tiles the SxS attention matrix into BLOCK_R x BLOCK_C blocks (e.g., 128x64). This allows the sub-blocks of inputB and inputC to be loaded once and reused across multiple inputA blocks (inner loop over idx_r), significantly reducing global memory access bandwidth for inputB and inputC.
- **software pipelining multibuffering**：The autotune configuration includes a 'multibuffer' parameter. When enabled, this implements software pipelining, overlapping the loading of the next tile of data (inputB, inputC, inputA) with the computation of the current tile. This hides memory latency and improves compute unit utilization.
- **sram gradient accumulation**：Gradients for inputB (dk) and inputC (dv) are accumulated in on-chip memory (SRAM/Registers) as `block_dk` and `block_dv` across the inner loop iterations. The results are only written back to global memory after processing all relevant inputA blocks for a specific inputB/inputC block. This minimizes expensive global memory write operations.
- **mixed precision computation**：The kernel utilizes mixed precision by performing accumulation in float32 (HIGH_TYPE) to ensure numerical stability during the backward pass (which is sensitive to precision loss), while storing inputs and outputs in bfloat16 (LOW_TYPE) to optimize memory bandwidth and storage.

### 代码/骨架出处
- `TritonPath`:
```python
raw_experiences/extracted_kernels/221-kernel_sdpa_bwd_kv.py
```

**坏实践（反例）**：每个 inputA 块算完就把 dk/dv 部分结果写回 HBM → 梯度中间结果反复写读 → ✅ 正解即本条：`block_dk`/`block_dv` 片上累加，处理完相关 inputA 块后一次性写回 + multibuffer 流水掩盖 load。

**已知实例**：attention §raw (`experiences/attention/exp_sdpa_bwd_kv_001.yaml`)
- 声称性能收益：25%
- 备注：This kernel is specifically optimized for the backward pass of inputB/inputC gradients in attention-like operators. It assumes inputs are in standard ND format. The 'multibuffer' optimization is particularly effective on hardware with high memory latency.

<!-- okf:related:start -->

# 相关

- 相关主题: [Scheduling Optimization: dynamic autotuning](exp_addmm_ascend_001.md) — 主题相关：共享 matrix-operations, 多核并行与负载均衡
- 相关主题: [Scheduling Optimization: dual task pipelining](exp_attn_ascend_dual_pipeline_001.md) — 主题相关：共享 attention, matrix-operations, 多核并行与负载均衡
- 相关主题: [Scheduling Optimization: grouped scheduling](exp_bmm_grouped_001.md) — 主题相关：共享 matrix-operations, 多核并行与负载均衡
- 相关主题: [Scheduling Optimization: two dimensional grid strategy](exp_causal_conv1d_prefill_003.md) — 主题相关：共享 多核并行与负载均衡
- 相关主题: [Scheduling Optimization: grid stride loop](exp_element_mul_001.md) — 主题相关：共享 matrix-operations, 多核并行与负载均衡
- 相关主题: [Scheduling Optimization: kernel fusion and recomputation](exp_entropy_backward_001.md) — 主题相关：共享 matrix-operations, 多核并行与负载均衡
- 相关主题: [Scheduling Optimization: persistent kernel grid stride](exp_entropy_backward_d_weight_001.md) — 主题相关：共享 matrix-operations, 多核并行与负载均衡
- 相关主题: [Scheduling Optimization: grid stride scheduling](exp_entropy_backward_fused_001.md) — 主题相关：共享 matrix-operations, 多核并行与负载均衡
- 相关主题: [Scheduling Optimization: grid stride loop](exp_flash_attn_gridstride_001.md) — 主题相关：共享 attention, matrix-operations, 多核并行与负载均衡
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
- 相关主题: [Scheduling Optimization: persistent kernel grid stride loop](exp_linear_persistent_001.md) — 主题相关：共享 matrix-operations, 多核并行与负载均衡
- 相关主题: [Scheduling Optimization: diagonal partitioning](exp_matmul_ascend_diag_001.md) — 主题相关：共享 matrix-operations, 多核并行与负载均衡
- 相关主题: [Scheduling Optimization: diagonal tiling strategy](exp_matmul_diag_tiling_001.md) — 主题相关：共享 matrix-operations, 多核并行与负载均衡
- 相关主题: [Scheduling Optimization: hardware specific instruction](exp_matmul_qn_001.md) — 主题相关：共享 matrix-operations, 多核并行与负载均衡
- 相关主题: [Scheduling Optimization: micro tiling strategy](exp_matmul_small_tile_001.md) — 主题相关：共享 matrix-operations, 多核并行与负载均衡
- 相关主题: [Scheduling Optimization: block swizzling](exp_matmul_swizzle_hardware_hint_001.md) — 主题相关：共享 matrix-operations, 多核并行与负载均衡
- 相关主题: [Scheduling Optimization: pid grouping strategy](exp_mm_grouped_reorder_001.md) — 主题相关：共享 matrix-operations, 多核并行与负载均衡
- 相关主题: [Scheduling Optimization: dynamic persistent scheduling](exp_partial_matmul_dynamic_001.md) — 主题相关：共享 matrix-operations, 多核并行与负载均衡

<!-- okf:related:end -->
