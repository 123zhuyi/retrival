---
schema_version: okf.v1
kind: operator_optimization
type: optimization_runbook
source_family: runbooks
category: compute-pipeline
dimension: 2.3
title: 'Scheduling Optimization: dynamic load balancing'
description: 'matmul 场景下，Scheduling Optimization: dynamic load balancing，收益 35%'
tags: [多核并行与负载均衡, matrix-operations, matmul]
confidence: verified
status: verified
created_at: 2026-08-21T00:00:00Z
updated_at: 2026-08-21T00:00:00Z
---

## Scheduling Optimization: dynamic load balancing

【优化维度】：多核并行与负载均衡
【泛化层级】：条件性（迁移条件：硬件 910b2, 910b3, 91093, 91095；失效边界：各组矩阵维度差异小时动态调度无优势；`dl.compile_hint` 是 Ascend 专有扩展，跨栈不可用；B 矩阵必须是 K-major 布局，否则显式转置假设不成立）

### 原则
- **dynamic load balancing**：The kernel implements a work-stealing scheduler using a `while` loop and a global block counter (`cur_block`). This allows cores to dynamically fetch tasks from the pool of matrix blocks across different groups. This is critical for Grouped GEMM where matrix dimensions vary significantly, preventing load imbalance where some cores sit idle while others process large groups.
- **hardware specific padding**：Utilizes `dl.compile_hint` with the argument "dot_pad_only_k". This provides specific instructions to the backend compiler (likely Ascend NPU) regarding how to handle padding for the dot product instruction. It optimizes the hardware instruction generation to handle boundary conditions (K-dimension padding) efficiently without excessive software overhead.
- **layout adaptation**：The kernel handles matrix B in a K-major layout (indicated by the stride logic and function name). It performs an explicit transpose (`tl.trans(b)`) on the loaded tile before the `tl.dot` operation. This allows the kernel to support non-standard data layouts for B while maintaining the high-throughput dot product computation flow required by the hardware.

### 代码/骨架出处
- `TritonPath`:
```python
raw_experiences/extracted_kernels/070-m_grouped_gemm_bKmajor_kernel.py
```

**坏实践（反例）**：分组间静态均分 block（每核固定配额）→ 组间维度差异大时尾部效应，快核空转 → ✅ 正解即本条：while + 全局 block 计数的 work-stealing 调度 + `dot_pad_only_k` 只 pad K 维。

**已知实例**：matmul §raw (`experiences/matmul/exp_grouped_gemm_bkmajor_001.yaml`)
- 声称性能收益：35%
- 备注：This kernel is specifically designed for scenarios involving batches of matrices of varying sizes (Grouped GEMM) on Ascend hardware. The `dl` hints are proprietary extensions for the Ascend software stack.

<!-- okf:related:start -->

# 相关

- 相关主题: [切分优化：A、B 均不转置 matmul 的 512B 行宽对齐分块](../memory-sram/exp_matmul_003.md) — 主题相关：共享 matmul
- 相关主题: [切分优化：A 不转置、B 转置 matmul 的 K0=256 统一分块](../memory-sram/exp_matmul_004.md) — 主题相关：共享 matmul
- 相关主题: [切分优化：双转置矩阵乘的 512B 对齐分块](../memory-sram/exp_matmul_005.md) — 主题相关：共享 matmul
- 相关主题: [异构并行融合：MatMul 派 Cube、归一化+位置编码派 Vector，dl.async_task 异步掩盖](../portability/exp_fused_matmul_rms_rope_001.md) — 主题相关：共享 matmul, matrix-operations
- 相关主题: [MatMul 调度：对角遍历 grouped_launch + dot_pad_only_k hint + 寄存器原位累加](../portability/exp_matmul_ascend_001.md) — 主题相关：共享 matmul, matrix-operations
- 相关主题: [波前调度 MatMul：对角遍历 + dot_pad_only_k + autotune tile 长宽比](../portability/exp_matmul_ascend_wavefront_001.md) — 主题相关：共享 matmul, matrix-operations
- 相关主题: [dot_pad_only_k hint + 对角 block 调度 + batch/head 循环内置融合](../portability/exp_rope_matmul_001.md) — 主题相关：共享 matmul, matrix-operations
- 相关主题: [Scheduling Optimization: dynamic autotuning](exp_addmm_ascend_001.md) — 主题相关：共享 matmul, matrix-operations, 多核并行与负载均衡
- 相关主题: [Scheduling Optimization: dual task pipelining](exp_attn_ascend_dual_pipeline_001.md) — 主题相关：共享 matrix-operations, 多核并行与负载均衡
- 相关主题: [Compute Optimization: block ptr utilization](exp_block_matmul_inv_001.md) — 主题相关：共享 matmul, matrix-operations
- 相关主题: [Scheduling Optimization: grouped scheduling](exp_bmm_grouped_001.md) — 主题相关：共享 matmul, matrix-operations, 多核并行与负载均衡
- 相关主题: [Compute Optimization: tiling strategy](exp_cube1_qk_gemm_001.md) — 主题相关：共享 matmul, matrix-operations
- 相关主题: [Scheduling Optimization: grid stride loop](exp_element_mul_001.md) — 主题相关：共享 matrix-operations, 多核并行与负载均衡
- 相关主题: [Scheduling Optimization: kernel fusion and recomputation](exp_entropy_backward_001.md) — 主题相关：共享 matrix-operations, 多核并行与负载均衡
- 相关主题: [Scheduling Optimization: persistent kernel grid stride](exp_entropy_backward_d_weight_001.md) — 主题相关：共享 matrix-operations, 多核并行与负载均衡
- 相关主题: [Scheduling Optimization: grid stride scheduling](exp_entropy_backward_fused_001.md) — 主题相关：共享 matrix-operations, 多核并行与负载均衡
- 相关主题: [Scheduling Optimization: grid stride loop](exp_flash_attn_gridstride_001.md) — 主题相关：共享 matrix-operations, 多核并行与负载均衡
- 相关主题: [Scheduling Optimization: operator fusion](exp_fused_attn_preprocess_001.md) — 主题相关：共享 matrix-operations, 多核并行与负载均衡
- 相关主题: [Compute Optimization: operator fusion](exp_fused_gemm_scale_001.md) — 主题相关：共享 matmul, matrix-operations
- 相关主题: [Scheduling Optimization: async pipeline scheduling](exp_fused_matmul_norm_sigmoid_001.md) — 主题相关：共享 matmul, matrix-operations, 多核并行与负载均衡
- 相关主题: [Scheduling Optimization: operator fusion](exp_fused_matmul_rms_001.md) — 主题相关：共享 matmul, matrix-operations, 多核并行与负载均衡
- 相关主题: [Scheduling Optimization: operator fusion](exp_fused_matmul_rmsnorm_001.md) — 主题相关：共享 matmul, matrix-operations, 多核并行与负载均衡
- 相关主题: [Scheduling Optimization: operator fusion](exp_fused_matmul_rope_001.md) — 主题相关：共享 matmul, matrix-operations, 多核并行与负载均衡
- 相关主题: [Scheduling Optimization: hardware unit isolation](exp_fused_norm_rope_matmul_001.md) — 主题相关：共享 matmul, 多核并行与负载均衡
- 相关主题: [Scheduling Optimization: dynamic work stealing](exp_grouped_gemm_001.md) — 主题相关：共享 matmul, matrix-operations, 多核并行与负载均衡
- 相关主题: [Scheduling Optimization: dynamic load balancing](exp_grouped_gemm_bnmajor_001.md) — 主题相关：共享 matmul, matrix-operations, 多核并行与负载均衡
- 相关主题: [Scheduling Optimization: dynamic load balancing](exp_grouped_matmul_001.md) — 主题相关：共享 matmul, matrix-operations, 多核并行与负载均衡
- 相关主题: [Scheduling Optimization: dynamic load balancing](exp_grouped_matmul_bkmajor_001.md) — 主题相关：共享 matmul, matrix-operations, 多核并行与负载均衡
- 相关主题: [Scheduling Optimization: activation recomputation](exp_hstu_attn_bwd_001.md) — 主题相关：共享 matrix-operations, 多核并行与负载均衡
- 相关主题: [Scheduling Optimization: static task scheduling](exp_hstu_attn_sched_001.md) — 主题相关：共享 matrix-operations, 多核并行与负载均衡
- 相关主题: [Scheduling Optimization: diagonal partitioning](exp_matmul_ascend_diag_001.md) — 主题相关：共享 matmul, matrix-operations, 多核并行与负载均衡
- 相关主题: [Compute Optimization: block tiling strategy](exp_matmul_bias_fused_001.md) — 主题相关：共享 matmul, matrix-operations
- 相关主题: [Scheduling Optimization: diagonal tiling strategy](exp_matmul_diag_tiling_001.md) — 主题相关：共享 matmul, matrix-operations, 多核并行与负载均衡
- 相关主题: [Scheduling Optimization: hardware specific instruction](exp_matmul_qn_001.md) — 主题相关：共享 matmul, matrix-operations, 多核并行与负载均衡
- 相关主题: [Scheduling Optimization: micro tiling strategy](exp_matmul_small_tile_001.md) — 主题相关：共享 matmul, matrix-operations, 多核并行与负载均衡
- 相关主题: [Scheduling Optimization: block swizzling](exp_matmul_swizzle_hardware_hint_001.md) — 主题相关：共享 matmul, matrix-operations, 多核并行与负载均衡
- 相关主题: [Compute Optimization: small tile strategy](exp_matmul_triton_base_001.md) — 主题相关：共享 matmul, matrix-operations
- 相关主题: [Compute Optimization: vector parallelism strategy](exp_matmul_vec_parallel_001.md) — 主题相关：共享 matmul, matrix-operations
- 相关主题: [Compute Optimization: kernel fusion](exp_mhc_fused_gemm_sqrsum_001.md) — 主题相关：共享 matmul, matrix-operations
- 相关主题: [Scheduling Optimization: pid grouping strategy](exp_mm_grouped_reorder_001.md) — 主题相关：共享 matmul, matrix-operations, 多核并行与负载均衡
- 相关主题: [Compute Optimization: grid stride scheduling](exp_partial_matmul_001.md) — 主题相关：共享 matmul, matrix-operations
- 相关主题: [Scheduling Optimization: dynamic persistent scheduling](exp_partial_matmul_dynamic_001.md) — 主题相关：共享 matmul, matrix-operations, 多核并行与负载均衡
- 相关主题: [Compute Optimization: fused rope computation](exp_rope_matmul_fused_001.md) — 主题相关：共享 matmul, matrix-operations
- 相关主题: [Scheduling Optimization: block tiling strategy](exp_sdpa_bwd_kv_001.md) — 主题相关：共享 matrix-operations, 多核并行与负载均衡

<!-- okf:related:end -->
