---
schema_version: okf.v1
kind: operator_optimization
type: optimization_runbook
source_family: runbooks
category: compute-pipeline
dimension: 2.3
title: 'Scheduling Optimization: operator fusion'
description: 'matmul 场景下，Scheduling Optimization: operator fusion，收益 25%'
tags: [多核并行与负载均衡, matrix-operations, matmul]
confidence: verified
status: verified
created_at: 2026-08-21T00:00:00Z
updated_at: 2026-08-21T00:00:00Z
---

## Scheduling Optimization: operator fusion

【优化维度】：多核并行与负载均衡
【泛化层级】：条件性（迁移条件：硬件 910b2, 910b3；失效边界：矩阵维度需足够大使 Cube 饱和，且后续归一化不能在 Vector 单元上成为瓶颈（备注自述）；依赖 dl 模块硬件原语，跨平台不可迁移）

### 原则
- **operator fusion**：Fuses the matrix multiplication (GEMM) operation with the RMS normalization. This eliminates the need to write the intermediate GEMM results to High Bandwidth Memory (HBM) and read them back for the normalization step, significantly reducing memory bandwidth consumption and latency.
- **async pipeline scheduling**：Utilizes `dl.async_task` to dispatch the GEMM workload to the Cube unit and the RMS Norm workload to the Vector unit. This asynchronous execution model allows for efficient pipelining, potentially overlapping computation phases or reducing the scheduling overhead between the two distinct hardware stages.
- **hardware unit specialization**：Explicitly scopes the `matmul_kernel` to `dl.async_task.cube` and the `rms_norm_block_kernel` to `dl.async_task.vector`. This ensures that the heavy matrix arithmetic is handled by the dedicated matrix engine while the element-wise and reduction operations are handled by the vector engine, maximizing the utilization of the Ascend NPU's heterogeneous architecture.

### 代码/骨架出处
- `TritonPath`:
```python
raw_experiences/extracted_kernels/041-fused_matmul_ckvkr_and_rms_norm_cq_kernel.py
```

**坏实践（反例）**：GEMM 结果写回 HBM 再读回做归一化 → 完整 [M, N] 中间张量两写两读，大矩阵下带宽主导时延 → ✅ 正解即本条：融合 + `dl.async_task` cube/vector 异步流水。

**已知实例**：matmul §raw (`experiences/matmul/exp_fused_matmul_rms_001.yaml`)
- 声称性能收益：25%
- 备注：This optimization relies on the `dl` module which provides hardware-specific primitives for Ascend NPUs. It is most effective when the matrix dimensions are large enough to saturate the Cube unit, and the subsequent RMS Norm does not introduce a significant bottleneck on the Vector unit.

<!-- okf:related:start -->

# 相关

- 相关主题: [访存优化：大尺寸 matmul 的 256 分块与 L0C 复用](../memory-sram/exp_matmul_001.md) — 主题相关：共享 matmul
- 相关主题: [切分优化：A、B 均不转置 matmul 的 512B 行宽对齐分块](../memory-sram/exp_matmul_003.md) — 主题相关：共享 matmul
- 相关主题: [切分优化：A 不转置、B 转置 matmul 的 K0=256 统一分块](../memory-sram/exp_matmul_004.md) — 主题相关：共享 matmul
- 相关主题: [切分优化：双转置矩阵乘的 512B 对齐分块](../memory-sram/exp_matmul_005.md) — 主题相关：共享 matmul
- 相关主题: [Cube-Vector 场景行宽必须 512B 对齐](../memory-sram/row-width-512b-align.md) — 主题相关：共享 matmul
- 相关主题: [异构并行融合：MatMul 派 Cube、归一化+位置编码派 Vector，dl.async_task 异步掩盖](../portability/exp_fused_matmul_rms_rope_001.md) — 主题相关：共享 matmul, matrix-operations
- 相关主题: [MatMul 调度：对角遍历 grouped_launch + dot_pad_only_k hint + 寄存器原位累加](../portability/exp_matmul_ascend_001.md) — 主题相关：共享 matmul, matrix-operations
- 相关主题: [波前调度 MatMul：对角遍历 + dot_pad_only_k + autotune tile 长宽比](../portability/exp_matmul_ascend_wavefront_001.md) — 主题相关：共享 matmul, matrix-operations
- 相关主题: [dot_pad_only_k hint + 对角 block 调度 + batch/head 循环内置融合](../portability/exp_rope_matmul_001.md) — 主题相关：共享 matmul, matrix-operations
- 相关主题: [matmul core mapping 策略选择（列/行/对角）](core-mapping-strategy-selection.md) — 主题相关：共享 matmul
- 相关主题: [大矩阵 matmul 用对角 core mapping 消除 L2 bank conflict](diagonal-core-mapping-l2-bank-conflict.md) — 主题相关：共享 matmul
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
- 相关主题: [流程优化: 标准 score-归一化-加权三步流程](exp_fused_attention_001.md) — 主题相关：共享 多核并行与负载均衡
- 相关主题: [Scheduling Optimization: operator fusion](exp_fused_attn_preprocess_001.md) — 主题相关：共享 matrix-operations, 多核并行与负载均衡
- 相关主题: [Compute Optimization: operator fusion](exp_fused_gemm_scale_001.md) — 主题相关：共享 matmul, matrix-operations
- 相关主题: [Scheduling Optimization: async pipeline scheduling](exp_fused_matmul_norm_sigmoid_001.md) — 主题相关：共享 matmul, matrix-operations, 多核并行与负载均衡
- 相关主题: [Scheduling Optimization: operator fusion](exp_fused_matmul_rmsnorm_001.md) — 主题相关：共享 matmul, matrix-operations, 多核并行与负载均衡
- 相关主题: [Scheduling Optimization: operator fusion](exp_fused_matmul_rope_001.md) — 主题相关：共享 matmul, matrix-operations, 多核并行与负载均衡
- 相关主题: [Scheduling Optimization: hardware unit isolation](exp_fused_norm_rope_matmul_001.md) — 主题相关：共享 matmul, 多核并行与负载均衡
- 相关主题: [Scheduling Optimization: dynamic work stealing](exp_grouped_gemm_001.md) — 主题相关：共享 matmul, matrix-operations, 多核并行与负载均衡
- 相关主题: [Scheduling Optimization: dynamic load balancing](exp_grouped_gemm_bkmajor_001.md) — 主题相关：共享 matmul, matrix-operations, 多核并行与负载均衡
- 相关主题: [Scheduling Optimization: dynamic load balancing](exp_grouped_gemm_bnmajor_001.md) — 主题相关：共享 matmul, matrix-operations, 多核并行与负载均衡
- 相关主题: [Scheduling Optimization: dynamic load balancing](exp_grouped_matmul_001.md) — 主题相关：共享 matmul, matrix-operations, 多核并行与负载均衡
- 相关主题: [Scheduling Optimization: dynamic load balancing](exp_grouped_matmul_bkmajor_001.md) — 主题相关：共享 matmul, matrix-operations, 多核并行与负载均衡
- 相关主题: [Scheduling Optimization: activation recomputation](exp_hstu_attn_bwd_001.md) — 主题相关：共享 matrix-operations, 多核并行与负载均衡
- 相关主题: [Scheduling Optimization: static task scheduling](exp_hstu_attn_sched_001.md) — 主题相关：共享 matrix-operations, 多核并行与负载均衡
- 相关主题: [Scheduling Optimization: diagonal traversal pattern](exp_launch_diagonal_001.md) — 主题相关：共享 matrix-operations, 多核并行与负载均衡
- 相关主题: [Scheduling Optimization: heterogeneous async scheduling](exp_lightning_attn_001.md) — 主题相关：共享 多核并行与负载均衡
- 相关主题: [Scheduling Optimization: persistent kernel grid stride loop](exp_linear_persistent_001.md) — 主题相关：共享 matrix-operations, 多核并行与负载均衡
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
- 相关主题: [Scheduling Optimization: three tier hierarchical dispatch](exp_solve_tril_004.md) — 主题相关：共享 多核并行与负载均衡

<!-- okf:related:end -->
