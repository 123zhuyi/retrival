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
【泛化层级】：条件性（迁移条件：硬件 910b2, 910b3；失效边界：依赖 Da Vinci dialect 与符合 `dl.async_task` 接口的子 kernel（备注自述）；blockM/blockN/blockK/BLOCK_NORM 需匹配 910b 的 L0/L1 容量与目标 shape，错配即失去片上驻留收益）

### 原则
- **operator fusion**：Fuses the Matrix Multiplication and RMS Normalization into a single kernel. This eliminates the need to write the intermediate MatMul results back to global memory (HBM) and read them back for the normalization step, significantly reducing memory bandwidth consumption and latency.
- **hardware unit dispatch**：Explicitly maps the MatMul computation to the 'cube' scope (Matrix Unit) and the RMS Norm computation to the 'vector' scope (Vector Unit). This leverages the specialized hardware pipelines of the Ascend NPU, ensuring that the heavy matrix math and the element-wise/reduction math are handled by their most efficient processing units.
- **async task scheduling**：Utilizes the 'dl.async_task' construct to decouple the execution of the Cube and Vector tasks. This allows the underlying runtime or hardware to pipeline these operations, potentially overlapping data transfers or computation phases to hide latency and improve overall throughput.
- **hardware scope dispatching**：The code explicitly utilizes the Da Vinci (dl) dialect to dispatch the `matmul_kernel` to the `cube` scope and the `rms_norm_ckv_kernel` to the `vector` scope. This ensures that the computationally intensive matrix multiplication is offloaded to the dedicated Cube unit, while the memory-bandwidth intensive normalization is handled by the Vector unit, maximizing hardware resource utilization on Ascend NPUs.
- **asynchronous task scheduling**：By wrapping the kernel calls in `dl.async_task`, the runtime is permitted to execute these operations asynchronously. This facilitates pipeline parallelism where the Vector unit can process normalization on a block of data while the Cube unit is processing the matrix multiplication for the next or current block, effectively hiding latency and improving throughput.
- **configurable tiling strategy**：The kernel exposes `blockM`, `blockN`, `blockK`, and `BLOCK_NORM` as `tl.constexpr` arguments. This allows for precise tuning of the tiling sizes to match the L0/L1 cache sizes of the specific hardware (910b) and the dynamic shape of the KV cache (defined by `total_seq_len` and `NOPE_DIM`), minimizing global memory access overhead.

### 代码/骨架出处
- `TritonPath`:
```python
raw_experiences/extracted_kernels/052-fused_matmul_qb_and_rmsnorm_kv_kernel.py
```

**坏实践（反例）**：matmul 与归一化拆成独立 kernel、且全程串行调度 → 中间结果 HBM 往返 + Cube/Vector 无法重叠 → ✅ 正解即本条：`dl.async_task` 双 scope 解耦执行，vector 处理当前块归一化时 cube 可算下一块矩阵乘。

**已知实例**：matmul §raw (`experiences/matmul/exp_fused_matmul_rmsnorm_001.yaml`)
- 声称性能收益：25%
- 备注：This optimization is specifically designed for the Ascend NPU architecture using the Da Vinci dialect extensions. It assumes that the `matmul_kernel` and `rms_norm_ckv_kernel` are defined elsewhere and compatible with the `dl.async_task` interface.

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
- 相关主题: [Scheduling Optimization: operator fusion](exp_fused_matmul_rms_001.md) — 主题相关：共享 matmul, matrix-operations, 多核并行与负载均衡
- 相关主题: [Scheduling Optimization: operator fusion](exp_fused_matmul_rope_001.md) — 主题相关：共享 matmul, matrix-operations, 多核并行与负载均衡
- 相关主题: [Scheduling Optimization: hardware unit isolation](exp_fused_norm_rope_matmul_001.md) — 主题相关：共享 matmul, 多核并行与负载均衡
- 相关主题: [Scheduling Optimization: dynamic work stealing](exp_grouped_gemm_001.md) — 主题相关：共享 matmul, matrix-operations, 多核并行与负载均衡
- 相关主题: [Scheduling Optimization: dynamic load balancing](exp_grouped_gemm_bkmajor_001.md) — 主题相关：共享 matmul, matrix-operations, 多核并行与负载均衡
- 相关主题: [Scheduling Optimization: dynamic load balancing](exp_grouped_gemm_bnmajor_001.md) — 主题相关：共享 matmul, matrix-operations, 多核并行与负载均衡
- 相关主题: [Scheduling Optimization: dynamic load balancing](exp_grouped_matmul_001.md) — 主题相关：共享 matmul, matrix-operations, 多核并行与负载均衡
- 相关主题: [Scheduling Optimization: dynamic load balancing](exp_grouped_matmul_bkmajor_001.md) — 主题相关：共享 matmul, matrix-operations, 多核并行与负载均衡
- 相关主题: [Scheduling Optimization: activation recomputation](exp_hstu_attn_bwd_001.md) — 主题相关：共享 matrix-operations, 多核并行与负载均衡
- 相关主题: [Scheduling Optimization: static task scheduling](exp_hstu_attn_sched_001.md) — 主题相关：共享 matrix-operations, 多核并行与负载均衡
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
