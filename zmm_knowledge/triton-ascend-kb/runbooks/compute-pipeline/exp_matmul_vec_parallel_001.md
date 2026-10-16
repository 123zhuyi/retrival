---
schema_version: okf.v1
kind: operator_optimization
type: optimization_runbook
source_family: runbooks
category: compute-pipeline
dimension: 2.1
title: 'Compute Optimization: vector parallelism strategy'
description: 'matmul 场景下，Compute Optimization: vector parallelism strategy，收益 35%'
tags: [AICore 单元利用/Scalar 削减, matrix-operations, matmul]
confidence: verified
status: verified
created_at: 2026-08-21T00:00:00Z
updated_at: 2026-08-21T00:00:00Z
---

## Compute Optimization: vector parallelism strategy

【优化维度】：AICore 单元利用/Scalar 削减
【泛化层级】：条件性（迁移条件：硬件 910b2, 910b3；失效边界：依赖 `dl.parallel` 方言特性，非 Ascend 环境不可用；VEC_PARALLEL=1 时退化为普通路径；SUB_BLK_M 划分需整除或配 mask；M/N/K < 32 时子块过小无并行收益）

### 原则
- **vector parallelism strategy**：The kernel implements a specialized vector parallel path using `dl.parallel`. When `VEC_PARALLEL` is not 1, the accumulator block is split along the M-dimension into sub-blocks (`SUB_BLK_M`). This allows the hardware to process these sub-blocks in parallel, significantly improving the utilization of vector units and instruction-level parallelism on the NPU.
- **grouped ordering for l2 reuse**：The program ID mapping uses a grouped ordering strategy (`GROUP_SIZE_M`). By calculating `pid_m` and `pid_n` based on `group_id`, the kernel ensures that blocks processed sequentially are likely to reside in the same cache lines or proximity in L2 cache. This promotes data reuse for matrix A across multiple blocks of matrix B, reducing global memory access latency.
- **activation fusion**：The `leaky_relu_custom` activation function is fused directly into the matmul computation loop. This is performed on the accumulator in FP32 precision before casting to the output type, eliminating the need for a separate kernel launch and reducing memory bandwidth usage for intermediate results.
- **autotuning integration**：The kernel is decorated with `@triton.autotune`, enabling it to dynamically select the best configuration of block sizes (`BLOCK_SIZE_M`, `BLOCK_SIZE_N`, `BLOCK_SIZE_K`) at runtime. This ensures optimal performance across a wide range of matrix dimensions (M, N, K).

### 代码/骨架出处
- `TritonPath`:
```python
raw_experiences/extracted_kernels/072-matmul_kernel.py
```

**坏实践（反例）**：激活函数另起 kernel、累加器写回 HBM 再读回激活 → 中间结果往返 + launch 开销 → ✅ 正解即本条：激活在 fp32 累加器上直接融合 + `dl.parallel` 沿 M 维子块并行 + GROUP_SIZE_M 提升 L2 复用。

**已知实例**：matmul §raw (`experiences/matmul/exp_matmul_vec_parallel_001.yaml`)
- 声称性能收益：35%
- 备注：The kernel utilizes hardware-specific dialect features (e.g., `dl.parallel`) indicative of Ascend NPU optimization. It supports generic memory layouts via stride arguments, making it versatile for different tensor formats without pre-transposition.

<!-- okf:related:start -->

# 相关

- 相关主题: [异构并行融合：MatMul 派 Cube、归一化+位置编码派 Vector，dl.async_task 异步掩盖](../portability/exp_fused_matmul_rms_rope_001.md) — 主题相关：共享 matmul, matrix-operations
- 相关主题: [MatMul 调度：对角遍历 grouped_launch + dot_pad_only_k hint + 寄存器原位累加](../portability/exp_matmul_ascend_001.md) — 主题相关：共享 matmul, matrix-operations
- 相关主题: [波前调度 MatMul：对角遍历 + dot_pad_only_k + autotune tile 长宽比](../portability/exp_matmul_ascend_wavefront_001.md) — 主题相关：共享 matmul, matrix-operations
- 相关主题: [Scheduling Optimization: dynamic autotuning](exp_addmm_ascend_001.md) — 主题相关：共享 matmul, matrix-operations
- 相关主题: [Compute Optimization: mixed precision compute](exp_attn_bwd_q_001.md) — 主题相关：共享 aicore 单元利用/scalar 削减, matrix-operations
- 相关主题: [Compute Optimization: blocking strategy](exp_attn_qk_cube_001.md) — 主题相关：共享 aicore 单元利用/scalar 削减, matrix-operations
- 相关主题: [Compute Optimization: attention sinks mechanism](exp_attn_sinks_prefill_001.md) — 主题相关：共享 aicore 单元利用/scalar 削减, matrix-operations
- 相关主题: [Compute Optimization: block ptr utilization](exp_block_matmul_inv_001.md) — 主题相关：共享 aicore 单元利用/scalar 削减, matmul, matrix-operations
- 相关主题: [Scheduling Optimization: grouped scheduling](exp_bmm_grouped_001.md) — 主题相关：共享 matmul, matrix-operations
- 相关主题: [Compute Optimization: block ptr tiling](exp_chunk_fwd_001.md) — 主题相关：共享 aicore 单元利用/scalar 削减, matrix-operations
- 相关主题: [Compute Optimization: block ptr abstraction](exp_chunk_kkt_001.md) — 主题相关：共享 aicore 单元利用/scalar 削减, matrix-operations
- 相关主题: [Compute Optimization: tiling strategy](exp_cube1_qk_gemm_001.md) — 主题相关：共享 aicore 单元利用/scalar 削减, matmul, matrix-operations
- 相关主题: [Compute Optimization: mixed precision compute](exp_flashattn_bwd_kv_001.md) — 主题相关：共享 aicore 单元利用/scalar 削减, matrix-operations
- 相关主题: [Compute Optimization: split k parallelism](exp_flashattn_splitk_gqa_001.md) — 主题相关：共享 aicore 单元利用/scalar 削减
- 相关主题: [Compute Optimization: Operator Fusion](exp_fused_delta_rule_001.md) — 主题相关：共享 aicore 单元利用/scalar 削减, matrix-operations
- 相关主题: [Compute Optimization: deep kernel fusion](exp_fused_delta_rule_npu_001.md) — 主题相关：共享 aicore 单元利用/scalar 削减, matrix-operations
- 相关主题: [Compute Optimization: operator fusion](exp_fused_entropy_001.md) — 主题相关：共享 aicore 单元利用/scalar 削减, matrix-operations
- 相关主题: [Compute Optimization: operator fusion](exp_fused_gemm_scale_001.md) — 主题相关：共享 aicore 单元利用/scalar 削减, matmul, matrix-operations
- 相关主题: [Scheduling Optimization: async pipeline scheduling](exp_fused_matmul_norm_sigmoid_001.md) — 主题相关：共享 matmul, matrix-operations
- 相关主题: [Scheduling Optimization: operator fusion](exp_fused_matmul_rms_001.md) — 主题相关：共享 matmul, matrix-operations
- 相关主题: [Scheduling Optimization: operator fusion](exp_fused_matmul_rmsnorm_001.md) — 主题相关：共享 matmul, matrix-operations
- 相关主题: [Scheduling Optimization: operator fusion](exp_fused_matmul_rope_001.md) — 主题相关：共享 matmul, matrix-operations
- 相关主题: [Scheduling Optimization: dynamic work stealing](exp_grouped_gemm_001.md) — 主题相关：共享 matmul, matrix-operations
- 相关主题: [Scheduling Optimization: dynamic load balancing](exp_grouped_gemm_bkmajor_001.md) — 主题相关：共享 matmul, matrix-operations
- 相关主题: [Scheduling Optimization: dynamic load balancing](exp_grouped_gemm_bnmajor_001.md) — 主题相关：共享 matmul, matrix-operations
- 相关主题: [Scheduling Optimization: dynamic load balancing](exp_grouped_matmul_001.md) — 主题相关：共享 matmul, matrix-operations
- 相关主题: [Scheduling Optimization: dynamic load balancing](exp_grouped_matmul_bkmajor_001.md) — 主题相关：共享 matmul, matrix-operations
- 相关主题: [计算优化: fixpipe 随路量化 + K 维循环展开 + 预取](exp_matmul_002.md) — 主题相关：共享 aicore 单元利用/scalar 削减, matmul
- 相关主题: [Scheduling Optimization: diagonal partitioning](exp_matmul_ascend_diag_001.md) — 主题相关：共享 matmul, matrix-operations
- 相关主题: [Compute Optimization: block tiling strategy](exp_matmul_bias_fused_001.md) — 主题相关：共享 aicore 单元利用/scalar 削减, matmul, matrix-operations
- 相关主题: [Scheduling Optimization: diagonal tiling strategy](exp_matmul_diag_tiling_001.md) — 主题相关：共享 matmul, matrix-operations
- 相关主题: [Scheduling Optimization: hardware specific instruction](exp_matmul_qn_001.md) — 主题相关：共享 matmul, matrix-operations
- 相关主题: [Scheduling Optimization: micro tiling strategy](exp_matmul_small_tile_001.md) — 主题相关：共享 matmul, matrix-operations
- 相关主题: [Scheduling Optimization: block swizzling](exp_matmul_swizzle_hardware_hint_001.md) — 主题相关：共享 matmul, matrix-operations
- 相关主题: [Compute Optimization: small tile strategy](exp_matmul_triton_base_001.md) — 主题相关：共享 aicore 单元利用/scalar 削减, matmul, matrix-operations
- 相关主题: [Compute Optimization: kernel fusion](exp_mhc_fused_gemm_sqrsum_001.md) — 主题相关：共享 aicore 单元利用/scalar 削减, matmul, matrix-operations
- 相关主题: [Scheduling Optimization: pid grouping strategy](exp_mm_grouped_reorder_001.md) — 主题相关：共享 matmul, matrix-operations
- 相关主题: [Compute Optimization: grid stride scheduling](exp_partial_matmul_001.md) — 主题相关：共享 aicore 单元利用/scalar 削减, matmul, matrix-operations
- 相关主题: [Scheduling Optimization: dynamic persistent scheduling](exp_partial_matmul_dynamic_001.md) — 主题相关：共享 matmul, matrix-operations
- 相关主题: [Compute Optimization: fused rope computation](exp_rope_matmul_fused_001.md) — 主题相关：共享 aicore 单元利用/scalar 削减, matmul, matrix-operations
- 相关主题: [Compute Optimization: vectorized non interleaved path](exp_rope_siso_001.md) — 主题相关：共享 aicore 单元利用/scalar 削减, matrix-operations
- 相关主题: [Compute Optimization: autotuning block size](exp_sdpa_bwd_q_001.md) — 主题相关：共享 aicore 单元利用/scalar 削减, matrix-operations
- 相关主题: [Compute Optimization: online softmax](exp_sdpa_infer_ascend_001.md) — 主题相关：共享 aicore 单元利用/scalar 削减, matrix-operations
- 相关主题: [Compute Optimization: kernel fusion](exp_ssm_gated_delta_001.md) — 主题相关：共享 aicore 单元利用/scalar 削减, matrix-operations

<!-- okf:related:end -->
