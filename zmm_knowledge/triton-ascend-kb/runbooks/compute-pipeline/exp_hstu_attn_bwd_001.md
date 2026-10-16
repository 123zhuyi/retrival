---
schema_version: okf.v1
kind: operator_optimization
type: optimization_runbook
source_family: runbooks
category: compute-pipeline
dimension: 2.3
title: 'Scheduling Optimization: activation recomputation'
description: 'attention 场景下，Scheduling Optimization: activation recomputation，收益 25%'
tags: [多核并行与负载均衡, matrix-operations, attention]
confidence: verified
status: verified
created_at: 2026-08-21T00:00:00Z
updated_at: 2026-08-21T00:00:00Z
---

## Scheduling Optimization: activation recomputation

【优化维度】：多核并行与负载均衡
【泛化层级】：条件性（迁移条件：硬件 910b2, 910b3, 91093, 91095；失效边界：重算 score 的代价随 head_dim 增大而上升，head_dim 很大或序列短时不如直接存取中间结果；依赖 seq_offsets 打包格式；dk/dv 以入参 in-place 聚合，要求外层循环正确分块）

### 原则
- **activation recomputation**：The kernel recomputes the QK dot product and SiLU activation during the backward pass instead of loading them from global memory. This trades compute for memory bandwidth, which is crucial for long sequences to avoid storing the large attention score matrix.
- **fused activation backward**：The gradient of the SiLU activation is computed in-place and fused with the matrix multiplication logic. The term `sig_trans * (1 + qk_trans * (1 - sig_trans))` is calculated directly in registers to update `dqk_trans`, minimizing memory reads/writes for intermediate states.
- **block wise accumulation**：Gradients for Keys (`dk`) and Values (`dv`) are accumulated across multiple blocks of the sequence. The kernel accepts `dk` and `dv` as input arguments, updates them via `+=`, and returns them, allowing the host to loop over sequence blocks without exceeding on-chip memory limits.
- **vectorized masking**：Causal masking is applied using `tl.where` on the full tensor block. This leverages the hardware's vectorized selection capabilities to handle the causal dependency efficiently without breaking the matrix multiplication pipeline with scalar loops.
- **packed sequence handling**：Utilizes seq_offsets to dynamically load sequence start and end indices for each batch. This enables efficient processing of variable-length sequences (packed format) without padding, significantly reducing wasted computation and memory bandwidth on padding tokens.
- **block tiling strategy**：Implements a loop over the sequence dimension (start_n) with a configurable blockN size. This tiling strategy allows the kernel to process long sequences in chunks that fit into the GPU's high-speed memory (SRAM/Cache), maximizing data reuse and minimizing global memory access latency.
- **pointer arithmetic optimization**：Pre-calculates base offsets for inputA, inputB, inputC, and Grad tensors based on batch and head indices outside the inner processing loop. This minimizes redundant address arithmetic operations inside the performance-critical loop, freeing up compute cycles for the actual attention logic.
- **conditional bias support**：Efficiently handles optional bias terms by calculating the bias offset only if HAS_BIAS is true and passing the adjusted pointer to the sub-kernel. This design avoids unnecessary memory access or branching logic when bias is not present in the model.

### 代码/骨架出处
- `TritonPath`:
```python
raw_experiences/extracted_kernels/010-_hstu_attn_bwd.py
```

**坏实践（反例）**：反向 pass 把完整 score 矩阵与激活中间值从 HBM 重新 load → 长序列下 [S, S] 级带宽消耗 → ✅ 正解即本条：反向中重算 score（`tl.dot`）与激活导数（寄存器内融合），基址偏移循环外预计算。

**已知实例**：attention §raw (`experiences/attention/exp_hstu_attn_bwd_001.yaml`)
- 声称性能收益：25%
- 备注：This kernel acts as the outer wrapper for the attention backward pass, managing batch/head dispatch and sequence tiling. It delegates the core computation to the one_col_block sub-function.

<!-- okf:related:start -->

# 相关

- 相关主题: [Scheduling Optimization: dynamic autotuning](exp_addmm_ascend_001.md) — 主题相关：共享 matrix-operations, 多核并行与负载均衡
- 相关主题: [Scheduling Optimization: dual task pipelining](exp_attn_ascend_dual_pipeline_001.md) — 主题相关：共享 attention, matrix-operations, 多核并行与负载均衡
- 相关主题: [Scheduling Optimization: grouped scheduling](exp_bmm_grouped_001.md) — 主题相关：共享 matrix-operations, 多核并行与负载均衡
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
- 相关主题: [Scheduling Optimization: block tiling strategy](exp_sdpa_bwd_kv_001.md) — 主题相关：共享 attention, matrix-operations, 多核并行与负载均衡
- 相关主题: [Compute Optimization: activation recomputation](exp_swiglu_bwd_001.md) — 同属优化知识，主题相关

<!-- okf:related:end -->
