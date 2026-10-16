---
schema_version: okf.v1
kind: operator_optimization
type: optimization_runbook
source_family: runbooks
category: compute-pipeline
dimension: 2.1
title: 'Compute Optimization: vectorized non interleaved path'
description: 'rope 场景下，Compute Optimization: vectorized non interleaved path，收益 25%'
tags: [AICore 单元利用/Scalar 削减, matrix-operations, rope]
confidence: verified
status: verified
created_at: 2026-08-21T00:00:00Z
updated_at: 2026-08-21T00:00:00Z
---

## Compute Optimization: vectorized non interleaved path

【优化维度】：AICore 单元利用/Scalar 削减
【泛化层级】：条件性（迁移条件：硬件 910b2, 910b3；失效边界：BLOCK_D 必须匹配硬件向量宽度（备注自述为性能关键）；rotary_interleaved 为编译期常量，运行时切换布局需重编译；helper 子 kernel 承担实际 load/store，主 kernel 的优化依赖其配合）

### 原则
- **vectorized non interleaved path**：For non-interleaved data layouts, the kernel constructs dimension ranges using tl.arange to process the entire head dimension block in a single pass. This eliminates loop overhead and maximizes vector unit utilization compared to the scalar iteration used in the interleaved path.
- **3d tiling strategy**：The kernel employs a 3D tiling strategy (blockN for tokens, BLOCK_H for heads, BLOCK_D for dimensions). This maps the computation efficiently to the hardware grid, ensuring data loaded into registers/SRAM is reused before eviction, optimizing memory bandwidth usage.
- **layout specific optimization**：By using a compile-time constant (rotary_interleaved), the kernel generates specialized code paths. The interleaved path handles element-wise pairs via a loop, while the non-interleaved path leverages bulk vector operations, ensuring optimal performance for both common tensor formats without runtime divergence.

### 代码/骨架出处
- `TritonPath`:
```python
raw_experiences/extracted_kernels/117-rotary_embedding_siso_kernel.py
```

**坏实践（反例）**：non-interleaved 布局也走逐元素对的标量循环（interleaved 路径的写法）→ 循环开销 + 向量单元闲置 → ✅ 正解即本条：`rotary_interleaved` 编译期分叉，non-interleaved 用 `tl.arange` 整块单 pass 向量处理。

**已知实例**：rope §raw (`experiences/rope/exp_rope_siso_001.yaml`)
- 声称性能收益：25%
- 备注：The performance relies heavily on the BLOCK_D size matching the hardware's vector width. The helper function rotary_embedding_rw_kernel is assumed to handle the actual load/store and arithmetic operations.

<!-- okf:related:start -->

# 相关

- 相关主题: [Scheduling Optimization: dynamic autotuning](exp_addmm_ascend_001.md) — 主题相关：共享 matrix-operations
- 相关主题: [Scheduling Optimization: dual task pipelining](exp_attn_ascend_dual_pipeline_001.md) — 主题相关：共享 matrix-operations
- 相关主题: [Compute Optimization: mixed precision compute](exp_attn_bwd_q_001.md) — 主题相关：共享 aicore 单元利用/scalar 削减, matrix-operations
- 相关主题: [Compute Optimization: blocking strategy](exp_attn_qk_cube_001.md) — 主题相关：共享 aicore 单元利用/scalar 削减, matrix-operations
- 相关主题: [Compute Optimization: attention sinks mechanism](exp_attn_sinks_prefill_001.md) — 主题相关：共享 aicore 单元利用/scalar 削减, matrix-operations
- 相关主题: [Compute Optimization: block ptr tiling](exp_block_inverse_001.md) — 主题相关：共享 aicore 单元利用/scalar 削减, matrix-operations
- 相关主题: [Compute Optimization: block ptr utilization](exp_block_matmul_inv_001.md) — 主题相关：共享 aicore 单元利用/scalar 削减, matrix-operations
- 相关主题: [Compute Optimization: block ptr tiling](exp_chunk_fwd_001.md) — 主题相关：共享 aicore 单元利用/scalar 削减, matrix-operations
- 相关主题: [Compute Optimization: block ptr abstraction](exp_chunk_kkt_001.md) — 主题相关：共享 aicore 单元利用/scalar 削减, matrix-operations
- 相关主题: [Compute Optimization: tiling strategy](exp_cube1_qk_gemm_001.md) — 主题相关：共享 aicore 单元利用/scalar 削减, matrix-operations
- 相关主题: [Compute Optimization: mixed precision compute](exp_flashattn_bwd_kv_001.md) — 主题相关：共享 aicore 单元利用/scalar 削减, matrix-operations
- 相关主题: [Compute Optimization: Operator Fusion](exp_fused_delta_rule_001.md) — 主题相关：共享 aicore 单元利用/scalar 削减, matrix-operations
- 相关主题: [Compute Optimization: deep kernel fusion](exp_fused_delta_rule_npu_001.md) — 主题相关：共享 aicore 单元利用/scalar 削减, matrix-operations
- 相关主题: [Compute Optimization: operator fusion](exp_fused_entropy_001.md) — 主题相关：共享 aicore 单元利用/scalar 削减, matrix-operations
- 相关主题: [Compute Optimization: operator fusion](exp_fused_gemm_scale_001.md) — 主题相关：共享 aicore 单元利用/scalar 削减, matrix-operations
- 相关主题: [Compute Optimization: operator fusion](exp_fused_index_001.md) — 主题相关：共享 aicore 单元利用/scalar 削减, matrix-operations
- 相关主题: [Compute Optimization: block tiling strategy](exp_matmul_bias_fused_001.md) — 主题相关：共享 aicore 单元利用/scalar 削减, matrix-operations
- 相关主题: [Compute Optimization: small tile strategy](exp_matmul_triton_base_001.md) — 主题相关：共享 aicore 单元利用/scalar 削减, matrix-operations
- 相关主题: [Compute Optimization: vector parallelism strategy](exp_matmul_vec_parallel_001.md) — 主题相关：共享 aicore 单元利用/scalar 削减, matrix-operations
- 相关主题: [Compute Optimization: kernel fusion](exp_mhc_fused_gemm_sqrsum_001.md) — 主题相关：共享 aicore 单元利用/scalar 削减, matrix-operations
- 相关主题: [Compute Optimization: Operator Fusion](exp_mhc_post_fusion_001.md) — 主题相关：共享 aicore 单元利用/scalar 削减, matrix-operations
- 相关主题: [Compute Optimization: grid stride scheduling](exp_partial_matmul_001.md) — 主题相关：共享 aicore 单元利用/scalar 削减, matrix-operations
- 相关主题: [Compute Optimization: high precision accumulation](exp_recompute_w_u_001.md) — 主题相关：共享 aicore 单元利用/scalar 削减, matrix-operations
- 相关主题: [Compute Optimization: fused rope computation](exp_rope_matmul_fused_001.md) — 主题相关：共享 aicore 单元利用/scalar 削减, matrix-operations
- 相关主题: [Compute Optimization: autotuning block size](exp_sdpa_bwd_q_001.md) — 主题相关：共享 aicore 单元利用/scalar 削减, matrix-operations
- 相关主题: [Compute Optimization: online softmax](exp_sdpa_infer_ascend_001.md) — 主题相关：共享 aicore 单元利用/scalar 削减, matrix-operations
- 相关主题: [Compute Optimization: constexpr dual path full vs boundary](exp_solve_tril_003.md) — 主题相关：共享 aicore 单元利用/scalar 削减
- 相关主题: [Compute Optimization: kernel fusion](exp_ssm_gated_delta_001.md) — 主题相关：共享 aicore 单元利用/scalar 削减, matrix-operations
- 相关主题: [Compute Optimization: interleaved vectorization](exp_swiglu_oai_001.md) — 主题相关：共享 aicore 单元利用/scalar 削减
- 相关主题: [Compute Optimization: micro kernel tiling](exp_tril_solve_001.md) — 主题相关：共享 aicore 单元利用/scalar 削减, matrix-operations

<!-- okf:related:end -->
