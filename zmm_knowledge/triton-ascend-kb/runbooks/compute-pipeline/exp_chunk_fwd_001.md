---
schema_version: okf.v1
kind: operator_optimization
type: optimization_runbook
source_family: runbooks
category: compute-pipeline
dimension: 2.1
title: 'Compute Optimization: block ptr tiling'
description: 'ssm 场景下，Compute Optimization: block ptr tiling，收益 35%'
tags: [AICore 单元利用/Scalar 削减, matrix-operations, ssm]
confidence: verified
status: verified
created_at: 2026-08-21T00:00:00Z
updated_at: 2026-08-21T00:00:00Z
---

## Compute Optimization: block ptr tiling

【优化维度】：AICore 单元利用/Scalar 削减
【泛化层级】：条件性（迁移条件：硬件 910b2, 910b3；失效边界：T < 128 时 chunk 化收益有限；BT < 16 时 tl.dot 矩阵单元利用率不足；依赖 cu_seqlens 变长索引，等长 batch 可简化；门控 exp(g) 需在 fp32 下计算，低精度下大正/负值易溢出）

### 原则
- **block ptr tiling**：Utilizes `tl.make_block_ptr` to efficiently load 2D blocks of inputA, inputB, inputC, and H tensors. This optimizes memory coalescing and leverages hardware load/store units to maximize bandwidth utilization.
- **fused gating masking**：Integrates the exponential gating logic (`exp(g)`) and causal masking (`m_A`) directly within the main computation loop. This fusion reduces memory traffic by avoiding the storage and reloading of intermediate attention matrices.
- **mixed precision accumulation**：Performs matrix multiplications (`tl.dot`) and accumulates results in `float32` (`b_o`, `b_A`) before casting to the output dtype. This ensures numerical stability during the recurrent state updates and attention score accumulation.
- **layout optimization**：Handles the implicit data layout requirements for Tensor Core (MMA) operations. The code structure ensures that data loaded into registers is aligned for `tl.dot` operations, minimizing the overhead associated with layout conversions (e.g., row-major to column-major).

### 代码/骨架出处
- `TritonPath`:
```python
chunk_fwd_kernel_o
```

**坏实践（反例）**：把门控指数、causal mask、score 矩阵各自物化成中间张量再逐步计算 → 中间 score 矩阵反复读写 HBM，带宽翻倍 → ✅ 正解即本条：`exp(g)` 门控与 causal mask 融合进主循环，`tl.dot` 结果在 fp32 累加器中聚合后再 downcast。

**已知实例**：ssm §raw (`experiences/ssm/exp_chunk_fwd_001.yaml`)
- 声称性能收益：35%
- 备注：This kernel is designed for chunkwise linear attention or state-space models. It supports variable-length sequences via `cu_seqlens` and includes logic for recurrent state updates (`h`).

<!-- okf:related:start -->

# 相关

- 相关主题: [matmul 必须用 tl.make_block_ptr，禁止退化为独立标量 tl.load](../memory-sram/block-ptr-vs-scalar-load.md) — 同属优化知识，主题相关
- 相关主题: [Memory Access Optimization: vectorized offset generation](../memory-sram/exp_attn_causal_mask_001.md) — 主题相关：共享 matrix-operations
- 相关主题: [Memory Access Optimization: block pointer abstraction](../memory-sram/exp_attn_ptr_setup_001.md) — 主题相关：共享 matrix-operations
- 相关主题: [Memory Access Optimization: block ptr utilization](../memory-sram/exp_causal_conv1d_bwd_001.md) — 同属优化知识，主题相关
- 相关主题: [Memory Access Optimization: make block ptr hardware acceleration](../memory-sram/exp_causal_conv1d_update_003.md) — 同属优化知识，主题相关
- 相关主题: [Memory Access Optimization: chunked recurrence strategy](../memory-sram/exp_chunk_gated_delta_001.md) — 主题相关：共享 matrix-operations, ssm
- 相关主题: [Memory Access Optimization: tiling strategy](../memory-sram/exp_flash_attn_001.md) — 主题相关：共享 matrix-operations
- 相关主题: [Memory Access Optimization: indirect load scalarization](../memory-sram/exp_grouped_attn_stage1_001.md) — 主题相关：共享 matrix-operations
- 相关主题: [Memory Access Optimization: operator fusion](../memory-sram/exp_hstu_attn_fwd_001.md) — 主题相关：共享 matrix-operations
- 相关主题: [Memory Access Optimization: block ptr utilization](../memory-sram/exp_paged_attn_decode_001.md) — 主题相关：共享 matrix-operations
- 相关主题: [Memory Access Optimization: Paged Memory Access Strategy](../memory-sram/exp_paged_attn_flash_001.md) — 主题相关：共享 matrix-operations
- 相关主题: [Memory Access Optimization: paged kv indirection](../memory-sram/exp_paged_prefill_001.md) — 主题相关：共享 matrix-operations
- 相关主题: [Memory Access Optimization: dynamic specialization with aligned vector load](../memory-sram/exp_rope_forward_001.md) — 主题相关：共享 matrix-operations
- 相关主题: [Memory Access Optimization: block ptr abstraction](../memory-sram/exp_sdpa_infer_001.md) — 主题相关：共享 matrix-operations
- 相关主题: [Memory Access Optimization: block ptr utilization](../memory-sram/exp_topk_pad_001.md) — 同属优化知识，主题相关
- 相关主题: [Cube/Vector 流水线编排 + 序列分半（down kernel）适配片上内存](../portability/exp_attn_bwd_q_down_001.md) — 主题相关：共享 matrix-operations
- 相关主题: [反向 KV kernel：HIVM CV 均衡 + 双缓冲 + 非对称 tile（128×64）+ 序列分半](../portability/exp_flashattn_bwd_kv_ascend_001.md) — 主题相关：共享 matrix-operations
- 相关主题: [Compute Optimization: mixed precision compute](exp_attn_bwd_q_001.md) — 主题相关：共享 aicore 单元利用/scalar 削减, matrix-operations
- 相关主题: [Compute Optimization: blocking strategy](exp_attn_qk_cube_001.md) — 主题相关：共享 aicore 单元利用/scalar 削减, matrix-operations
- 相关主题: [Compute Optimization: attention sinks mechanism](exp_attn_sinks_prefill_001.md) — 主题相关：共享 aicore 单元利用/scalar 削减, matrix-operations
- 相关主题: [Compute Optimization: block ptr tiling](exp_block_inverse_001.md) — 主题相关：共享 aicore 单元利用/scalar 削减, matrix-operations
- 相关主题: [Compute Optimization: block ptr utilization](exp_block_matmul_inv_001.md) — 主题相关：共享 aicore 单元利用/scalar 削减, matrix-operations
- 相关主题: [Compute Optimization: w loop unrolling with threshold](exp_causal_conv1d_update_002.md) — 主题相关：共享 aicore 单元利用/scalar 削减
- 相关主题: [Compute Optimization: block ptr abstraction](exp_chunk_kkt_001.md) — 主题相关：共享 aicore 单元利用/scalar 削减, matrix-operations, ssm
- 相关主题: [Compute Optimization: tiling strategy](exp_cube1_qk_gemm_001.md) — 主题相关：共享 aicore 单元利用/scalar 削减, matrix-operations
- 相关主题: [Compute Optimization: mixed precision compute](exp_flashattn_bwd_kv_001.md) — 主题相关：共享 aicore 单元利用/scalar 削减, matrix-operations
- 相关主题: [Compute Optimization: Operator Fusion](exp_fused_delta_rule_001.md) — 主题相关：共享 aicore 单元利用/scalar 削减, matrix-operations, ssm
- 相关主题: [Compute Optimization: deep kernel fusion](exp_fused_delta_rule_npu_001.md) — 主题相关：共享 aicore 单元利用/scalar 削减, matrix-operations, ssm
- 相关主题: [Compute Optimization: operator fusion](exp_fused_entropy_001.md) — 主题相关：共享 aicore 单元利用/scalar 削减, matrix-operations
- 相关主题: [Compute Optimization: operator fusion](exp_fused_gemm_scale_001.md) — 主题相关：共享 aicore 单元利用/scalar 削减, matrix-operations
- 相关主题: [Compute Optimization: operator fusion](exp_fused_index_001.md) — 主题相关：共享 aicore 单元利用/scalar 削减, matrix-operations
- 相关主题: [标量消除: block_ptr boundary_check 与 i32 比较定向消除](exp_kda_bwd_intra_scalar_opt_001.md) — 主题相关：共享 aicore 单元利用/scalar 削减, ssm
- 相关主题: [计算优化: 对角块标量循环改 tl.dot](exp_kda_bwd_scalar_loop_to_dot_001.md) — 主题相关：共享 aicore 单元利用/scalar 削减, ssm
- 相关主题: [标量消除: i64/i32 标量降级定向消除](exp_kda_bwd_wy_scalar_opt_001.md) — 主题相关：共享 aicore 单元利用/scalar 削减, ssm
- 相关主题: [计算优化: phase split 与 barrier 外提](exp_kda_bwd_wy_structure_opt_001.md) — 主题相关：共享 aicore 单元利用/scalar 削减, ssm
- 相关主题: [Compute Optimization: npu simd vectorization](exp_layernorm_npu_simd_001.md) — 主题相关：共享 aicore 单元利用/scalar 削减
- 相关主题: [Compute Optimization: block tiling strategy](exp_matmul_bias_fused_001.md) — 主题相关：共享 aicore 单元利用/scalar 削减, matrix-operations
- 相关主题: [Compute Optimization: small tile strategy](exp_matmul_triton_base_001.md) — 主题相关：共享 aicore 单元利用/scalar 削减, matrix-operations
- 相关主题: [Compute Optimization: vector parallelism strategy](exp_matmul_vec_parallel_001.md) — 主题相关：共享 aicore 单元利用/scalar 削减, matrix-operations
- 相关主题: [Compute Optimization: kernel fusion](exp_mhc_fused_gemm_sqrsum_001.md) — 主题相关：共享 aicore 单元利用/scalar 削减, matrix-operations
- 相关主题: [Compute Optimization: Operator Fusion](exp_mhc_post_fusion_001.md) — 主题相关：共享 aicore 单元利用/scalar 削减, matrix-operations
- 相关主题: [Compute Optimization: grid stride scheduling](exp_partial_matmul_001.md) — 主题相关：共享 aicore 单元利用/scalar 削减, matrix-operations
- 相关主题: [Compute Optimization: high precision accumulation](exp_recompute_w_u_001.md) — 主题相关：共享 aicore 单元利用/scalar 削减, matrix-operations
- 相关主题: [Compute Optimization: fused rope computation](exp_rope_matmul_fused_001.md) — 主题相关：共享 aicore 单元利用/scalar 削减, matrix-operations
- 相关主题: [Compute Optimization: vectorized non interleaved path](exp_rope_siso_001.md) — 主题相关：共享 aicore 单元利用/scalar 削减, matrix-operations
- 相关主题: [Compute Optimization: autotuning block size](exp_sdpa_bwd_q_001.md) — 主题相关：共享 aicore 单元利用/scalar 削减, matrix-operations
- 相关主题: [Compute Optimization: online softmax](exp_sdpa_infer_ascend_001.md) — 主题相关：共享 aicore 单元利用/scalar 削减, matrix-operations
- 相关主题: [Compute Optimization: kernel fusion](exp_ssm_gated_delta_001.md) — 主题相关：共享 aicore 单元利用/scalar 削减, matrix-operations, ssm
- 相关主题: [Compute Optimization: micro kernel tiling](exp_tril_solve_001.md) — 主题相关：共享 aicore 单元利用/scalar 削减, matrix-operations

<!-- okf:related:end -->
