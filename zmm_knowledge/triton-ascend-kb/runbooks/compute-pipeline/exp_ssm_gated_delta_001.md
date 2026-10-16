---
schema_version: okf.v1
kind: operator_optimization
type: optimization_runbook
source_family: runbooks
category: compute-pipeline
dimension: 2.1
title: 'Compute Optimization: kernel fusion'
description: 'ssm 场景下，Compute Optimization: kernel fusion，收益 50%'
tags: [AICore 单元利用/Scalar 削减, matrix-operations, ssm]
confidence: verified
status: verified
created_at: 2026-08-21T00:00:00Z
updated_at: 2026-08-21T00:00:00Z
---

## Compute Optimization: kernel fusion

【优化维度】：AICore 单元利用/Scalar 削减
【泛化层级】：条件性（迁移条件：硬件 910b2, 910b3；失效边界：BK/BV 需按 L1 容量调（备注自述，典型 64x64/128x128），超出即 spill；递推本质串行，并行度仅来自 head/value 维分块；IS_VARLEN/IS_CONTINUOUS_BATCHING/IS_SPEC_DECODING 启发式组合多时编译产物膨胀）

### 原则
- **kernel fusion**：Fuses the recurrent state update loop, gating logic (g, beta), and output projection into a single kernel. This eliminates the need to write and read back the intermediate hidden state (h) to global memory for every time step, significantly reducing memory bandwidth pressure.
- **2d tiling strategy**：Implements a 2D blocking strategy on the hidden state matrix (inputB x inputC) using block sizes BK and BV. This parallelizes the inherently sequential recurrence across the head and value dimensions, allowing the hardware to utilize vector units effectively for the matrix operations (outer products and dot products) within each step.
- **heuristic specialization**：Uses compile-time heuristics to handle complex deployment scenarios like Variable Length sequences (IS_VARLEN), Continuous Batching (IS_CONTINUOUS_BATCHING), and Speculative Decoding (IS_SPEC_DECODING). This generates optimized code paths for specific input shapes and memory layouts without runtime branching overhead.
- **fp32 accumulation**：Performs the accumulation of the hidden state `b_h` in float32 precision. This is critical for maintaining numerical stability in gated recurrent networks where the state is updated multiplicatively over many steps, preventing overflow or underflow.
- **inplace state update**：Supports in-place updates for the final state (`INPLACE_FINAL_STATE`) and optimized indexing for continuous batching. This reduces memory footprint and avoids redundant data copies when managing the KV cache or state cache in LLM inference systems.

### 代码/骨架出处
- `TritonPath`:
```python
raw_experiences/extracted_kernels/151-fused_recurrent_gated_delta_rule_fwd_kernel.py
```

**坏实践（反例）**：递推每个 timestep 把隐状态 h 写回 HBM 再读回 → 每步一次 HBM 往返，递推越长越慢 → ✅ 正解即本条：h 全程片上驻留（fp32 累加）+ 门控与输出投影融合 + 编译期裁剪部署场景分支。

**已知实例**：ssm §raw (`experiences/ssm/exp_ssm_gated_delta_001.yaml`)
- 声称性能收益：50%
- 备注：This kernel is specifically designed for selective state-space-model-like architectures. It handles the delta-rule style update efficiently. Ensure BK and BV are tuned to the L1 cache size of the specific hardware (e.g., 64x64 or 128x128).

<!-- okf:related:start -->

# 相关

- 相关主题: [Scheduling Optimization: dynamic autotuning](exp_addmm_ascend_001.md) — 主题相关：共享 matrix-operations
- 相关主题: [Scheduling Optimization: dual task pipelining](exp_attn_ascend_dual_pipeline_001.md) — 主题相关：共享 matrix-operations
- 相关主题: [Compute Optimization: mixed precision compute](exp_attn_bwd_q_001.md) — 主题相关：共享 aicore 单元利用/scalar 削减, matrix-operations
- 相关主题: [Compute Optimization: blocking strategy](exp_attn_qk_cube_001.md) — 主题相关：共享 aicore 单元利用/scalar 削减, matrix-operations
- 相关主题: [Compute Optimization: attention sinks mechanism](exp_attn_sinks_prefill_001.md) — 主题相关：共享 aicore 单元利用/scalar 削减, matrix-operations
- 相关主题: [Compute Optimization: block ptr tiling](exp_block_inverse_001.md) — 主题相关：共享 aicore 单元利用/scalar 削减, matrix-operations
- 相关主题: [Compute Optimization: block ptr utilization](exp_block_matmul_inv_001.md) — 主题相关：共享 aicore 单元利用/scalar 削减, matrix-operations
- 相关主题: [Scheduling Optimization: grouped scheduling](exp_bmm_grouped_001.md) — 主题相关：共享 matrix-operations
- 相关主题: [Compute Optimization: block ptr tiling](exp_chunk_fwd_001.md) — 主题相关：共享 aicore 单元利用/scalar 削减, matrix-operations, ssm
- 相关主题: [Compute Optimization: block ptr abstraction](exp_chunk_kkt_001.md) — 主题相关：共享 aicore 单元利用/scalar 削减, matrix-operations, ssm
- 相关主题: [Compute Optimization: tiling strategy](exp_cube1_qk_gemm_001.md) — 主题相关：共享 aicore 单元利用/scalar 削减, matrix-operations
- 相关主题: [Scheduling Optimization: grid stride loop](exp_element_mul_001.md) — 主题相关：共享 matrix-operations
- 相关主题: [Compute Optimization: mixed precision compute](exp_flashattn_bwd_kv_001.md) — 主题相关：共享 aicore 单元利用/scalar 削减, matrix-operations
- 相关主题: [Compute Optimization: Operator Fusion](exp_fused_delta_rule_001.md) — 主题相关：共享 aicore 单元利用/scalar 削减, matrix-operations, ssm
- 相关主题: [Compute Optimization: deep kernel fusion](exp_fused_delta_rule_npu_001.md) — 主题相关：共享 aicore 单元利用/scalar 削减, matrix-operations, ssm
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
- 相关主题: [Compute Optimization: vectorized non interleaved path](exp_rope_siso_001.md) — 主题相关：共享 aicore 单元利用/scalar 削减, matrix-operations
- 相关主题: [Compute Optimization: autotuning block size](exp_sdpa_bwd_q_001.md) — 主题相关：共享 aicore 单元利用/scalar 削减, matrix-operations
- 相关主题: [Compute Optimization: online softmax](exp_sdpa_infer_ascend_001.md) — 主题相关：共享 aicore 单元利用/scalar 削减, matrix-operations
- 相关主题: [Compute Optimization: micro kernel tiling](exp_tril_solve_001.md) — 主题相关：共享 aicore 单元利用/scalar 削减, matrix-operations

<!-- okf:related:end -->
