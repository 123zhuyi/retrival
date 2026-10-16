---
schema_version: okf.v1
kind: operator_optimization
type: optimization_runbook
source_family: runbooks
category: compute-pipeline
dimension: 2.1
title: 'Compute Optimization: deep kernel fusion'
description: 'ssm 场景下，Compute Optimization: deep kernel fusion，收益 35%'
tags: [AICore 单元利用/Scalar 削减, matrix-operations, ssm]
confidence: verified
status: verified
created_at: 2026-08-21T00:00:00Z
updated_at: 2026-08-21T00:00:00Z
---

## Compute Optimization: deep kernel fusion

【优化维度】：AICore 单元利用/Scalar 削减
【泛化层级】：条件性（迁移条件：硬件 910b2, 910b3；失效边界：BK×BV 状态驻留寄存器受寄存器堆容量约束，过大即 spill；时间维是串行循环，长序列收益来自省带宽而非并行度；全 fp32 累加使寄存器占用翻倍，需与 BK/BV 一起权衡）

### 原则
- **deep kernel fusion**：Fuses multiple stages including Softplus, Sigmoid, Exponential gating, Matrix-Vector multiplication (outer product), and Reductions into a single kernel. This eliminates global memory access for intermediate activations (like the gated state or intermediate projections), significantly reducing memory bandwidth pressure.
- **register level state persistence**：The recurrent hidden state 'h' (shape [BK, BV]) is maintained entirely in registers across the time dimension loop 'for i in range(T)'. This avoids the high latency of writing and reading the state back to global memory (HBM) at every timestep, which is the primary bottleneck for recurrent architectures.
- **numerically stable softplus**：Implements a numerically stable Softplus activation using a threshold check ('softplus_threshold'). For inputs where 'beta_x > threshold', it defaults to the identity function 'x' to prevent overflow in 'tl.exp', ensuring robustness on NPU hardware which may have different floating point behavior than GPUs.
- **mixed precision accumulation**：Explicitly promotes loaded vectors ('b_q', 'b_k', 'b_v', 'b_a', etc.) to 'tl.float32' before performing arithmetic operations. This preserves precision during the recurrent accumulation of the hidden state 'h', preventing gradient degradation or instability during long sequence inference or training.
- **efficient masking and vectorization**：Utilizes pointer arithmetic and 'tl.load' with masks to handle variable sequence lengths ('IS_VARLEN') and boundary conditions. This allows the kernel to process batches with different lengths efficiently without dynamic branching or padding overhead.

### 代码/骨架出处
- `TritonPath`:
```python
raw_experiences/extracted_kernels/179-fused_sigmoid_gating_delta_rule_update_npu_kernel.py
```

**坏实践（反例）**：门控（softplus/sigmoid/exp）、外积、归约拆成多个 kernel 逐步执行 → 中间激活反复落 HBM，递推 loop 被带宽锁死 → ✅ 正解即本条：深度融合 + 状态 h 全程寄存器驻留 + 载入即升 fp32 保精度。

**已知实例**：ssm §raw (`experiences/ssm/exp_fused_delta_rule_npu_001.yaml`)
- 声称性能收益：35%
- 备注：This kernel is specifically optimized for NPU architectures (like Ascend 910B) where memory bandwidth is a limiting factor. The block sizes BK and BV should be tuned based on the specific NPU's SRAM size and register file capacity to avoid spilling.

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
- 相关主题: [Compute Optimization: mixed precision compute](exp_flashattn_bwd_kv_001.md) — 主题相关：共享 aicore 单元利用/scalar 削减, matrix-operations
- 相关主题: [Compute Optimization: Operator Fusion](exp_fused_delta_rule_001.md) — 主题相关：共享 aicore 单元利用/scalar 削减, matrix-operations, ssm
- 相关主题: [Compute Optimization: operator fusion](exp_fused_entropy_001.md) — 主题相关：共享 aicore 单元利用/scalar 削减, matrix-operations
- 相关主题: [Compute Optimization: operator fusion](exp_fused_gemm_scale_001.md) — 主题相关：共享 aicore 单元利用/scalar 削减, matrix-operations
- 相关主题: [Compute Optimization: operator fusion](exp_fused_index_001.md) — 主题相关：共享 aicore 单元利用/scalar 削减, matrix-operations
- 相关主题: [Scheduling Optimization: deep operator fusion](exp_fused_transformer_block_001.md) — 同属优化知识，主题相关
- 相关主题: [标量消除: i64/i32 标量降级定向消除](exp_kda_bwd_wy_scalar_opt_001.md) — 主题相关：共享 aicore 单元利用/scalar 削减, ssm
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
