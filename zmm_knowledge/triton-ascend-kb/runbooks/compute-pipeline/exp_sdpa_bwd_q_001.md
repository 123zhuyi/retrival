---
schema_version: okf.v1
kind: operator_optimization
type: optimization_runbook
source_family: runbooks
category: compute-pipeline
dimension: 2.1
title: 'Compute Optimization: autotuning block size'
description: 'attention 场景下，Compute Optimization: autotuning block size，收益 30%'
tags: [AICore 单元利用/Scalar 削减, matrix-operations, attention]
confidence: verified
status: verified
created_at: 2026-08-21T00:00:00Z
updated_at: 2026-08-21T00:00:00Z
---

## Compute Optimization: autotuning block size

【优化维度】：AICore 单元利用/Scalar 削减
【泛化层级】：条件性（迁移条件：硬件 910b2, 910b3, 91093, 91095；失效边界：BLOCK_R/BLOCK_C 的 64/128 选择依赖 autotune 覆盖目标 shape；softmax 梯度涉及指数与减法，纯 bf16 累加会数值不稳（混合精度是硬性前提）；S < 128 时 autotune 收益有限）

### 原则
- **autotuning block size**：The kernel utilizes @triton.autotune to dynamically select the optimal BLOCK_R and BLOCK_C (64 or 128) configurations. This allows the kernel to adapt to varying sequence lengths and hardware cache hierarchies, maximizing data reuse within the register file and L1 cache.
- **mixed precision computation**：The kernel employs a mixed-precision strategy, using bfloat16 (LOW_TYPE) for memory loads/stores to save bandwidth, while performing accumulation in float32 (HIGH_TYPE). This is critical for the backward pass to maintain numerical stability during the exponential and subtraction operations involved in the softmax gradient calculation.
- **multibuffer prefetching**：The configuration includes a 'multibuffer' option. This enables software pipelining to overlap memory access latency with computation, ensuring the arithmetic units are not stalled while waiting for inputB and inputC tiles to be loaded from global memory.
- **vectorized memory access**：Pointer arithmetic is carefully constructed to ensure memory accesses are coalesced. Masks are applied to handle boundary conditions (sequence length not divisible by block size) without branching, maintaining high throughput.

### 代码/骨架出处
- `TritonPath`:
```python
kernel_sdpa_bwd_q
```

**坏实践（反例）**：反向梯度全程 bf16 计算（含 exp 与减法）→ softmax 梯度数值不稳 → ✅ 正解即本条：bf16 存取省带宽 + fp32 累加 + autotune 选 BLOCK_R/BLOCK_C + multibuffer 预取。

**已知实例**：attention §raw (`experiences/attention/exp_sdpa_bwd_q_001.yaml`)
- 声称性能收益：30%
- 备注：Optimized for the backward pass of scaled dot-product attention-like operators (inputA 梯度计算). The mixed precision strategy is essential to avoid gradient instability.
- 已知实例（硬件实测，ascend910b，fp32-ieee dot 骨架，BM/BN=64/64，2026-09-24）：**causal m_lo/m_hi 三角区裁剪**（对齐 CANN `IsCubeBlockNeedCompute`）——dkdv 内层 m-loop 起点 = `max(0,(nb*BN-DELTA)//BM*BM)`，dq 内层 n-loop 上界 = `min(S_k,(mb+1)*BM+DELTA)`；Sq=Sk（DELTA=0）、BM=BN 时精确等腰三角，每 program 平均只扫一半。3 个 causal bf16 D128 case 扫描量减半（c47 213→109ms、c45 107→55ms、c41 54→28ms），verify 50/50，geomean 0.0598→0.0644（1.077×）。证据：`output/opt_iter_6.py` + `opt_iter_6/optimized_perf_result.json.speedup_vs_torch=0.0644 vs opt_iter_3/optimized_perf_result.json.speedup_vs_torch=0.0598`。

<!-- okf:related:start -->

# 相关

- 相关主题: [Memory Access Optimization: block size tuning](../memory-sram/exp_sum_scale_001.md) — 同属优化知识，主题相关
- 相关主题: [Scheduling Optimization: dynamic autotuning](exp_addmm_ascend_001.md) — 主题相关：共享 matrix-operations
- 相关主题: [Scheduling Optimization: dual task pipelining](exp_attn_ascend_dual_pipeline_001.md) — 主题相关：共享 attention, matrix-operations
- 相关主题: [Compute Optimization: mixed precision compute](exp_attn_bwd_q_001.md) — 主题相关：共享 aicore 单元利用/scalar 削减, attention, matrix-operations
- 相关主题: [Compute Optimization: blocking strategy](exp_attn_qk_cube_001.md) — 主题相关：共享 aicore 单元利用/scalar 削减, attention, matrix-operations
- 相关主题: [Compute Optimization: attention sinks mechanism](exp_attn_sinks_prefill_001.md) — 主题相关：共享 aicore 单元利用/scalar 削减, attention, matrix-operations
- 相关主题: [Compute Optimization: block ptr tiling](exp_block_inverse_001.md) — 主题相关：共享 aicore 单元利用/scalar 削减, matrix-operations
- 相关主题: [Compute Optimization: block ptr utilization](exp_block_matmul_inv_001.md) — 主题相关：共享 aicore 单元利用/scalar 削减, matrix-operations
- 相关主题: [Scheduling Optimization: grouped scheduling](exp_bmm_grouped_001.md) — 主题相关：共享 matrix-operations
- 相关主题: [Compute Optimization: block ptr tiling](exp_chunk_fwd_001.md) — 主题相关：共享 aicore 单元利用/scalar 削减, matrix-operations
- 相关主题: [Compute Optimization: block ptr abstraction](exp_chunk_kkt_001.md) — 主题相关：共享 aicore 单元利用/scalar 削减, matrix-operations
- 相关主题: [Compute Optimization: tiling strategy](exp_cube1_qk_gemm_001.md) — 主题相关：共享 aicore 单元利用/scalar 削减, matrix-operations
- 相关主题: [Compute Optimization: mixed precision compute](exp_flashattn_bwd_kv_001.md) — 主题相关：共享 aicore 单元利用/scalar 削减, attention, matrix-operations
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
- 相关主题: [Compute Optimization: vectorized non interleaved path](exp_rope_siso_001.md) — 主题相关：共享 aicore 单元利用/scalar 削减, matrix-operations
- 相关主题: [Compute Optimization: online softmax](exp_sdpa_infer_ascend_001.md) — 主题相关：共享 aicore 单元利用/scalar 削减, attention, matrix-operations
- 相关主题: [Compute Optimization: kernel fusion](exp_ssm_gated_delta_001.md) — 主题相关：共享 aicore 单元利用/scalar 削减, matrix-operations
- 相关主题: [Compute Optimization: micro kernel tiling](exp_tril_solve_001.md) — 主题相关：共享 aicore 单元利用/scalar 削减, matrix-operations

<!-- okf:related:end -->
