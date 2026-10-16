---
schema_version: okf.v1
kind: operator_optimization
type: optimization_runbook
source_family: runbooks
category: compute-pipeline
dimension: 2.1
title: '计算优化: phase split 与 barrier 外提'
description: 'ssm 场景下，计算优化: phase split 与 barrier 外提，收益 30%'
tags: [AICore 单元利用/Scalar 削减, ssm]
confidence: provisional
status: verified
created_at: 2026-08-21T00:00:00Z
updated_at: 2026-08-21T00:00:00Z
---

## 计算优化: phase split 与 barrier 外提

【优化维度】：AICore 单元利用/Scalar 削减
【泛化层级】：条件性（迁移条件：硬件 910b2, 910b3；失效边界：phase split 要求被拆出的分支（i_k==0）确实与主循环无数据依赖，且 dv2 写入完成后 Phase 2 才能读 dv——barrier 位置错会读脏数据；barrier 外提要求循环体内无跨迭代依赖；建议先做标量消除再做结构优化（备注自述））

### 原则
- **优化1-Phase split**：将i_k==0分支（计算dA/dv2/db）外提为独立的V-only循环，消除内循环的条件分支
- **优化2-barrier外提**：将tl.debug_barrier()从i_v循环内移到循环后，barrier次数从NV×NK降为NK
- **优化3-循环不变量外提**：预计算BT_arange[:, None]*V等循环不变地址分量，循环内仅做简单偏移
- **优化4-stride预计算**：预计算H*K和H*inputC，避免循环内重复乘法
- **优化5-inputB%BK==0快速路径**：当K能被BK整除时，BK_mask全True，跳过mask生成
- **注意**：Phase split需确保dv2写入完成后Phase 2才能读取dv，barrier位置需正确
- **典型收益**：Phase split和barrier外提各贡献显著延迟下降，与其他标量消除优化叠加达到3.33x加速

### 代码/骨架出处
- `Triton`:
```python
# Phase split 示例
# Phase 1: V-only loop（dA, dv2, db）
for i_v in tl.range(tl.cdiv(V, BV), num_stages=2):
    b_dA += tl.dot(b_dv, tl.trans(b_v)) * scalar
    b_dvb = tl.dot(b_A, b_dv)
    b_dv2 = b_dvb * b_beta[:, None]
    b_db += tl.sum(b_dvb * b_v, 1)
    tl.store(p_dv2, casted_b_dv2, mask=BV_2d_mask)
tl.debug_barrier()

# Phase 2: K×V nested loop（dq, dk, dw, dgk）
for i_k in range(...):
    for i_v in tl.range(...):
        # 无 if i_k==0 分支
        ...
```

**坏实践（反例）**：`tl.debug_barrier()` 放在 i_v 内层循环里（NV×NK 次）、`i_k==0` 分支留在嵌套循环内 → 每次迭代都付分支与全核同步开销 → ✅ 正解即本条：i_k==0 分支外提为独立 V-only Phase 1，barrier 移到循环后（降为 NK 次），循环不变量与 stride 预计算。

**已知实例**：ssm §raw (`experiences/ssm/exp_kda_bwd_wy_structure_opt_001.yaml`)
- 声称性能收益：30%
- 备注：此bundle聚焦kernel结构层面的优化，与标量消除优化正交互。实施时建议先完成标量消除，再做结构优化，便于独立验证每步效果。

<!-- okf:related:start -->

# 相关

- 相关主题: [反量化域必须按 x.dtype 分流（fp16 域 vs fp32 域）](../algorithm-numerics/dequant-domain-dtype-split.md) — 同属优化知识，主题相关
- 相关主题: [编译器融合消除中间 dtype 舍入时，拆分独立 kernel 强制 GM round-trip](../algorithm-numerics/split-kernel-gm-roundtrip.md) — 同属优化知识，主题相关
- 相关主题: [Memory Access Optimization: fused multi-tensor split and reshape](../memory-sram/exp_fused_qkvzba_split_001.md) — 同属优化知识，主题相关
- 相关主题: [Memory Access Optimization: fused split and dual-norm single-pass load](../memory-sram/exp_fused_split_qk_norm_001.md) — 同属优化知识，主题相关
- 相关主题: [Compute Optimization: mixed precision compute](exp_attn_bwd_q_001.md) — 主题相关：共享 aicore 单元利用/scalar 削减
- 相关主题: [Compute Optimization: blocking strategy](exp_attn_qk_cube_001.md) — 主题相关：共享 aicore 单元利用/scalar 削减
- 相关主题: [Compute Optimization: online softmax algorithm](exp_attn_reduce_stage2_001.md) — 主题相关：共享 aicore 单元利用/scalar 削减
- 相关主题: [Compute Optimization: attention sinks mechanism](exp_attn_sinks_prefill_001.md) — 主题相关：共享 aicore 单元利用/scalar 削减
- 相关主题: [Compute Optimization: vectorization alignment](exp_backward_sum_001.md) — 主题相关：共享 aicore 单元利用/scalar 削减
- 相关主题: [Compute Optimization: block ptr tiling](exp_block_inverse_001.md) — 主题相关：共享 aicore 单元利用/scalar 削减
- 相关主题: [Compute Optimization: block ptr utilization](exp_block_matmul_inv_001.md) — 主题相关：共享 aicore 单元利用/scalar 削减
- 相关主题: [Compute Optimization: pointer base precompute](exp_causal_conv1d_prefill_004.md) — 主题相关：共享 aicore 单元利用/scalar 削减
- 相关主题: [Compute Optimization: conv bias silu triple fusion](exp_causal_conv1d_prefill_006.md) — 主题相关：共享 aicore 单元利用/scalar 削减
- 相关主题: [Compute Optimization: dim is multiple mask elision](exp_causal_conv1d_prefill_007.md) — 主题相关：共享 aicore 单元利用/scalar 削减
- 相关主题: [Compute Optimization: w loop unrolling with threshold](exp_causal_conv1d_update_002.md) — 主题相关：共享 aicore 单元利用/scalar 削减
- 相关主题: [Compute Optimization: dim is multiple mask elimination](exp_causal_conv1d_update_005.md) — 主题相关：共享 aicore 单元利用/scalar 削减
- 相关主题: [Compute Optimization: numerical stability](exp_celoss_001.md) — 主题相关：共享 aicore 单元利用/scalar 削减
- 相关主题: [Compute Optimization: two pass algorithm](exp_celoss_bwd_001.md) — 主题相关：共享 aicore 单元利用/scalar 削减
- 相关主题: [Compute Optimization: numerical stability](exp_celoss_smooth_001.md) — 主题相关：共享 aicore 单元利用/scalar 削减
- 相关主题: [Compute Optimization: two pass algorithm](exp_celoss_smooth_bwd_001.md) — 主题相关：共享 aicore 单元利用/scalar 削减
- 相关主题: [Compute Optimization: chunked scan strategy](exp_chunk_cumsum_001.md) — 主题相关：共享 aicore 单元利用/scalar 削减, ssm
- 相关主题: [Compute Optimization: block ptr tiling](exp_chunk_fwd_001.md) — 主题相关：共享 aicore 单元利用/scalar 削减, ssm
- 相关主题: [Compute Optimization: block ptr abstraction](exp_chunk_kkt_001.md) — 主题相关：共享 aicore 单元利用/scalar 削减, ssm
- 相关主题: [Compute Optimization: online softmax algorithm](exp_crossentropy_001.md) — 主题相关：共享 aicore 单元利用/scalar 削减
- 相关主题: [Compute Optimization: grid stride scheduling](exp_crossentropy_backward_001.md) — 主题相关：共享 aicore 单元利用/scalar 削减
- 相关主题: [Compute Optimization: tiling strategy](exp_cube1_qk_gemm_001.md) — 主题相关：共享 aicore 单元利用/scalar 削减
- 相关主题: [Compute Optimization: gemm based histogram](exp_digit_hist_001.md) — 主题相关：共享 aicore 单元利用/scalar 削减
- 相关主题: [Compute Optimization: numerical stability reduction](exp_entropy_epilogue_001.md) — 主题相关：共享 aicore 单元利用/scalar 削减
- 相关主题: [Compute Optimization: split k parallelism](exp_flashattn_splitk_gqa_001.md) — 主题相关：共享 aicore 单元利用/scalar 削减
- 相关主题: [tl.trans 标量化灾难与寄存器重排替代（split/interleave/reshape 安全集）](triton-ascend-trans-scalarization.md) — 同属优化知识，主题相关

<!-- okf:related:end -->
