---
schema_version: okf.v1
kind: operator_optimization
type: optimization_runbook
source_family: runbooks
category: memory-sram
dimension: 1.2
title: 切分优化：A、B 均不转置 matmul 的 512B 行宽对齐分块
description: matmul 场景下，切分优化：A、B 均不转置 matmul 的 512B 行宽对齐分块，收益 10%
tags: [Tiling/SRAM 预算, aic, matmul]
confidence: verified
status: verified
created_at: 2026-08-21T00:00:00Z
updated_at: 2026-08-21T00:00:00Z
---

## 切分优化：A、B 均不转置 matmul 的 512B 行宽对齐分块

【优化维度】：Tiling/SRAM 预算
【泛化层级】：条件性（迁移条件：硬件 910b2, 910b3，fp32/fp16/bf16，m ≥128、n ≥256、k ≥256，A、B 均不转置；失效边界：分块行宽无法对齐 512B 整数倍（如 K 维过碎）时访存碎片抵消带宽收益；矩阵尺寸小于分块粒度时 tile 内 padding 浪费）

### 原则
- **分块策略**：适配Ascend后端512B行宽整数倍要求，M0=128、K0=256、N0=256划分分块行宽，单次行数最大化以充分发挥带宽
- **带宽优化**：分块行宽严格对齐512B整数倍，减少访存碎片，提升Ascend后端带宽利用率
- **数据复用**：大行数分块设计提升单次加载数据的计算复用率

### 代码/骨架出处
- `Triton`:
```python
@triton.jit def matmul_no_trans_kernel(x, y, output, M, N, K, blockM=128, blockK=256, blockN=256, **kwargs):
    # A、B都不转置 - 适配Ascend 512B行宽的分块代码
    pid = triton.program_id(0)
    rm = pid // (N // blockN)
    rn = pid % (N // blockN)
    # 按M0=128、K0=256、N0=256分块计算指针
    x_ptr = x + rm * blockM * K + triton.arange(0, blockM)[:, None] * K + triton.arange(0, blockK)[None, :]
    y_ptr = y + rn * blockN + triton.arange(0, blockK)[:, None] * N + triton.arange(0, blockN)[None, :]
    acc = triton.zeros((blockM, blockN), dtype=triton.float32)
    for k in range(0, K, blockK):
        a = triton.load(x_ptr, mask=triton.arange(0, blockK)[None, :] < K - k, other=0.0)
        b = triton.load(y_ptr, mask=triton.arange(0, blockK)[:, None] < K - k, other=0.0)
        acc += a @ b
        x_ptr += blockK
        y_ptr += blockK * N
    acc = acc.to(x.dtype)
    output_ptr = output + rm * blockM * N + triton.arange(0, blockM)[:, None] * N + triton.arange(0, blockN)[None, :]
    triton.store(output_ptr, acc, mask=triton.arange(0, blockM)[:, None] < M - rm * blockM and triton.arange(0, blockN)[None, :] < N - rn * blockN)
```

**坏实践（反例）**：分块行宽不对齐 512B 整数倍
```python
BLOCK_M, BLOCK_K, BLOCK_N = 96, 192, 192   # fp16 下行宽 192×2B=384B，非 512B 整数倍
for k in range(0, K, BLOCK_K):
    a = triton.load(x_ptr, ...)            # GM→UB 行访问跨对齐边界，产生访存碎片
```
后果：GM→UB 连续行访问被对齐边界切碎，访存事务数增多，后端带宽利用率下降。✅ 正解：M0=128、K0=256、N0=256，分块行宽严格对齐 512B 整数倍，单次加载行数最大化，提升数据计算复用率。

**已知实例**：matmul §raw (`experiences/matmul/exp_matmul_003.yaml`)
- 声称性能收益：10%
- 备注：适用于 A、B 矩阵均不转置的 matmul 场景，分块行宽严格对齐后端 512B 整数倍要求，最大化带宽利用

<!-- okf:related:start -->

# 相关

- 实践案例: [ChunkGatedDeltaRuleFwdH（GDN/KDA chunk 间状态递推前向）Ascend C → Triton 语义转化卡](../../ops/chunk-gated-delta-rule-fwd-h.md) — 主题相关：共享 matmul
- 实践案例: [FlashAttentionScoreGrad training-backward on Ascend NPU](../../ops/flash-attention-score-grad.md) — 主题相关：共享 matmul
- 实践案例: [FlashAttentionScore training-forward on Ascend NPU](../../ops/flash-attention-score.md) — 主题相关：共享 matmul
- 实践案例: [FusedFloydAttentionGrad (Floyd attention backward) on Ascend NPU](../../ops/fused-floyd-attention-grad.md) — 主题相关：共享 matmul
- 实践案例: [FusedFloydAttention (Floyd multi-dim attention with dual KV groups) on Ascend NPU](../../ops/fused-floyd-attention.md) — 主题相关：共享 matmul
- 实践案例: [MlaPreprocess (MLA attention preprocessing fusion) on Ascend NPU](../../ops/mla-preprocess.md) — 主题相关：共享 matmul
- 实践案例: [MlaPrologV3 (MLA pre-process) on Ascend NPU](../../ops/mla-prolog-v3.md) — 主题相关：共享 matmul
- 实践案例: [MlaProlog (MLA attention preprocessing — prolog V1) on Ascend NPU](../../ops/mla-prolog.md) — 主题相关：共享 matmul
- 实践案例: [RainFusionAttention (CATLASS block-sparse inference attention) on Ascend NPU](../../ops/rain-fusion-attention.md) — 主题相关：共享 matmul
- 实践案例: [RecurrentGatedDeltaRule (linear attention / SSM-style gated delta rule) on Ascend NPU](../../ops/recurrent-gated-delta-rule.md) — 主题相关：共享 matmul
- 开发指南: [MatMul / GEMM 高性能算子](../../reference/op-class-matmul.md) — 主题相关：共享 matmul
- 相关主题: [matmul core mapping 策略选择（列/行/对角）](../compute-pipeline/core-mapping-strategy-selection.md) — 主题相关：共享 matmul
- 相关主题: [大矩阵 matmul 用对角 core mapping 消除 L2 bank conflict](../compute-pipeline/diagonal-core-mapping-l2-bank-conflict.md) — 主题相关：共享 matmul
- 相关主题: [Scheduling Optimization: dynamic autotuning](../compute-pipeline/exp_addmm_ascend_001.md) — 主题相关：共享 matmul
- 相关主题: [Compute Optimization: block ptr utilization](../compute-pipeline/exp_block_matmul_inv_001.md) — 主题相关：共享 matmul
- 相关主题: [Scheduling Optimization: grouped scheduling](../compute-pipeline/exp_bmm_grouped_001.md) — 主题相关：共享 matmul
- 相关主题: [Compute Optimization: tiling strategy](../compute-pipeline/exp_cube1_qk_gemm_001.md) — 主题相关：共享 matmul
- 相关主题: [Compute Optimization: operator fusion](../compute-pipeline/exp_fused_gemm_scale_001.md) — 主题相关：共享 matmul
- 相关主题: [Scheduling Optimization: async pipeline scheduling](../compute-pipeline/exp_fused_matmul_norm_sigmoid_001.md) — 主题相关：共享 matmul
- 相关主题: [Scheduling Optimization: operator fusion](../compute-pipeline/exp_fused_matmul_rms_001.md) — 主题相关：共享 matmul
- 相关主题: [Scheduling Optimization: operator fusion](../compute-pipeline/exp_fused_matmul_rmsnorm_001.md) — 主题相关：共享 matmul
- 相关主题: [Scheduling Optimization: operator fusion](../compute-pipeline/exp_fused_matmul_rope_001.md) — 主题相关：共享 matmul
- 相关主题: [Scheduling Optimization: hardware unit isolation](../compute-pipeline/exp_fused_norm_rope_matmul_001.md) — 主题相关：共享 matmul
- 相关主题: [Scheduling Optimization: dynamic work stealing](../compute-pipeline/exp_grouped_gemm_001.md) — 主题相关：共享 matmul
- 相关主题: [Scheduling Optimization: dynamic load balancing](../compute-pipeline/exp_grouped_gemm_bkmajor_001.md) — 主题相关：共享 matmul
- 相关主题: [Scheduling Optimization: dynamic load balancing](../compute-pipeline/exp_grouped_gemm_bnmajor_001.md) — 主题相关：共享 matmul
- 相关主题: [计算优化: fixpipe 随路量化 + K 维循环展开 + 预取](../compute-pipeline/exp_matmul_002.md) — 主题相关：共享 aic, matmul
- 相关主题: [访存优化：大尺寸 matmul 的 256 分块与 L0C 复用](exp_matmul_001.md) — 主题相关：共享 aic, matmul
- 相关主题: [切分优化：A 不转置、B 转置 matmul 的 K0=256 统一分块](exp_matmul_004.md) — 主题相关：共享 aic, matmul, tiling/sram 预算
- 相关主题: [切分优化：双转置矩阵乘的 512B 对齐分块](exp_matmul_005.md) — 主题相关：共享 aic, matmul, tiling/sram 预算

<!-- okf:related:end -->
