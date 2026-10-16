---
schema_version: okf.v1
kind: operator_optimization
type: optimization_runbook
source_family: runbooks
category: memory-sram
dimension: 1.2
title: 切分优化：A 不转置、B 转置 matmul 的 K0=256 统一分块
description: matmul 场景下，切分优化：A 不转置、B 转置 matmul 的 K0=256 统一分块，收益 10%
tags: [Tiling/SRAM 预算, aic, matmul]
confidence: verified
status: verified
created_at: 2026-08-21T00:00:00Z
updated_at: 2026-08-21T00:00:00Z
---

## 切分优化：A 不转置、B 转置 matmul 的 K0=256 统一分块

【优化维度】：Tiling/SRAM 预算
【泛化层级】：条件性（迁移条件：硬件 910b2, 910b3，fp32/fp16/bf16，m ≥128、n ≥128、k ≥256，A 不转置、B 转置；失效边界：K <256 时 K0=256 固定分块产生大量尾部 mask，padding 占比升高；A、B 均不转置的场景应改用 M0/K0/N0 组合分块（见对应条目））

### 原则
- **分块策略**：统一分块行宽K0=256（满足512B整数倍），M0和N0可灵活调整，优先保证K0对齐
- **带宽优化**：单一K0分块减少分块适配开销，提升Ascend后端带宽利用率
- **兼容性**：M0/N0宽松限制适配不同尺寸矩阵的转置乘法场景

### 代码/骨架出处
- `Triton`:
```python
@triton.jit def matmul_a_no_trans_b_trans_kernel(x, y, output, M, N, K, blockM=256, blockK=256, blockN=256, **kwargs):
    # A不转置、B转置 - K0=256统一分块行宽
    pid = triton.program_id(0)
    rm = pid // (N // blockN)
    rn = pid % (N // blockN)
    # B转置后调整指针计算逻辑，K0固定为256
    x_ptr = x + rm * blockM * K + triton.arange(0, blockM)[:, None] * K + triton.arange(0, blockK)[None, :]
    y_ptr = y + rn * blockK * N + triton.arange(0, blockK)[:, None] + triton.arange(0, blockN)[None, :] * N
    acc = triton.zeros((blockM, blockN), dtype=triton.float32)
    for k in range(0, K, blockK):
        a = triton.load(x_ptr, mask=triton.arange(0, blockK)[None, :] < K - k, other=0.0)
        b = triton.load(y_ptr, mask=triton.arange(0, blockK)[:, None] < K - k, other=0.0)
        acc += a @ b
        x_ptr += blockK
        y_ptr += blockK
    acc = acc.to(x.dtype)
    output_ptr = output + rm * blockM * N + triton.arange(0, blockM)[:, None] * N + triton.arange(0, blockN)[None, :]
    triton.store(output_ptr, acc, mask=triton.arange(0, blockM)[:, None] < M - rm * blockM and triton.arange(0, blockN)[None, :] < N - rn * blockN)
```

**坏实践（反例）**：B 转置场景逐 case 更换 K0，行宽不对齐 512B
```python
BLOCK_K = 128                            # fp16 下行宽 128×2B=256B，非 512B 整数倍
for k in range(0, K, BLOCK_K):
    b = triton.load(y_ptr, ...)          # 转置 B 的行访问跨对齐边界
```
后果：多套 K0 适配增加分块开销，且行宽不对齐 512B 导致访存碎片，带宽利用率下降。✅ 正解：K0 统一固定为 256（满足 512B 整数倍），M0/N0 按实际矩阵尺寸灵活调整，优先保证 K0 对齐。

**已知实例**：matmul §raw (`experiences/matmul/exp_matmul_004.yaml`)
- 声称性能收益：10%
- 备注：适用于 A 不转置、B 转置的 matmul 场景，K0 固定为 256 满足 512B 整数倍要求，M0/N0 可根据实际矩阵尺寸灵活调整

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
- 相关主题: [切分优化：A、B 均不转置 matmul 的 512B 行宽对齐分块](exp_matmul_003.md) — 主题相关：共享 aic, matmul, tiling/sram 预算
- 相关主题: [切分优化：双转置矩阵乘的 512B 对齐分块](exp_matmul_005.md) — 主题相关：共享 aic, matmul, tiling/sram 预算

<!-- okf:related:end -->
