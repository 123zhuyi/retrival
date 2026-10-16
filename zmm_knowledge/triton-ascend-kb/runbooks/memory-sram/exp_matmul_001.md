---
schema_version: okf.v1
kind: operator_optimization
type: optimization_runbook
source_family: runbooks
category: memory-sram
dimension: 1.1
title: 访存优化：大尺寸 matmul 的 256 分块与 L0C 复用
description: matmul 场景下，访存优化：大尺寸 matmul 的 256 分块与 L0C 复用，收益 40%
tags: [访存效率, aic, matmul]
confidence: verified
status: verified
created_at: 2026-08-21T00:00:00Z
updated_at: 2026-08-21T00:00:00Z
---

## 访存优化：大尺寸 matmul 的 256 分块与 L0C 复用

【优化维度】：访存效率
【泛化层级】：条件性（迁移条件：硬件 910b2, 910b3，fp16/bf16，m/n ≥256、k ≥1024；失效边界：m/n/k <256 的小尺寸矩阵硬套 256 分块会导致 tile 大量 padding，应改用 128 分块；K 非 BLOCK_K 整数倍时尾部 mask 开销占比升高）

### 原则
- **分块策略**：L0级Tile采用128x128x128块划分，适配910b2/910b3 Cube L0层访存粒度
- **数据排布**：输入张量nd保持不变，减少重排开销
- **寄存器复用**：中间结果存入L0C，提升数据复用

### 代码/骨架出处
- `Triton`:
```python
@triton.jit def matmul_kernel(x, y, output, M, N, K, blockM=256, blockN=256, blockK=256, **kwargs):
    # 256分块基础代码
    pid = triton.program_id(0)
    rm = pid // (N // blockN)
    rn = pid % (N // blockN)
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

**坏实践（反例）**：小尺寸矩阵硬套 256 分块
```python
BLOCK_M = BLOCK_N = BLOCK_K = 256        # m/n/k 仅百余的小矩阵仍用 256 分块
acc = triton.zeros((256, 256), dtype=triton.float32)  # 累加器大面积为 padding
```
后果：tile 内大量无效 padding，Cube 单元利用率与有效带宽双双下降。✅ 正解：m/n/k ≥256 的大矩阵用 256 分块，L0 级内层 tile 按 128×128×128 划分以适配 Cube L0 访存粒度，中间结果驻留 L0C 提升复用；小尺寸矩阵退回 128 分块。

**已知实例**：matmul §raw (`experiences/matmul/exp_matmul_001.yaml`)
- 声称性能收益：40%
- 备注：适用于 m/n/k≥256 的大尺寸矩阵，小尺寸矩阵建议使用 128 分块

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
- 相关主题: [反量化域必须按 x.dtype 分流（fp16 域 vs fp32 域）](../algorithm-numerics/dequant-domain-dtype-split.md) — 同属优化知识，主题相关
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
- 相关主题: [计算优化: fixpipe 随路量化 + K 维循环展开 + 预取](../compute-pipeline/exp_matmul_002.md) — 主题相关：共享 aic, matmul
- 相关主题: [Memory Access Optimization: fused add-RMSNorm with weight hoisting](exp_gemma_rmsnorm_001.md) — 主题相关：共享 matmul, 访存效率
- 相关主题: [切分优化：A、B 均不转置 matmul 的 512B 行宽对齐分块](exp_matmul_003.md) — 主题相关：共享 aic, matmul
- 相关主题: [切分优化：A 不转置、B 转置 matmul 的 K0=256 统一分块](exp_matmul_004.md) — 主题相关：共享 aic, matmul
- 相关主题: [切分优化：双转置矩阵乘的 512B 对齐分块](exp_matmul_005.md) — 主题相关：共享 aic, matmul

<!-- okf:related:end -->
