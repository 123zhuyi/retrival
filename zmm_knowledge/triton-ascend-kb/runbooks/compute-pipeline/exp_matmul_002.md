---
schema_version: okf.v1
kind: operator_optimization
type: optimization_runbook
source_family: runbooks
category: compute-pipeline
dimension: 2.1
title: '计算优化: fixpipe 随路量化 + K 维循环展开 + 预取'
description: 'matmul 场景下，计算优化: fixpipe 随路量化 + K 维循环展开 + 预取，收益 35%'
tags: [AICore 单元利用/Scalar 削减, aic, matmul]
confidence: verified
status: verified
created_at: 2026-08-21T00:00:00Z
updated_at: 2026-08-21T00:00:00Z
---

## 计算优化: fixpipe 随路量化 + K 维循环展开 + 预取

【优化维度】：AICore 单元利用/Scalar 削减
【泛化层级】：条件性（迁移条件：硬件 910b2, 910b3；失效边界：随路量化依赖 Cube fixpipe 硬件通路，非 Ascend 平台不可用；K 维 4 路展开要求 K ≥ 1024 级别，K 小时展开 tail 处理占比升高；预取深度受 UB 容量约束；仅适配 Triton 框架（备注自述））

### 原则
- **量化融合**：将fp16计算与Cube fixpipe随路量化融合，提升计算效率
- **循环展开**：展开K维循环，减少分支判断开销
- **内存预取**：使用triton.preload做数据预取，隐藏访存延迟

### 代码/骨架出处
- `Triton`:
```python
@triton.jit def matmul_optimized_kernel(x, y, output, M, N, K, **kwargs):
    # 融合预取+循环展开的优化代码
    pid = triton.program_id(0)
    rm = pid // (N // 256)
    rn = pid % (N // 256)
    x_ptr = x + rm * 256 * K + triton.arange(0, 256)[:, None] * K + triton.arange(0, 256)[None, :]
    y_ptr = y + rn * 256 + triton.arange(0, 256)[:, None] * N + triton.arange(0, 256)[None, :]
    acc = triton.zeros((256, 256), dtype=triton.float32)
    # 循环展开4次
    for k in range(0, K, 256*4):
        a1 = triton.load(x_ptr, mask=triton.arange(0, 256)[None, :] < K - k, other=0.0)
        b1 = triton.load(y_ptr, mask=triton.arange(0, 256)[:, None] < K - k, other=0.0)
        a2 = triton.load(x_ptr+256, mask=triton.arange(0, 256)[None, :] < K - k -256, other=0.0)
        b2 = triton.load(y_ptr+256*N, mask=triton.arange(0, 256)[:, None] < K - k -256, other=0.0)
        a3 = triton.load(x_ptr+512, mask=triton.arange(0, 256)[None, :] < K - k -512, other=0.0)
        b3 = triton.load(y_ptr+512*N, mask=triton.arange(0, 256)[:, None] < K - k -512, other=0.0)
        a4 = triton.load(x_ptr+768, mask=triton.arange(0, 256)[None, :] < K - k -768, other=0.0)
        b4 = triton.load(y_ptr+768*N, mask=triton.arange(0, 256)[:, None] < K - k -768, other=0.0)
        acc += a1@b1 + a2@b2 + a3@b3 + a4@b4
        x_ptr += 1024
        y_ptr += 1024 * N
    triton.store(output + rm * 256 * N + rn * 256 + triton.arange(0,256)[:,None]*N + triton.arange(0,256)[None,:], acc.to(x.dtype))
```

**坏实践（反例）**：K 维循环每次迭代只做一次 load+dot 且不做预取 → 每次迭代串行等待访存，分支判断与访存延迟无法隐藏 → ✅ 正解即本条：K 维 4 路展开（4 组独立 load 并行发射）+ `triton.preload` 预取 + fixpipe 随路量化输出。

**已知实例**：matmul §raw (`experiences/matmul/exp_matmul_002.yaml`)
- 声称性能收益：35%
- 备注：仅适配Triton框架，

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
- 开发指南: [昇腾 950 平台特性：CV 直通数据通路与 RegBase（含 Triton 可达性边界）](../portability/ascend-950-cv-datapath-regbase.md) — 同属优化知识，主题相关
- 相关主题: [访存优化：大尺寸 matmul 的 256 分块与 L0C 复用](../memory-sram/exp_matmul_001.md) — 主题相关：共享 aic, matmul
- 相关主题: [切分优化：A、B 均不转置 matmul 的 512B 行宽对齐分块](../memory-sram/exp_matmul_003.md) — 主题相关：共享 aic, matmul
- 相关主题: [切分优化：A 不转置、B 转置 matmul 的 K0=256 统一分块](../memory-sram/exp_matmul_004.md) — 主题相关：共享 aic, matmul
- 相关主题: [切分优化：双转置矩阵乘的 512B 对齐分块](../memory-sram/exp_matmul_005.md) — 主题相关：共享 aic, matmul
- 相关主题: [matmul core mapping 策略选择（列/行/对角）](core-mapping-strategy-selection.md) — 主题相关：共享 matmul
- 相关主题: [大矩阵 matmul 用对角 core mapping 消除 L2 bank conflict](diagonal-core-mapping-l2-bank-conflict.md) — 主题相关：共享 matmul
- 相关主题: [Scheduling Optimization: dynamic autotuning](exp_addmm_ascend_001.md) — 主题相关：共享 matmul
- 相关主题: [Compute Optimization: block ptr utilization](exp_block_matmul_inv_001.md) — 主题相关：共享 aicore 单元利用/scalar 削减, matmul
- 相关主题: [Scheduling Optimization: grouped scheduling](exp_bmm_grouped_001.md) — 主题相关：共享 matmul
- 相关主题: [Compute Optimization: tiling strategy](exp_cube1_qk_gemm_001.md) — 主题相关：共享 aicore 单元利用/scalar 削减, matmul
- 相关主题: [Compute Optimization: operator fusion](exp_fused_gemm_scale_001.md) — 主题相关：共享 aicore 单元利用/scalar 削减, matmul
- 相关主题: [Scheduling Optimization: async pipeline scheduling](exp_fused_matmul_norm_sigmoid_001.md) — 主题相关：共享 matmul
- 相关主题: [Compute Optimization: block tiling strategy](exp_matmul_bias_fused_001.md) — 主题相关：共享 aicore 单元利用/scalar 削减, matmul
- 相关主题: [Compute Optimization: small tile strategy](exp_matmul_triton_base_001.md) — 主题相关：共享 aicore 单元利用/scalar 削减, matmul
- 相关主题: [Compute Optimization: vector parallelism strategy](exp_matmul_vec_parallel_001.md) — 主题相关：共享 aicore 单元利用/scalar 削减, matmul
- 相关主题: [Compute Optimization: kernel fusion](exp_mhc_fused_gemm_sqrsum_001.md) — 主题相关：共享 aicore 单元利用/scalar 削减, matmul
- 相关主题: [Compute Optimization: grid stride scheduling](exp_partial_matmul_001.md) — 主题相关：共享 aicore 单元利用/scalar 削减, matmul
- 相关主题: [Compute Optimization: fused rope computation](exp_rope_matmul_fused_001.md) — 主题相关：共享 aicore 单元利用/scalar 削减, matmul

<!-- okf:related:end -->
