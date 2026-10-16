---
schema_version: okf.v1
kind: operator_optimization
type: optimization_runbook
source_family: runbooks
category: memory-sram
dimension: "1.4"
title: Cube-Vector 场景行宽必须 512B 对齐
description: matmul 与 Cube-Vector 向量化 load 的块行宽必须为 512B 整数倍，否则 partial cache-line read 导致 MTE2 带宽骤降。
tags: [alignment, 512b, row_width, cube_vector, matmul, block_n]
confidence: verified
status: verified
created_at: '2026-08-10T00:00:00Z'
updated_at: '2026-08-10T00:00:00Z'
---

## CT-10 Cube-Vector 行宽 512B 对齐

【优化维度】：内存｜硬件对齐
【泛化层级】：Triton通用（迁移条件：Cube-Vector 场景，含 matmul 与需要 Cube 单元参与的向量化 load；失效边界：纯 Vector-Vector 元素级操作适用 32B 对齐，见对应 CT）

<原则：Ascend Cube 单元要求参与计算的矩阵/张量行宽为 512B 整数倍。违反时硬件触发 partial cache-line read，有效内存带宽显著低于 ~1.8 TB/s 峰值，在 profiling 中表现为 `aiv_mte2_time` 异常升高；同时 partial cache-line 使一次 load 拆成两次 L2 读取，bank conflict 概率翻倍。>

<约束：matmul 的 `BLOCK_N * dtype_bytes` 必须为 512 的整数倍；所有参与 Cube-Vector 计算的张量行宽同理。>
<违反后果：非对齐行宽导致 MTE2 partial cache-line read，有效带宽下降；大矩阵场景下常与 L2 bank conflict 叠加。>
<满足方式：按 dtype 选择最小对齐元素数；必要时对尾部 padding；autotune 候选必须满足。>

| dtype | 元素大小 | 最小对齐元素数 | 推荐 matmul BLOCK_N | 推荐 BLOCK_K |
|---|---|---|---|---|
| fp8 | 1B | 512 | 256 | 256 |
| fp16 / bf16 | 2B | 256 | 128 或 256 | 256 |
| fp32 | 4B | 128 | 128 | 128 或 256 |

```python
# matmul 标准搜索空间（已满足 512B 行宽对齐）
configs = [
    triton.Config({'BLOCK_M': bm, 'BLOCK_N': bn, 'BLOCK_K': 256, 'GROUP_SIZE': gs})
    for bm in [128, 256]
    for bn in [128, 256]    # 均为 256 的整数倍（fp16/bf16）
    for gs in [8, 16, 32]
]
```

**坏实践（反例）**：
```python
BLOCK_N = 100  # fp16: 200 bytes，非 512B 整数倍
```
后果：Cube-Vector load 退化为 partial cache-line，MTE2 带宽利用率低，kernel 变成 memory-bound；与 bank conflict 叠加时 `aiv_mte2_time` 可达 theoretical 的 2–5 倍。✅ 正解即 `BLOCK_N` 取 256/128/512 的整数倍。

## 已知实例（P3 挂载）

### Experience YAML 反链
- `exp_matmul_003` — 切分优化 （收益 10%）
  - 适用于A、B矩阵均不转置的matmul场景，分块行宽严格对齐Ascend后端512B整数倍要求，最大化带宽利用
- `exp_matmul_005` — 切分优化 （收益 32%）
  - 适用于A、B矩阵均转置的matmul场景，M0=256、K0=256满足512B整数倍，N0=128适配转置后的数据排布

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
- 相关主题: [Compute Optimization: vectorization alignment](../compute-pipeline/exp_backward_sum_001.md) — 同属优化知识，主题相关
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
- 相关主题: [Compute Optimization: memory alignment](../compute-pipeline/exp_moe_wgrad_npu_001.md) — 同属优化知识，主题相关
- 相关主题: [Triton-Ascend kernel 调试排查清单](../meta-skills/debugging-checklist.md) — alignment 检查项的具体契约
- 相关主题: [Memory Access Optimization: sorted bin grouping with padded alignment](exp_scatter_moe_padded_001.md) — 同属优化知识，主题相关
- 相关主题: [并行轴选择必须与连续访问对齐，而非与最大 numel 对齐](parallel-axis-align.md) — 同属优化知识，主题相关
- 相关主题: [pergroup BLOCK_K 必须整除 group_size，选 min(group_size, 128)](pergroup-blockk-align.md) — 同属优化知识，主题相关

<!-- okf:related:end -->
