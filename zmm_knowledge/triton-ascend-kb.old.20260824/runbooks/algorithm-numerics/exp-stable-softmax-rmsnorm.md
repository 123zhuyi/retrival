---
schema_version: okf.v1
kind: operator_optimization
type: optimization_runbook
source_family: runbooks
category: algorithm-numerics
dimension: "3.2"
title: exp 稳定化 — softmax/rmsnorm 必须减最大值、fp32 累加
description: fp16 下 exp 直接求和会溢出为 Inf，softmax/rmsnorm 等算子必须减最大值后用 fp32 累加，最终 cast 回 fp16/bf16。
tags: [softmax, rmsnorm, exp_overflow, fp32_accumulate, numerical_stability]
confidence: verified
status: verified
created_at: "2026-08-10T00:00:00Z"
updated_at: "2026-08-10T00:00:00Z"
---

## CT-12 exp 稳定化 — softmax/rmsnorm 必须减最大值、fp32 累加

【优化维度】：数值｜数值稳定
【泛化层级】：Triton通用（迁移条件：所有含 exp、sum、mean、var、rsqrt 的算子；失效边界：输入已被证明绝对值极小且不会溢出时可放宽，但仍建议统一处理）

<原则：fp16 的 `exp(x)` 在 x > 65504 时溢出为 Inf。softmax、attention score、rmsnorm 等算子若直接 `tl.exp(x)` 再求和，结果会 NaN/Inf。正确做法是先把输入减去最大值（数学等价，结果不变），并在 fp32 中完成累加/归约，最后 cast 回 fp16/bf16。>

**softmax 稳定化模板**：

```python
# 错误：直接 exp
p = tl.exp(x)
s = tl.sum(p)

# 正确：减最大值 + fp32 累加
x_fp32 = x.to(tl.float32)
m = tl.max(x_fp32, axis=0)
p = tl.exp(x_fp32 - m)          # 最大值为 0，不会溢出
s = tl.sum(p)                   # fp32 累加
out = (p / s).to(tl.float16)    # 最终 cast 回低精度
```

**rmsnorm 稳定化模板**：

```python
# 错误：fp16 累加 + 1/sqrt
acc = tl.zeros([BLOCK], dtype=tl.float16)
acc += x * x
inv_rms = 1.0 / tl.sqrt(mean_sq + eps)

# 正确：fp32 累加 + rsqrt
x_fp32 = x.to(tl.float32)
mean_sq = tl.sum(x_fp32 * x_fp32) / N
inv_rms = tl.rsqrt(mean_sq + eps)   # 比 1/sqrt 精度更好
out = (x_fp32 * inv_rms * weight).to(tl.float16)
```

**数值边界**：

| 操作 | fp16 危险区 | 安全做法 |
|---|---|---|
| `tl.exp(x)` | x > 11.09（ln(65504)）即可能溢出 | x - max(x) 后再 exp |
| `tl.sum(fp16)` | 累加多个 fp16 会精度损失/溢出 | fp32 累加器 |
| `1/tl.sqrt(x)` | 小 x 下精度差 | `tl.rsqrt(x)` |
| `x / denom` | denom 可能为 0 | `tl.maximum(denom, eps)` |

**坏实践（反例）**：
```python
p = tl.exp(x)
s = tl.sum(p)
out = p / s
```
后果：fp16 下大值 exp 溢出为 Inf，softmax 输出 NaN，训练直接崩溃。✅ 正解即减最大值 + fp32 累加。

**诊断清单**：
1. 模型训练出现 NaN/Inf，优先检查所有 softmax/rmsnorm/attention 算子。
2. 确认 kernel 内是否有 `tl.exp` 直接作用于原始输入。
3. 确认归约/累加是否使用 fp32 accumulator。
4. 用 fp32 参考实现对比，定位首个误差超阈值的位置。
5. 修复后做 full-sequence 精度回归。

## 已知实例（P3 挂载）

### Experience YAML 反链
- `exp_stability_001` — 数值稳定性优化 （收益 数值稳定性提升100%）
  - 适用于所有包含指数运算的场景（如Softmax、注意力分数计算），fp16/bf16下未做该处理易出现exp(>65504)返回Inf，910b2/910b3平台需强制启用该优化
- `exp_stability_002` — 数值稳定性优化 （收益 数值稳定性提升95%）
  - 适用于Attention的sqrt(d_k)、均方差计算、梯度平方根等场景；910b2/910b3平台的硬件sqrt指令对负数输入直接返回NaN，必须前置该防护逻辑
- `exp_stability_003` — 数值精度优化 （收益 计算精度提升（MAE降低80%））
  - 适用于多层激活、累积求和等复杂计算场景；fp16/bf16的动态范围有限，中间计算易丢失小数位，910b2/910b3平台建议所有核心计算步骤启用float32转换
- `exp_stability_004` — 数值精度优化 （收益 累加精度提升（MSE降低90%））
  - 适用于大步数累加、梯度累积、损失求和等场景；fp16累加器在1000次累加后误差可达1e-3，float32累加器误差仅1e-7，910b2/910b3平台建议所有累加操作启用高精度累加器

<!-- okf:related:start -->

# 相关

- 实践案例: [FlashAttentionScoreGrad training-backward on Ascend NPU](../../ops/flash-attention-score-grad.md) — 主题相关：共享 softmax
- 实践案例: [FlashAttentionScore training-forward on Ascend NPU](../../ops/flash-attention-score.md) — 主题相关：共享 softmax
- 实践案例: [FusedFloydAttention (Floyd multi-dim attention with dual KV groups) on Ascend NPU](../../ops/fused-floyd-attention.md) — 主题相关：共享 softmax
- 实践案例: [MlaPreprocess (MLA attention preprocessing fusion) on Ascend NPU](../../ops/mla-preprocess.md) — 主题相关：共享 rmsnorm
- 实践案例: [MlaPrologV3 (MLA pre-process) on Ascend NPU](../../ops/mla-prolog-v3.md) — 主题相关：共享 rmsnorm
- 实践案例: [MlaProlog (MLA attention preprocessing — prolog V1) on Ascend NPU](../../ops/mla-prolog.md) — 主题相关：共享 rmsnorm
- 实践案例: [MoeGatingTopKSoftmaxV2 (MoE gating topk + softmax) on Ascend NPU](../../ops/moe-gating-top-k-softmax-v2.md) — 主题相关：共享 softmax
- 实践案例: [MoeGatingTopKSoftmax (MoE gating softmax + topK) on Ascend NPU](../../ops/moe-gating-top-k-softmax.md) — 主题相关：共享 softmax
- 实践案例: [PromptFlashAttention inference/prefill on Ascend NPU](../../ops/prompt-flash-attention.md) — 主题相关：共享 softmax
- 实践案例: [RainFusionAttention (CATLASS block-sparse inference attention) on Ascend NPU](../../ops/rain-fusion-attention.md) — 主题相关：共享 softmax
- 实践案例: [RMSNorm-RoPE fused kernel on Ascend 910B1](../../ops/rmsnorm-rope.md) — 主题相关：共享 rmsnorm
- 开发指南: [Fused Attention / Flash Attention 高性能算子](../../reference/op-class-attention.md) — 主题相关：共享 softmax
- 开发指南: [Elementwise 算子优化](../../reference/op-class-elementwise.md) — 同属优化知识，主题相关
- 开发指南: [融合算子综合页](../../reference/op-class-fused_ops.md) — 主题相关：共享 rmsnorm
- 开发指南: [归一化算子优化](../../reference/op-class-normalization.md) — 主题相关：共享 numerical_stability, rmsnorm, softmax
- 开发指南: [归约算子优化](../../reference/op-class-reduction.md) — 主题相关：共享 numerical_stability
- 开发指南: [Softmax / LogSoftmax 算子优化](../../reference/op-class-softmax.md) — 主题相关：共享 numerical_stability, softmax
- 相关主题: [Rmsnorm Fused Skeleton](../../reference/skeleton-rmsnorm_fused.md) — 同属优化知识，主题相关
- 相关主题: [exp_matmul_vec_parallel_001 — matmul 向量并行 + activation fusion](../compute-pipeline/instances/traj-exp-matmul-vec-parallel-001.md) — 主题相关：共享 fp32_accumulate
- 相关主题: [算子融合模式与约束决策](../compute-pipeline/operator-fusion-patterns.md) — 主题相关：共享 rmsnorm
- 相关主题: [Misc 算子计算优化：向量化、微内核与阶段融合](../compute-pipeline/traj-misc-compute.md) — 主题相关：共享 numerical_stability
- 相关主题: [Normalization Scheduling 优化聚合](../compute-pipeline/traj-normalization-scheduling.md) — 主题相关：共享 rmsnorm
- 相关主题: [Reduction Compute 优化聚合](../compute-pipeline/traj-reduction-compute.md) — 主题相关：共享 rmsnorm
- 相关主题: [Softmax 计算优化：Online Softmax、Two-Pass 向量化与数值稳定](../compute-pipeline/traj-softmax-compute.md) — 主题相关：共享 numerical_stability, softmax
- 相关主题: [Matmul 内存访问优化：分块策略、L0 复用与算子融合](../memory-sram/traj-matmul-memory_access.md) — 主题相关：共享 rmsnorm
- 相关主题: [attention 单 kernel 融合失败诊断--softmax 需 workspace，大 D UB 放不下](../meta-skills/attention-fusion-failure-diagnosis.md) — 同属优化知识，主题相关
- 相关主题: [Attention 工作流优化：中小序列标准 Attention 的三步计算、数值稳定与维度对齐](../meta-skills/traj-attention-workflow_opt.md) — 主题相关：共享 softmax
- 相关主题: [统计归约必须用 fp32 累加器，禁止 fp16/bf16 直接累加](fp32-reduction-accumulator.md) — 主题相关：共享 numerical_stability
- 相关主题: [Online Softmax / Flash Attention 单遍稳定计算](online-softmax-flash-attention.md) — 主题相关：共享 numerical_stability
- 相关主题: [大序列 Attention 的分块计算与 Online Softmax 数值稳定](traj-attention-memory_numerical.md) — 同属优化知识，主题相关
- 相关主题: [低精度数值稳定性：exp 最大偏移与 sqrt 非负保护](traj-general_stability-numerical_stability.md) — 主题相关：共享 numerical_stability, softmax
- 相关主题: [Normalization Compute 优化聚合](traj-normalization-compute.md) — 主题相关：共享 rmsnorm
- 相关主题: [Normalization 内存访问优化：Tiling、Two-Pass 与 Kernel Fusion](traj-normalization-memory_access.md) — 主题相关：共享 rmsnorm

<!-- okf:related:end -->
