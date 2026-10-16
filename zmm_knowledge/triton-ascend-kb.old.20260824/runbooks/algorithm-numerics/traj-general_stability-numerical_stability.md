---
schema_version: okf.v1
kind: example
type: code_example
source_family: trajectory
category: general_stability
dimension: "3.3"
title: 低精度数值稳定性：exp 最大偏移与 sqrt 非负保护
description: 针对 Ascend 910B 系列上 fp16/bf16 低精度场景的数值稳定性优化，核心是在指数运算前减去最大值防止 Inf，以及在平方根前用 eps 截断负数防止 NaN。
tags: [general_stability, numerical_stability, fp16, bf16, exp, sqrt, softmax, 910b2, 910b3]
confidence: verified
status: active
resource: https://gitcode.com/zhou_huaman/cannbot-skills_5171/blob/d972095e7b5842b59302088cdaa07749eef5b901/ops/triton-knowledge-retrieval/knowledge/raw/experiences/general_stability/
created_at: "2026-08-11T00:00:00Z"
updated_at: "2026-08-11T00:00:00Z"
---

# 低精度数值稳定性：exp 最大偏移与 sqrt 非负保护

## 概述

在 Ascend 910B2/910B3 上，fp16/bf16 等低精度格式虽然能显著提升 Cube/Vector 吞吐，但动态范围有限，容易在指数、平方根、累加等运算中产生 **Inf/NaN**。本组 2 条 experience 覆盖了两种最常见的数值稳定性问题：

1. **指数运算上溢**：`exp(x)` 在 fp16 下当 `x > 65504` 的 ln 阈值时会返回 `Inf`；
2. **平方根负数**：`sqrt(x)` 对微小负数（如 `-1e-10`）直接返回 `NaN`。

这些问题在 Attention（`exp(QK^T/sqrt(d_k))`）、Softmax、RMSNorm/LayerNorm（均方差开方）、以及梯度平方根等场景中反复出现。**数值稳定性优化通常没有传统意义上的“性能提升百分比”，但它决定了低精度 kernel 能否正确运行。** 本卡中的“收益”指的是不稳定 case 的修复率或 NaN/Inf 消除率。

**核心原则**：

| 问题 | 根因 | 通用解法 | 精度损失 |
|---|---|---|---|
| `exp(x)` 上溢 | fp16/bf16 指数动态范围小 | 先算 `max_val = max(x)`，再 `exp(x - max_val)` | 无，因为 `exp(x-max)/sum(exp(x-max))` 与原始 softmax 等价 |
| `sqrt(x)` NaN | 浮点误差导致理论非负数出现微小负值 | `sqrt(max(x, eps))`，fp16/bf16 用 `eps=1e-6`，fp32 用 `eps=1e-8` | 极小，eps 远小于有效精度 |

**适用条件**：
- 数据类型为 fp16/bf16（fp32 相对安全但仍建议保留保护逻辑）；
- 包含 softmax、attention score、rmsnorm variance、sqrt(d_k)、梯度平方根等运算；
- 训练场景尤其需要，因为一次 NaN 会导致梯度传播失败。

## 共同模式

### 1. 指数运算前减去最大值（max-shift）

Softmax 和 attention score 的标准形式是 `exp(x_i) / sum(exp(x_j))`。数学上，分子分母同时乘以 `exp(-max)` 不改变结果：

```
softmax(x_i) = exp(x_i - max) / sum(exp(x_j - max))
```

**根因**：
- fp16 的最大可表示值约为 65504，`exp(11.09)` 就会溢出；
- 在 attention 中 `QK^T / sqrt(d_k)` 的数值可能达到几十甚至上百；
- 直接 `exp(x)` 会得到 `Inf`，再做除法得到 `Inf/Inf = NaN`。

**实现要点**：
- 先对目标维度求 `max_val = tl.max(x, axis=dim)`；
- 所有元素减去 `max_val` 后再 `tl.exp`；
- 求和、归一化在 FP32 中完成，最后 downcast。

**收益与代价**：
- **收益**：可 100% 消除 fp16/bf16 下指数上溢导致的 Inf；
- **代价**：仅增加一次 `tl.max`，在 SIMD 硬件上耗时可忽略；
- **精度**：数学等价，无精度损失。

**常见坑**：
- `max_val` 必须与被减张量在维度上广播一致；
- 对 full NaN 输入，`tl.max` 可能返回 NaN，需额外处理；
- Online Softmax 把 max-shift 融入迭代过程，避免了两趟扫描。

### 2. 平方根前用 eps 截断负数

`sqrt(x)` 的数学定义域是 `x ≥ 0`，但由于前序运算的浮点舍入，`x` 可能出现 `-1e-10` 这样的微小负数。Ascend 910B 的硬件 `sqrt` 对负数直接返回 `NaN`。

**根因**：
- RMSNorm/LayerNorm 中 variance = `sum(x^2)/N`，理论上非负，但 fp16/bf16 累加可能产生微小负值；
- `sqrt(d_k)` 中若 `d_k` 为 0 或配置错误会得到 0 或负数；
- 梯度平方根（如 Adam 等优化器）中梯度可能含噪声。

**实现要点**：
- 低精度场景用 `eps=1e-6`：`safe_x = tl.maximum(x, eps)`；
- fp32 场景用 `eps=1e-8`；
- 如果只需要避免 NaN 不在乎梯度，可用 `tl.maximum(x, 0.0)`；
- 在方差计算中，建议先 `tl.maximum(variance, eps)` 再 `tl.sqrt`。

**收益与代价**：
- **收益**：可消除 95% 以上因微小负数导致的 NaN；
- **代价**：一个 `tl.maximum` 指令；
- **精度**：eps 远小于 fp16 的有效精度，对最终结果影响可忽略。

### 3. 何时需要强制启用

在 Ascend 910B 上，以下场景应**无条件**启用上述保护：
- **Softmax / LogSoftmax**：必须 max-shift；
- **Attention score**：`exp(QK^T / sqrt(d_k))` 必须 max-shift；
- **RMSNorm / LayerNorm variance**：`sqrt(variance + eps)`，eps 不可省略；
- **sqrt(d_k)**：用 `sqrt(max(d_k, eps))`；
- **任何用户自定义的指数或平方根核**：尤其是输入范围不可控时。

## 收益统计

| experience | 收益 | 简介 |
|---|---|---|
| `exp_stability_001` | 数值稳定性提升 100% | 指数运算前减去维度内最大值，消除 fp16/bf16 下 `exp(x)` 上溢为 Inf 的问题。 |
| `exp_stability_002` | 数值稳定性提升 95% | 平方根前用 `max(x, eps)` 截断负数，避免 910B 硬件 `sqrt` 对负数返回 NaN。 |

## 代表 Kernel

本组 experience 的原始 YAML 未附带独立 kernel 文件，以下用 `exp_stability_002` 的 sqrt 保护逻辑作为代表示例，展示核心代码结构：

```python
@triton.jit
def sqrt_negative_opt_kernel(
    input_ptr,
    output_ptr,
    n_elements,
    eps: tl.constexpr = 1e-6,
    BLOCK_SIZE: tl.constexpr = 128,
):
    pid = triton.program_id(0)
    offsets = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    mask = offsets < n_elements

    # 加载数据并提升精度
    data = tl.load(input_ptr + offsets, mask=mask).to(tl.float32)

    # 核心优化：确保被开方数非负
    safe_data = tl.maximum(data, eps)

    # 平方根计算（无 NaN 风险）
    sqrt_data = tl.sqrt(safe_data)

    # 存储结果
    tl.store(output_ptr + offsets, sqrt_data, mask=mask)
```

对应 `exp_stability_001` 的 max-shift 模式：

```python
# Softmax / attention score 稳定化示例
max_val = tl.max(scores, axis=1)              # 沿 softmax 维度取最大
stable_scores = scores - max_val[:, None]     # 最大偏移
exp_scores = tl.exp(stable_scores)            # 此时最大值为 0，不会上溢
sum_exp = tl.sum(exp_scores, axis=1)
softmax_out = exp_scores / sum_exp[:, None]
```

## 已知实例

- `exp_stability_002` — Numerical Stability （数值稳定性提升 95%）
  - 适用于 Attention 的 `sqrt(d_k)`、均方差计算、梯度平方根等场景；910B 硬件 `sqrt` 对负数输入返回 NaN，必须前置 `max(x, eps)` 保护。
- `exp_stability_001` — Numerical Stability （数值稳定性提升 100%）
  - 适用于所有包含指数运算的场景（Softmax、attention score 计算）；fp16/bf16 下未做 max-shift 易出现 `exp(>65504)` 返回 Inf，910B 平台需强制启用。

## 关联知识

- 开发指南: [Normalization 算子优化](../../reference/op-class-normalization.md)

<!-- okf:related:start -->

# 相关

- 对应指南: [Fused Attention / Flash Attention 高性能算子](../../reference/op-class-attention.md) — 主题相关：共享 910b2, 910b3, softmax
- 对应指南: [Elementwise 算子优化](../../reference/op-class-elementwise.md) — 主题相关：共享 910b2, 910b3, exp
- 对应指南: [归一化算子优化](../../reference/op-class-normalization.md) — 主题相关：共享 910b2, 910b3, numerical_stability, softmax
- 对应指南: [归约算子优化](../../reference/op-class-reduction.md) — 主题相关：共享 910b2, 910b3, numerical_stability
- 对应指南: [Softmax / LogSoftmax 算子优化](../../reference/op-class-softmax.md) — 主题相关：共享 910b2, 910b3, numerical_stability, softmax
- 相关主题: [Elementwise absolute value (Abs)](../../ops/elementwise-abs.md) — 主题相关：共享 bf16
- 相关主题: [Elementwise add with a tail mask](../../ops/elementwise-add.md) — 主题相关：共享 bf16
- 相关主题: [FlashAttentionScoreGrad training-backward on Ascend NPU](../../ops/flash-attention-score-grad.md) — 主题相关：共享 softmax
- 相关主题: [MoeFinalizeRouting (MoE FFN output routing merge) on Ascend NPU](../../ops/moe-finalize-routing.md) — 主题相关：共享 bf16, fp16
- 相关主题: [MoeGatingTopKSoftmax (MoE gating softmax + topK) on Ascend NPU](../../ops/moe-gating-top-k-softmax.md) — 主题相关：共享 bf16, fp16, softmax
- 相关主题: [MoeInitRoutingQuantV2 (MoE routing init V2 + dynamic quant + drop) on Ascend NPU](../../ops/moe-init-routing-quant-v2.md) — 主题相关：共享 bf16, fp16
- 相关主题: [MoeInitRoutingQuant (MoE routing init + quantize) on Ascend NPU](../../ops/moe-init-routing-quant.md) — 主题相关：共享 bf16, fp16
- 相关主题: [MoeTokenPermuteWithEp (MoE token permute with Expert Parallel) on Ascend NPU](../../ops/moe-token-permute-with-ep.md) — 主题相关：共享 bf16, fp16
- 相关主题: [MoeTokenPermute (MoE token permute by indices) on Ascend NPU](../../ops/moe-token-permute.md) — 主题相关：共享 bf16, fp16
- 相关主题: [MoeTokenUnpermuteGrad (MoE token unpermute backward) on Ascend NPU](../../ops/moe-token-unpermute-grad.md) — 主题相关：共享 bf16, fp16
- 相关主题: [MoeTokenUnpermute (MoE token unpermute / gather+reduce) on Ascend NPU](../../ops/moe-token-unpermute.md) — 主题相关：共享 bf16, fp16
- 相关主题: [NsaCompressWithCache (NSA KV compression) on Ascend NPU](../../ops/nsa-compress-with-cache.md) — 主题相关：共享 bf16, fp16
- 相关主题: [RecurrentGatedDeltaRule (linear attention / SSM-style gated delta rule) on Ascend NPU](../../ops/recurrent-gated-delta-rule.md) — 主题相关：共享 bf16
- 相关主题: [RingAttentionUpdate (Ring Attention output rescale merge) on Ascend NPU](../../ops/ring-attention-update.md) — 主题相关：共享 bf16, fp16, softmax
- 相关主题: [RMSNorm-RoPE fused kernel on Ascend 910B1](../../ops/rmsnorm-rope.md) — 主题相关：共享 bf16
- 相关主题: [exp_matmul_vec_parallel_001 — matmul 向量并行 + activation fusion](../compute-pipeline/instances/traj-exp-matmul-vec-parallel-001.md) — 主题相关：共享 910b2, 910b3
- 相关主题: [Misc 算子计算优化：向量化、微内核与阶段融合](../compute-pipeline/traj-misc-compute.md) — 主题相关：共享 910b2, 910b3, numerical_stability
- 相关主题: [Softmax 计算优化：Online Softmax、Two-Pass 向量化与数值稳定](../compute-pipeline/traj-softmax-compute.md) — 主题相关：共享 910b2, 910b3, numerical_stability, softmax
- 相关主题: [bf16 融合反量化写单表达式解锁 BLOCK_K=128](../memory-sram/bf16-fused-dequant-blockk128.md) — 主题相关：共享 bf16
- 相关主题: [attention 单 kernel 融合失败诊断--softmax 需 workspace，大 D UB 放不下](../meta-skills/attention-fusion-failure-diagnosis.md) — 同属优化知识，主题相关
- 相关主题: [fp16 中间步骤溢出诊断--逐层定位首个溢出点](../meta-skills/fp16-intermediate-overflow-diagnosis.md) — 同属优化知识，主题相关
- 相关主题: [Attention 工作流优化：中小序列标准 Attention 的三步计算、数值稳定与维度对齐](../meta-skills/traj-attention-workflow_opt.md) — 主题相关：共享 910b2, 910b3, softmax
- 相关主题: [反量化域必须按 x.dtype 分流（fp16 域 vs fp32 域）](dequant-domain-dtype-split.md) — 同属优化知识，主题相关
- 相关主题: [exp 稳定化 — softmax/rmsnorm 必须减最大值、fp32 累加](exp-stable-softmax-rmsnorm.md) — 主题相关：共享 numerical_stability, softmax
- 相关主题: [统计归约必须用 fp32 累加器，禁止 fp16/bf16 直接累加](fp32-reduction-accumulator.md) — 主题相关：共享 numerical_stability
- 相关主题: [Online Softmax / Flash Attention 单遍稳定计算](online-softmax-flash-attention.md) — 主题相关：共享 numerical_stability
- 相关主题: [大序列 Attention 的分块计算与 Online Softmax 数值稳定](traj-attention-memory_numerical.md) — 主题相关：共享 910b2, 910b3
- 相关主题: [数值精度优化：fp16/bf16 下的 fp32 计算与累加](traj-general_stability-numerical_precision.md) — 主题相关：共享 910b2, 910b3, bf16, fp16, general_stability

<!-- okf:related:end -->
