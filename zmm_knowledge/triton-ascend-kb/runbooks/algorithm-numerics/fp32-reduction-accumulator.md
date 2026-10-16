---
schema_version: okf.v1
kind: operator_optimization
type: optimization_runbook
source_family: runbooks
category: algorithm-numerics
dimension: "3.2"
title: 统计归约必须用 fp32 累加器，禁止 fp16/bf16 直接累加
description: 归一化/归约统计量累加前必须将输入升精度到 tl.float32，禁止用原始 dtype 直接累加
tags: [fp32_accumulator, reduction, numerical_stability, mixed_precision]
confidence: verified
status: verified
created_at: "2026-08-05T00:00:00Z"
updated_at: "2026-08-05T00:00:00Z"
---

## OPT-1 统计归约 fp32 累加器

【优化维度】：数值｜混合精度
【泛化层级】：Triton通用（迁移条件：归约统计量含小量相乘/求和/除法；失效边界：纯整数归约或对精度不敏感的场景可放宽）

<原则：归一化统计量（mean/var/rstd/inv_std^3）涉及小量相乘、求和、除法，fp16/bf16 累加会在小 spatial size 或数值接近 0 时产生显著精度误差。var 公式 (sq-sum*sum/N)/N 对累加顺序敏感；inv_std^3、1/S 等小量乘法容易下溢。升 fp32 累加可消除这些误差。关联硬件依据：Vector 单元 fp32 累加无额外开销（寄存器位宽足够）。>

<骨架（占位名伪码）：
sum_acc = tl.full((), 0.0, tl.float32)
sq_acc = tl.full((), 0.0, tl.float32)
x = tl.load(input_ptr + offsets, mask=mask, other=0.0).to(tl.float32)
sum_acc += tl.sum(x, axis=0)
sq_acc += tl.sum(x * x, axis=0)
mean = sum_acc / elemCount
var = (sq_acc - sum_acc * sum_acc / elemCount) / elemCount
>

**坏实践（反例）**：用原始 dtype 直接累加
```python
x = tl.load(input_ptr + offsets, mask=mask, other=0.0)   # fp16/bf16
sum_acc += tl.sum(x, axis=0)                              # 低精度累加
sq_acc += tl.sum(x * x, axis=0)
```
后果：var 公式精度损失，rstd 误差导致 verify 失败（小 spatial size 或数值接近 0 时尤为严重）。✅ 正解即本条 fp32 累加器。

**已知实例**：GroupNormSwish §1 N1、AdaptiveInstanceNorm2DBackward §1 N1（`triton-op-generator/template/normalization.md`）

<!-- okf:related:start -->

# 相关

- 开发指南: [归一化算子优化](../../reference/op-class-normalization.md) — 主题相关：共享 numerical_stability
- 开发指南: [归约算子优化](../../reference/op-class-reduction.md) — 主题相关：共享 numerical_stability, reduction
- 开发指南: [Softmax / LogSoftmax 算子优化](../../reference/op-class-softmax.md) — 主题相关：共享 numerical_stability, reduction
- 相关主题: [exp 稳定化 — softmax/rmsnorm 必须减最大值、fp32 累加](exp-stable-softmax-rmsnorm.md) — 主题相关：共享 numerical_stability
- 相关主题: [Online Softmax / Flash Attention 单遍稳定计算](online-softmax-flash-attention.md) — 主题相关：共享 numerical_stability

<!-- okf:related:end -->
