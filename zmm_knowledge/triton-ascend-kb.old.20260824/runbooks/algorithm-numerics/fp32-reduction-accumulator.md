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

- 实践案例: [MoeFinalizeRouting (MoE FFN output routing merge) on Ascend NPU](../../ops/moe-finalize-routing.md) — 同属优化知识，主题相关
- 实践案例: [MoeGatingTopKSoftmax (MoE gating softmax + topK) on Ascend NPU](../../ops/moe-gating-top-k-softmax.md) — 同属优化知识，主题相关
- 实践案例: [MoeInitRoutingQuantV2 (MoE routing init V2 + dynamic quant + drop) on Ascend NPU](../../ops/moe-init-routing-quant-v2.md) — 同属优化知识，主题相关
- 实践案例: [MoeInitRoutingQuant (MoE routing init + quantize) on Ascend NPU](../../ops/moe-init-routing-quant.md) — 同属优化知识，主题相关
- 实践案例: [MoeTokenPermuteWithEp (MoE token permute with Expert Parallel) on Ascend NPU](../../ops/moe-token-permute-with-ep.md) — 同属优化知识，主题相关
- 实践案例: [MoeTokenPermute (MoE token permute by indices) on Ascend NPU](../../ops/moe-token-permute.md) — 同属优化知识，主题相关
- 实践案例: [MoeTokenUnpermuteGrad (MoE token unpermute backward) on Ascend NPU](../../ops/moe-token-unpermute-grad.md) — 同属优化知识，主题相关
- 实践案例: [MoeTokenUnpermute (MoE token unpermute / gather+reduce) on Ascend NPU](../../ops/moe-token-unpermute.md) — 同属优化知识，主题相关
- 实践案例: [NsaCompressWithCache (NSA KV compression) on Ascend NPU](../../ops/nsa-compress-with-cache.md) — 主题相关：共享 reduction
- 实践案例: [RingAttentionUpdate (Ring Attention output rescale merge) on Ascend NPU](../../ops/ring-attention-update.md) — 同属优化知识，主题相关
- 开发指南: [归一化算子优化](../../reference/op-class-normalization.md) — 主题相关：共享 numerical_stability
- 开发指南: [归约算子优化](../../reference/op-class-reduction.md) — 主题相关：共享 numerical_stability, reduction
- 开发指南: [Softmax / LogSoftmax 算子优化](../../reference/op-class-softmax.md) — 主题相关：共享 numerical_stability, reduction
- 相关主题: [Misc 算子计算优化：向量化、微内核与阶段融合](../compute-pipeline/traj-misc-compute.md) — 主题相关：共享 numerical_stability
- 相关主题: [Reduction Compute 优化聚合](../compute-pipeline/traj-reduction-compute.md) — 主题相关：共享 reduction
- 相关主题: [Softmax 计算优化：Online Softmax、Two-Pass 向量化与数值稳定](../compute-pipeline/traj-softmax-compute.md) — 主题相关：共享 numerical_stability
- 相关主题: [Sum-and-Scale 归约内存访问优化](../memory-sram/traj-reduction-memory_access.md) — 主题相关：共享 reduction
- 相关主题: [Softmax Backward 内存访问优化聚合](../memory-sram/traj-softmax-memory_access.md) — 主题相关：共享 reduction
- 相关主题: [fp16 中间步骤溢出诊断--逐层定位首个溢出点](../meta-skills/fp16-intermediate-overflow-diagnosis.md) — 同属优化知识，主题相关
- 相关主题: [跨 tile 归约用 tl.atomic_add 作为可靠原语](atomic-add-cross-tile-reduction.md) — atomic_add 跨 tile 合并部分和时仍需 fp32 累加保证精度
- 相关主题: [反量化域必须按 x.dtype 分流（fp16 域 vs fp32 域）](dequant-domain-dtype-split.md) — 同属优化知识，主题相关
- 相关主题: [exp 稳定化 — softmax/rmsnorm 必须减最大值、fp32 累加](exp-stable-softmax-rmsnorm.md) — 主题相关：共享 numerical_stability
- 相关主题: [Online Softmax / Flash Attention 单遍稳定计算](online-softmax-flash-attention.md) — 主题相关：共享 numerical_stability
- 相关主题: [Attention 计算优化：Online Softmax、Split-K、Sparse Attention 与混合精度](traj-attention-compute.md) — 主题相关：共享 mixed_precision
- 相关主题: [数值精度优化：fp16/bf16 下的 fp32 计算与累加](traj-general_stability-numerical_precision.md) — 同属优化知识，主题相关
- 相关主题: [低精度数值稳定性：exp 最大偏移与 sqrt 非负保护](traj-general_stability-numerical_stability.md) — 主题相关：共享 numerical_stability

<!-- okf:related:end -->
