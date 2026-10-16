---
schema_version: okf.v1
kind: operator_optimization
type: optimization_runbook
source_family: runbooks
category: memory-sram
dimension: "1.2"
title: 非 2 的幂的语义维度必须 pad 到 next_pow2 + 极值填充 + 截断
description: bitonic sort/butterfly FFT 要求长度 2 的幂时，host 侧 pad 到 next_pow2，pad 值取不影响顺序的极值，处理完截断
tags: [pad_pow2, extremum_fill, bitonic, fft, truncate]
confidence: verified
status: verified
created_at: "2026-08-05T00:00:00Z"
updated_at: "2026-08-05T00:00:00Z"
---

## OPT-4 pad 到 next_pow2 + 极值填充

【优化维度】：内存｜Tiling·SRAM 预算
【泛化层级】：条件性（迁移条件：算法（bitonic sort/butterfly FFT）要求长度 2 的幂；失效边界：算法不要求 2 的幂时无须 pad）

<原则：算法前提是长度 2 的幂；非极值填充会污染真实元素。pad 值取不影响真实元素相对顺序的极值，处理完截断回 [:, :n]。pad 长度应有上限避免指数膨胀。与 [[pow2-block-constexpr]] 互补：G2 是 BLOCK 取 pow2，本条是语义维度 pad 到 pow2。>

<骨架（占位名伪码）：
L = triton.next_power_of_2(n)
# pad 值规则：
#   sort 升序 -> +inf（pad 元素排末尾）
#   sort 降序 -> -inf
#   FFT 类 -> 0（zero-pad，不改频谱）
if sort_ascending:
    x_pad = F.pad(x, (0, L - n), value=float('inf'))
else:
    x_pad = F.pad(x, (0, L - n), value=float('-inf'))
# 处理后截断
out = out[..., :n]
>

**坏实践（反例）**：非极值填充或无上限 pad
```python
x_pad = F.pad(x, (0, L - n), value=0.0)  # sort 场景 0 会污染真实元素顺序
```
后果：pad 元素混入真实结果，sort 顺序错误。✅ 正解即极值填充 + 截断 + 上限保护。

**已知实例**：math-compute M2（Sort L1.3 pad 上限 32768、HyenaFft zero-pad 到 fft_size=2*seqlen）

<!-- okf:related:start -->

# 相关

- 相关主题: [数学归一化与算法前置条件必须显式处理（FFT 归一化/方向交替/模归约）](../algorithm-numerics/math-normalize-precondition.md) — 同属优化知识，主题相关
- 相关主题: [Scheduling Optimization: pad slot early exit](../compute-pipeline/exp_causal_conv1d_prefill_005.md) — 同属优化知识，主题相关
- 相关主题: [pad slot 单次检查提前退出，跳过无效填充序列](../compute-pipeline/pad-slot-early-exit.md) — 同属优化知识，主题相关
- 相关主题: [BLOCK 向上取 2 的幂并声明为 constexpr](pow2-block-constexpr.md) — 同属优化知识，主题相关

<!-- okf:related:end -->
