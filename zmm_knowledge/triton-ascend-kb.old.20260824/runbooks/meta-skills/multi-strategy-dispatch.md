---
schema_version: okf.v1
kind: operator_optimization
type: optimization_runbook
source_family: runbooks
category: meta-skills
dimension: "6.3"
title: 按 (维度, 模式, 采样比例) 多策略分派，禁止单 kernel 统所有路径
description: host 侧按特征分派到特化 kernel，通用 kernel 仅作兜底；禁止单一通用 kernel 统一处理所有路径
tags: [multi_strategy, dispatch, specialization, no_universal_kernel]
confidence: verified
status: verified
created_at: "2026-08-05T00:00:00Z"
updated_at: "2026-08-05T00:00:00Z"
---

## OPT-1 多策略分派原则

【优化维度】：元技能｜设计决策
【泛化层级】：条件性（迁移条件：算子计算特征高度异构--不同 mode/比例/维度的最优数据访问模式完全不同；失效边界：特征单一的同构算子无须分派）

<原则：异构算子（如重采样有 nearest/bilinear/bicubic × 上/下采样 × 整数/分数倍）各场景最优数据访问模式完全不同。单一通用 kernel 用 gather 统一处理会丢失所有特化机会，坐标解码 overhead 极大。必须 host 侧按特征分派到特化 kernel，通用 kernel 仅兜底。>

<骨架（占位名伪码）：

```python
# host 侧 _route 分派
if mode == 'nearest' and H_out == 2*H_in and W_out == 2*W_in:
    nearest_2x_upsample_kernel[grid](...)      # 整数倍上采样：reshape broadcast
elif mode == 'nearest' and H_in == 2*H_out and W_in == 2*W_out:
    nearest_half_downsample_kernel[grid](...)  # 0.5x 下采样：strided load
else:
    general_kernel[grid](...)                  # 兜底通用路径
```

>

**坏实践（反例）**：单 kernel 统所有路径
```python
# 一个 universal kernel 用 gather 处理所有 mode 和比例
universal_resample_kernel[grid](x, output, mode, scale_h, scale_w, ...)
```
后果：坐标解码 overhead 极大，丢失特化机会；历史上出现过迭代 58 轮仍无法性能/精度双达标。✅ 正解即按特征多策略分派。

**已知实例**：tensor-transform G3、Interpolate L1.1（`triton-op-generator/template/tensor-transform.md`）

<!-- okf:related:start -->

# 相关

- 实践案例: [FusedFloydAttention (Floyd multi-dim attention with dual KV groups) on Ascend NPU](../../ops/fused-floyd-attention.md) — 同属优化知识，主题相关
- 相关主题: [Moe Dispatch Skeleton](../../reference/skeleton-moe_dispatch.md) — 同属优化知识，主题相关
- 相关主题: [matmul core mapping 策略选择（列/行/对角）](../compute-pipeline/core-mapping-strategy-selection.md) — 同属优化知识，主题相关
- 相关主题: [关键长度参数必须 tl.constexpr 静态化触发启动级特化](../portability/constexpr-launch-specialization.md) — 同属优化知识，主题相关
- 相关主题: [precompute dispatch（预反量化分派，gs<=64 或大 M 中等 N 启用）](precompute-dispatch.md) — 主题相关：共享 dispatch
- 相关主题: [量化 matmul autotune 配置策略诊断--按 shape 范围特化](quant-matmul-autotune-strategy.md) — 同属优化知识，主题相关

<!-- okf:related:end -->
