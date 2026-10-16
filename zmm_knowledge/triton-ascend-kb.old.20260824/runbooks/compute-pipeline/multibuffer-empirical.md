---
schema_version: okf.v1
kind: operator_optimization
type: optimization_runbook
source_family: runbooks
category: compute-pipeline
dimension: "2.2"
title: multibuffer/unit_flag 等编译选项需实测验证，禁止默认开启
description: 内存密集型算子的 multibuffer 在含迭代标量循环的算子上可能劣化，必须 Phase 4 实测后再决定
tags: [multibuffer, unit_flag, empirical, no_default_enable]
confidence: verified
status: verified
created_at: "2026-08-05T00:00:00Z"
updated_at: "2026-08-05T00:00:00Z"
---

## OPT-6 multibuffer 实测验证

【优化维度】：计算｜流水掩盖
【泛化层级】：条件性（迁移条件：内存密集型算子考虑开 multibuffer；失效边界：纯 element-wise 无循环时通常有收益，含标量迭代循环时可能劣化）

<原则：multibuffer 的流水线优化收益在含迭代标量循环（如 top-k 逐元素比较）的算子上会被迭代开销掩盖，实测可能劣化。禁止默认开启，必须 Phase 4 实测对比。>

<骨架（占位名伪码）：
# Phase 4 实测两版对比
kernel[grid](..., multibuffer=False)   # 基线
kernel[grid](..., multibuffer=True)    # 对照
# 仅当 benchmark 显示提升且精度不变才采纳
>

**坏实践（反例）**：盲目添加 multibuffer=True
```python
kernel[grid](..., multibuffer=True, unit_flag=True)  # 期望提升，未实测
```
后果：MoeGatingTopKSoftmax v3 实测 0.8527x vs 基线 0.8836x（劣化）；v10 实测 0.5720x vs 0.8194x（严重劣化）。✅ 正解即实测验证后再启用。

**已知实例**：transformer-inference T6（MoeGatingTopKSoftmax v3/v10）

<!-- okf:related:start -->

# 相关

- 相关主题: [SDA（Scaled Dot Attention）Ascend 硬件适配：HiVM 分块、多缓冲预取与 Cube/Vector 均衡](../portability/traj-misc-hardware_adaptation.md) — 主题相关：共享 multibuffer

<!-- okf:related:end -->
