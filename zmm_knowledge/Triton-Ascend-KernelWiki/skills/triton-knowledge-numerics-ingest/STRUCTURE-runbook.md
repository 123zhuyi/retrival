# runbook 条目骨架（`runbooks/algorithm-numerics/`）

> 分类3「算法与数值」泛化优化点库骨架。纪律同前（扁平 OPT/AP/CT、增量合并、算子无关、坏实践必填），占位名换本域：`accDtype`/`reduceAxis`/`maxVal`/`sumVal`/`logSumExp`/`workspaceBuf`…

```markdown
---
schema_version: okf.v1
kind: operator_optimization
type: optimization_runbook
source_family: runbooks
category: algorithm-numerics
dimension: <3.1|3.2|3.3|3.4>
title: <短标题>
description: <一句话：什么场景 + 什么手段 + 精度与性能各什么变化>
tags: [<dimension 主题标签>, <其他标签>]
confidence: verified
status: verified
created_at: '<YYYY-MM-DD>T00:00:00Z'
updated_at: '<YYYY-MM-DD>T00:00:00Z'
---

## OPT-N <短标题>

【优化维度】：精度｜计算
【泛化层级】：Triton通用 ｜ 条件性（迁移条件：<条件>；失效边界：<边界>）

<原则：为什么有效（数值依据 / 复杂度账 / Ascend 归约机制）>

<骨架（占位名伪码，可代入）：>

**精度影响**：<误差指标变化，含校验口径>
**性能影响**：<可复现数据出处>

**坏实践（反例）**：<错误写法 → 后果 → ✅ 正解即本条>（无实例则「待补充」）

**已知实例**：<op> §y（<相对路径#锚点>）
```

**CT-N（约束陷阱）**：dtype 支持矩阵、数值边界（如 FP8 适用 shape 限制），正文为「约束 → 违反后果 → 满足方式」。

**AP-N（反模式）**：典型如"FP16 直接累加大数组导致精度塌掉"类无法内联的反例。
