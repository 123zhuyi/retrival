# runbook 条目骨架（`runbooks/compute-pipeline/`）

> 分类2「计算与流水」泛化优化点库骨架。纪律同分类1骨架（扁平 OPT/AP/CT、增量合并、算子无关、坏实践必填、ID 不复用不重排），占位名换本域：`gridX`/`gridY`/`numCores`/`stageDepth`/`idxExpr`/`coreIdx`/`repeatTimes`…

```markdown
---
schema_version: okf.v1
kind: operator_optimization
type: optimization_runbook
source_family: runbooks
category: compute-pipeline
dimension: <2.1|2.2|2.3|2.4>
title: <短标题>
description: <一句话：什么场景 + 什么手段 + 什么收益>
tags: [<dimension 主题标签>, <其他标签>]
confidence: verified
status: verified
created_at: '<YYYY-MM-DD>T00:00:00Z'
updated_at: '<YYYY-MM-DD>T00:00:00Z'
---

## OPT-N <短标题>

【优化维度】：计算｜搬运
【泛化层级】：Triton通用 ｜ 条件性（迁移条件：<条件>；失效边界：<边界>）

<原则：为什么有效（关联硬件依据：Scalar 单元弱 / DMA-计算三级流水 / AICore 数量）>

<骨架（占位名伪码或分配公式，可代入）：>

**坏实践（反例）**：<错误写法 → 后果 → ✅ 正解即本条>（无实例则「待补充」）

**已知实例**：<op> §y（<相对路径#锚点>）
```

**CT-N（约束陷阱）**：coreDim ≤ 65535、同步语义误用等，正文为「约束 → 违反后果 → 满足方式」。

**AP-N（反模式）**：无法内联的纯坏写法（如无界 grid、全程 PipeBarrier）。
