# runbook 条目骨架（`runbooks/autotuning/`）

> 分类5「调优空间设计」库骨架。纪律同前（扁平 OPT/AP/CT、增量合并、算子无关、坏实践必填），占位名换本域：`blockCandidates`/`numStages`/`tuneKey`/`shapeDim`…

```markdown
---
schema_version: okf.v1
kind: operator_optimization
type: optimization_runbook
source_family: runbooks
category: autotuning
dimension: 5.1
title: <短标题>
description: <一句话：剪掉什么/怎么设计 key + 省多少调参时间 + 性能是否保住>
tags: [autotuning, <其他标签>]
confidence: verified
status: verified
created_at: '<YYYY-MM-DD>T00:00:00Z'
updated_at: '<YYYY-MM-DD>T00:00:00Z'
---

## OPT-N <短标题>

【优化维度】：调优效率
【泛化层级】：Triton通用 ｜ 条件性（迁移条件：<条件>；失效边界：<边界>）

<原则：为什么这些候选/这个 key 设计是合理的（UB 约束 / 对齐约束 / shape 泛化）>

<骨架（可执行的剪枝判据或 key 设计模板，占位名）：>

**调参时间影响**：<数据出处>
**最终性能影响**：<数据出处>

**坏实践（反例）**：<如：全组合暴力搜索 → 编译爆炸 → ✅ 正解即本条>（无实例则「待补充」）

**已知实例**：<op> §y（<相对路径#锚点>）
```

**CT-N（约束陷阱）**：autotune 与动态 shape / 编译缓存的交互坑，正文为「约束 → 违反后果 → 满足方式」。
