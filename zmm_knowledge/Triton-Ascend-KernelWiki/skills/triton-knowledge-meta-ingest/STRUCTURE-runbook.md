# runbook 条目骨架（`runbooks/meta-skills/`）

> 分类6「Agent 元技能」库骨架。除 OPT/AP/CT 外新增 **DG-***（决策指南，诊断树/判定表）。纪律同前（扁平编号、增量合并、坏实践必填），条目以"触发条件 → 动作"为核心。

```markdown
---
schema_version: okf.v1
kind: meta_skill
type: decision_guide
source_family: runbooks
category: meta-skills
dimension: <6.1|6.2|6.3>
title: <短标题>
description: <一句话：什么症状下 + 按什么规则 + 做什么>
tags: [<dimension 主题标签>, <其他标签>]
confidence: verified
status: verified
created_at: '<YYYY-MM-DD>T00:00:00Z'
updated_at: '<YYYY-MM-DD>T00:00:00Z'
---

## DG-N <短标题>   # 决策指南；修复策略/时序规则用 OPT-N

【触发条件】：<什么症状 / 什么 profiling 指标 / 什么报错>

【判定/动作规则】：
<决策树或映射表，例如：>
<UB 溢出报错 → 先调 Tiling（kernel 跑不起来时性能优化无意义）>
<memory-bound 判定 → 走 runbooks/memory-sram 对应 OPT>
<compute-bound 判定 → 走 runbooks/compute-pipeline 对应 OPT>

【判定依据】：<为什么这个规则成立（含轨迹出处）>

【关联优化点】：<跨分类链接，不复制内容>

**坏实践（反例）**：<错误的顺序/误判 → 后果 → ✅ 正解即本条>（无实例则「待补充」）

**已知实例**：<轨迹/算子>（<相对路径#锚点>）
```

**纪律**：每条必须含触发条件与判定依据；时序规则无反例时 `confidence: provisional`；跨分类只链接不复制。
