# runbook 条目骨架（`runbooks/memory-sram/`）

> 分类1「内存与切块」泛化优化点库骨架。写条目时严格对齐本骨架；纪律见 [`SKILL.md`](SKILL.md)。

**关键纪律**：扁平 `OPT-*` / `AP-*` / `CT-*`，跨算子单一共享、**增量合并**（已有该点 → 只 append「已知实例」反链；新点 → 新增 ID，**不复用、不重排**）；条目**算子无关**——只用占位名（`inputA`/`blockM`/`blockN`/`ubBudget`/`tileLen`/`dtypeBytes`…），**禁算子业务名、禁 golden 变量名**；标题瘦身 `## OPT-N <短标题>`；每条必含「坏实践（反例）」（无可填实例写「待补充」，不省略）；各字段块与列表项之间空一行。

```markdown
---
schema_version: okf.v1
kind: operator_optimization
type: optimization_runbook
source_family: runbooks
category: memory-sram
dimension: <1.1|1.2|1.3|1.4>
title: <短标题>
description: <一句话：什么场景 + 什么手段 + 什么收益>
tags: [<dimension 主题标签>, <其他标签>]
confidence: verified
status: verified
created_at: '<YYYY-MM-DD>T00:00:00Z'
updated_at: '<YYYY-MM-DD>T00:00:00Z'
---

## OPT-N <短标题>

【优化维度】：搬运｜内存
【泛化层级】：Triton通用 ｜ 条件性（迁移条件：<条件>；失效边界：<边界>）

<原则：一两句说清为什么有效（关联硬件依据：UB 容量 / DMA 对齐 / NZ 依赖）>

<骨架（占位名伪码或公式，可代入）：>

**坏实践（反例）**：<错误写法 → 后果 → ✅ 正解即本条>（无实例则「待补充」）

**已知实例**：<op> §y（<相对路径#锚点>）
```

**CT-N（约束陷阱）**变体：frontmatter 同构，正文为「约束 → 违反后果（报错/静默错误）→ 满足方式」，对齐硬校验类（32B/512B）优先落 CT。

**AP-N（反模式）**变体：无法内联进正向 OPT 的纯坏写法，正文为「坏写法 → 为什么坏 → 正确做法」。
