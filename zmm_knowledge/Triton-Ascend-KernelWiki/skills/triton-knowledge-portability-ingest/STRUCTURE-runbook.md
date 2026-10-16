# runbook 条目骨架（`runbooks/portability/`）

> 分类4「规范与移植」库骨架。纪律同前（扁平 OPT/AP/CT、增量合并、算子无关、坏实践必填），占位名换本域：`socVersion`/`blockK`/`multipleOfHint`/`dynShape`…

```markdown
---
schema_version: okf.v1
kind: operator_optimization
type: optimization_runbook
source_family: runbooks
category: portability
dimension: <4.1|4.2|4.3>
title: <短标题>
description: <一句话：什么平台/版本 + 什么问题 + 什么手段>
tags: [<dimension 主题标签>, <平台标签>]
platforms: [910b, 910c, 310p]   # 按实际验证填写，未验证平台不得列入
confidence: verified
status: verified
created_at: '<YYYY-MM-DD>T00:00:00Z'
updated_at: '<YYYY-MM-DD>T00:00:00Z'
---

## CT-N <短标题>   # 本分类多为约束陷阱；正向优化用 OPT-N

【适用条件】：平台 <...> ｜ triton-ascend 版本 <...> ｜ dtype <...> ｜ shape <...>

<约束：限制是什么（编译器/平台/版本）>

<违反后果：报错原文或行为差异（带出处）>

<满足方式：workaround / hint / 平台分支（占位名骨架）>

<回退条件（如为 workaround）：哪个版本起不再需要>

**坏实践（反例）**：<错误写法 → 后果 → ✅ 正解即本条>（无实例则「待补充」）

**已知实例**：<op> §y（<相对路径#锚点>）
```

**纪律**：适用条件四元组（平台/版本/dtype/shape）必填；未验证平台写 `unverified`，不猜测。
