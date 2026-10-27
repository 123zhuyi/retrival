# ops 实测案例卡骨架（triton-knowledge-reflux S1 产出）

> 参考既有范式：`ops/gdn-chunked-linear-attention.md`。与静态分析卡（confidence: inferred）的
> 本质区别：本骨架产出的是**硬件实测验证过**的案例卡（confidence: verified, status: draft）。

```markdown
---
schema_version: okf.v1
kind: operator
type: case_study
source_family: ops
title: <op-name> 实测案例：<一句话架构/手段概括>
description: <2-3 句：算子语义 + 最终架构 + 实测结果 + 对哪类任务有参考价值>
tags: [<op 小写标签>, <category>, <架构标签如 ascend910b1>, <核心手段标签>, case_study, triton_ascend]
confidence: verified        # 硬件实测（区别于静态分析卡的 inferred）
status: draft               # 待人工 review 后转 verified
resource: file://<工作目录>/report.md
created_at: '<YYYY-MM-DD>T00:00:00Z'
updated_at: '<YYYY-MM-DD>T00:00:00Z'
---

sources:
  - url: file://<工作目录>/report.md
    role: primary

## 实测记录（verified run）

- **arch**: <arch> | **生成任务**: <工作目录名> | **日期**: <YYYY-MM-DD>
- **精度**: <passed>/<total> shapes 全过（证据：`output/iter_{N}/verify/verify_result.json`）
- **几何平均加速比**: <best_speedup>（口径：<vs PyTorch eager 现场实测 | vs aclnn 实测（<framework_latency_source>）>，证据：`summary.json#perf_data.speedup_vs_torch`）
- **目标达成**: target_speedup=<X>，target_reached=<true/false>
- **基线口径警示**：per_shape 的 speedup 为「该 shape vs 全局 geomean」时仅顶层几何平均有对比意义

| case | shape | framework (ms) | impl (ms) | speedup |
|---|---|---|---|---|
| （从 summary.json#perf_data.per_shape_results 摘录代表性 case，全量指回轨迹） |

## 架构与关键决策

<从 report.md/sketch.txt 蒸馏：kernel 拆分、grid 设计、host 门控 dispatch 条件与依据、
多路径取舍。写成"决策 → 依据 → 实测支撑"，不贴大段代码。>

## 优化历程（Phase 4 摘要）

<逐轮：命中优化点 → 结果（提升/无提升/平台不可用）→ 归因。无提升轮同样要记——
这是 S3 已知实例 append 的来源，也是他人复用时最想知道的"哪些路走不通"。>

## 复用决策要点

1. <什么条件下该用本架构（如：T 超过阈值切 chunked 路径）>
2. <关键参数经验（如 chunkSize 先试 128）>
3. <已知死路（平台 API 不可用 / 无提升的优化点）>

<!-- okf:related:start -->（由 okf_graph.py inject 生成）

# 相关

- 实践案例: <指向被消费过的 runbook 条目，双向>
```

## 写卡纪律

1. **每个数字都有出处**：表内数据从 `summary.json#perf_data.per_shape_results` 摘录；摘要性结论括注 evidence 字段路径。工作目录是易失的（可能被清理），若已归档则 evidence 指向归档路径，并在卡内注明归档位置。
2. **同名算子多次生成**：不建第二张卡。在既有卡上追加「## 实测记录（<日期>，<工作目录名>）」小节或合并实测表，新架构与旧架构**并列记录**（多样性保护），不覆盖。
3. **正文禁摄入口吻**：「本次生成任务」改写为「该实测运行」；ingest 过程只进 `log/`。
4. **frontmatter 不加私有字段**：实测元信息放正文「实测记录」节，不扩 frontmatter schema（kind/type 白名单约束）。
5. **回链义务**：卡内至少回链 2 张被消费/相关的 runbook 条目（「已知实例 ⇄ 实例化」），孤卡不过 lint 语义审计。
