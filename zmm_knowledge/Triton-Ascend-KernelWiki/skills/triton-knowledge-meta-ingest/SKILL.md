---
name: triton-knowledge-meta-ingest
description: "分类6 Agent 元技能（Meta-Action Strategies）知识生产 skill：蒸馏瓶颈诊断树、编译/精度修复链、轨迹级动作时序等高阶决策模式，产出 runbooks/meta-skills/ 优化点。手动触发。"
disable-model-invocation: true
argument-hint: <轨迹路径> [主题]
---

# triton-knowledge-meta-ingest — 分类6「Agent 元技能」知识生产

覆盖二级维度 **6.1–6.3**，落点 `runbooks/meta-skills/`。**本分类是 Triton 知识库区别于传统文档库的核心**：不只沉淀"什么是对的"，更沉淀"Agent 该怎么想、按什么顺序做"。

## 提取点清单（本分类只收这些主题）

| 维度 | 主题 | 提取点 |
|---|---|---|
| 6.1 Bottleneck Diagnosis Tree | 瓶颈诊断树 | Profiling 驱动的 Memory-bound vs Compute-bound 判定链、Roofline 模型定位（引导 Agent 快速精准定位瓶颈） |
| 6.2 Compilation & Accuracy Repair Strategy | 修复策略 | 编译报错修复链（对齐/越界）、精度降级时的自动 Step-up 提升策略（提高 Agent 自动 Debug 与 self-correction 成功率） |
| 6.3 Trajectory-Level Action Sequence | 动作时序 | 多步优化动作组合规则（如：**先调整 Tiling 解决 OOM → 再开 Double Buffering 掩盖 Latency → 最后加 Align Hints**）——提取轨迹中的高阶决策时序模式 |

## 输入

**只收轨迹**（`<轨迹集占位>`，完整 session/report，含失败迭代）——元技能来自对"过程"的分析，文档仓不产生本分类知识。失败轨迹与成功轨迹同等重要：诊断树和修复链的分支往往来自失败路径。

## 工作流（五阶段）

### 阶段 1：采集

通读完整轨迹：每一轮迭代的「观察到的问题 → 采取的动作 → 结果」，特别关注**动作的顺序**和**改判的拐点**（哪个证据让 Agent 改变方向）。

### 阶段 2：提取决策模式（三类）

- **诊断链**：从症状到结论的推理路径（什么 profiling 指标 → 判定为什么瓶颈 → 排除法顺序）；
- **修复链**：报错类型 → 修复动作 → 验证方式的映射表（对齐报错怎么修、越界报错怎么修、精度超标怎么 step-up）；
- **动作时序**：多步优化的有效顺序与**反例顺序**（先做什么后做什么、为什么这个顺序对、颠倒会怎样）。

### 阶段 3：蒸馏落盘

- 诊断树 → `DG-*`（决策指南型条目，本分类特有类型，用决策树/判定表表达）；
- 修复链 → `OPT-*`（修复策略）或 `CT-*`（报错-原因映射）；
- 动作时序 → `OPT-*`（顺序规则）+ `AP-*`（反例顺序）。
- 条目按 [`STRUCTURE-runbook.md`](STRUCTURE-runbook.md) 骨架：`dimension`（6.1–6.3）、泛化层级、「坏实践」必填。
- **每条元技能必须给出判定依据**：不是"应该先调 tiling"，而是"出现 UB 溢出报错时，tiling 调整优先于一切性能优化——因为 kernel 根本跑不起来"。

### 阶段 4：增量合并

同主题诊断/修复链已存在 → 补充分支与反例，不新建；新模式 → 新增 ID。**干净上下文子 Agent 合并**。

### 阶段 5：校验 + 图谱

自检 → `okf_graph.py candidates → judge → inject → viz → verify`。

## 关键原则

- **元技能 = 可执行的决策规则**：每条必须含触发条件（什么症状/指标）与动作（做什么），不收"要系统性思考"这类空话。
- **时序规则必须带反例**：动作顺序条目须给出颠倒顺序的真实失败案例（来自轨迹），否则降 `confidence: provisional`。
- **与其他分类的关系**：元技能引用具体优化点时用跨分类链接（`runbooks/<category>/...`），不复制内容——知识依赖单向性。
