---
name: triton-knowledge-ingest
description: "triton-ascend 知识库摄入顶层编排器：大原则（新 source 接入 / 批次级增量 / 大版本升级三路由）+ 6 大分类生产 skill 与 runbooks 治理边界。持跨树共享图引擎脚本（okf_graph.py / okf_judge_aggregate.py）。不亲自产卡——正文著作委派对应分类生产 skill。手动触发。"
disable-model-invocation: true
---

# triton-knowledge-ingest — 知识库摄入顶层编排器

把 Agent 轨迹和上游文档编译进本库 `reference/` / `ops/` / `runbooks/` 内容树，并维护 6 大分类 runbook 的治理边界。**本 skill 是编排器**：持大原则 + 三路由 + 跨树共享图引擎，**不亲自产卡**——正文著作委派给对应分类的生产 skill。

> 架构定位：本 skill（编排器）+ 6 个分类生产 skill（见 §6 链接表）。与 AscendC 版 `cannbot-knowledge/skills/ops-knowledge-ingest` 同构，差异在生产层：Triton 知识按 **6 大分类 / 18 个二级维度** 组织（依据 `docs/triton-knowledge-design/triton ascend 知识库内容设计.md`），且主要从 **Agent 轨迹**蒸馏"诊断-修改-验证"技能。

## §0 必读背景

- **本库是什么**：triton-ascend-knowledge 是面向昇腾 NPU / Triton-Ascend DSL 算子开发的 OKF 知识库，治理成可被「大模型检索 + 人渐进式浏览」消费的知识。
- **三个内容树**：
  - **`reference/`** — triton-ascend 官方文档、API、IR 结构化抽取的 OKF 知识库（`<上游文档仓占位>`，知识源后续补充）。
  - **`ops/`** — 按算子的成功案例设计 wiki（tiling 方案、UB 预算、性能数据）。生产来源：成功探索轨迹（`<轨迹集占位>`，如 triton_ascend_output 的 report/summary/session）。
  - **`runbooks/`** — **核心**：6 大分类优化点库 + 实战 field notes + 版本迁移笔记。每个分类一个子目录，单一共享、增量合并：
    - `runbooks/memory-sram/`（分类1 内存与切块：访存效率 / Tiling·SRAM 预算 / ND↔NZ / 32B·512B 对齐）
    - `runbooks/compute-pipeline/`（分类2 计算与流水：单元利用·Scalar 削减 / 流水掩盖 / 多核负载·coreDim / 融合）
    - `runbooks/algorithm-numerics/`（分类3 算法与数值：算法重构 / 混合精度 / 数值稳定 / 跨 Block 归约）
    - `runbooks/portability/`（分类4 规范与移植：动态 shape / 编译器约束·hints / 910B·910C·310P）
    - `runbooks/autotuning/`（分类5 调优空间设计：搜索空间剪枝 / num_stages / autotune key）
    - `runbooks/meta-skills/`（分类6 Agent 元技能：瓶颈诊断树 / 修复链 / 轨迹动作时序）
- **跨树共享层**：`graph/`（知识图谱产物）、`search/`（检索索引）、`log/`（按天审计日志）、根 `index.md`。
- **知识 ≠ 流水账**：卡片记整合后的系统知识；摄入过程（轨迹区间、批次、核实动作）只进 `log/`，不进卡片、不进 frontmatter。

## §1 会话定向（每次必做，先读后写）

1. **规范与生产 skill**：目标知识库根目录下的 `SPEC-*.md`（占位，从 Asc 规范改写）；各分类生产 skill（见 §6 链接表）。
2. **各树入口**：`reference/index.md`、`ops/index.md`、`runbooks/index.md` 及目标分类的 `runbooks/<category>/index.md` — 已有实体，防重复造卡。
3. `log/` 最近 1–3 天 `<YYYY-MM-DD>.md` — 近期操作，防重复劳动。

对大库还要 `grep` 确认目标是否已覆盖，而不只看 index。

## §2 输入路由（三路由）

| 路由 | 触发 | 执行 skill | 流程子文档 |
|---|---|---|---|
| **新 source 接入** | 首次接入某上游（新轨迹集 / 新文档仓 / 新算子批次） | 按知识分类路由到 §6 对应生产 skill | [`references/new-source-onboarding.md`](references/new-source-onboarding.md) |
| **批次级增量** | 同 source 推进新批次（新一批轨迹、文档仓新 commit 区间） | 同上（各分类 skill 增量更新） | [`references/incremental-sync.md`](references/incremental-sync.md) |
| **大版本升级** | triton-ascend 版本切换（含 API/IR 变更可能） | 编排器主导 + 各分类 skill 分流受影响条目；详见目标知识库根目录下的 `SPEC-Version-update.md`（占位） | [`references/version-bump.md`](references/version-bump.md) |

**判路由**：看上游变化性质——新仓/新轨迹集=新 source 接入；同 source 推进=批次级增量；版本切换=大版本升级。版本升级时可由编排器沉淀架构性变更到 `runbooks/<对应分类>/` 的版本迁移条目。

## §3 摄入不变量（所有路由通用）

### §3.1 单实体原子动作（建/更新任何一个卡都走这五步）
1. **理解源** — 读全相关一手材料（轨迹的 report/summary/session、文档原文）。
2. **影响分析** — 查重（是否已有卡？）、定范围（建新卡 vs 增强既有卡）、找受影响的其它卡。
3. **建/更新卡** — 按对应分类生产 skill 的规范写。**更新既有卡前先读它现在写了什么，在其上增量补充，绝不整体覆盖**。
4. **交叉链接** — 相关卡间用相对链接，且**双向**；runbook 条目 ⇄ ops 案例卡经「实例化 / 已知实例」互链。
5. **维护三件套** — 写完卡必同步：**逐层 `index.md`**；**检索/图谱重建**（见 §4）；当天 **`log/<YYYY-MM-DD>.md`** 顶部插入 `## [HH:MM] <op> | <题>`。

### §3.2 红线（不变量）
- **frontmatter 必填非空 okf.v1 路由字段**：`schema_version`/`kind`/`type`/`source_family`/`title`/`description`/`tags`/`created_at`/`updated_at`；runbook 条目卡另需 `dimension`（二级维度编号，如 `1.2`）；`resource` 按 profile 要求填写。
- **source pin 在内容根 `index.md` frontmatter**（轨迹集 `trajectory_repo`/`trajectory_ref`/`trajectory_commit` 或文档仓 `upstream_repo`/`upstream_ref`/`upstream_commit`），即 watermark。不建静态 `resource/` 清单层。
- **每卡 ≥1 source、有且仅一 `primary`、`resource==primary.url`**；轨迹源 = 仓内 session/report 路径 + commit，文档源 = 上游 @sha URL。
- **蒸馏非照搬**、`snake_case` 无数字前缀、目录扁平 ≤3 层（自内容根）、正文不嵌图片。
- **runbook 条目算子无关**（占位名骨架）；算子特定事实落 `ops/` 案例卡。
- **GPU 概念须转换**：CUDA/SIMT 语境概念（Warp/Wavefront 等）不得直接入库，必须映射为 AICore 多核调度 / Vector·Cube 单元 / Mask·Repeat 控制后再写。
- **矛盾绝不静默覆盖**：并存 + 日期/来源 + `> ⚠️ 矛盾标记`。
- **每卡自包含**；**正文禁摄入/变更口吻**（「本次批次 / 留待后续」只进 `log/`）。

### §3.3 溯源纪律（本库最硬的一条）
- **核到一手源**（轨迹中的 benchmark 数据 / kernel 代码 / 报错原文 / 官方文档），**转述 ≠ 权威**。
- **性能数字必须可复现**：runbook/案例卡中的加速比、延迟必须能指回具体轨迹的 perf_result/benchmark 记录，禁止估算编造。
- **无一手可核的断言** → 标 `unverified`/`gap`，不冒充事实。

### §3.4 质量护栏
- **不过度提炼**：写得好的机制、关键代码片段该保留就保留并标出处。
- **一概念一 canonical 卡**：同类优化点**丰富既有条目**（append 已知实例），不按案例复制第二份。
- **复用优先**：已有卡只双向链接、不重复造；**串行单写者**：有概念交集的实体串行摄入。
- **干净上下文合并**：runbook 已存在时的去重合并，另起干净上下文子 Agent 执行（防被单算子细节污染而误泛化）。

## §4 跨树共享设施（图引擎）

图引擎脚本留在本 skill 的 `scripts/`，被各分类生产 skill 跨树调用（路径不变）：

- `scripts/okf_graph.py` — 知识图谱主程序：`candidates`（确定性召回）→ `judge`（agent fan-out LLM 判定）→ `inject`（写卡片 `# 相关`）→ `viz`（生成 `graph/viz*.html`）→ `verify`（校验闭环）；`related`/`explain` 读时检索。
- `scripts/okf_judge_aggregate.py` — 把判定结果聚合为 `graph/edge_judgments.json`。

**新增/改动卡片后必须重跑图谱**：`candidates → judge → inject → viz → verify`。图谱判定缓存位于目标知识库根目录的 `graph/edge_judgments.json`，按内容指纹增量复判，卡片不变则零 LLM。

## §5 红线速查（任何路由都不许碰）
- [ ] **正文著作委派各分类生产 skill**，本编排器不亲自产卡、不重写各分类格式。
- [ ] **不建 `resource/` 清单**；source pin 在内容根 index.md（= watermark）。
- [ ] **每卡有且仅一 `primary`、`resource==primary.url`**；轨迹源带路径+commit，文档源带 @sha。
- [ ] **多源是融合蒸馏非多段拼接**；蒸馏非照搬。
- [ ] **runbook 条目算子无关**；GPU 概念先转换再入库。
- [ ] **性能数字必须可复现**（指回轨迹 benchmark 记录），禁止编造。
- [ ] **收尾跑完图谱增量**（`candidates → judge → inject → viz → verify`）后才算落地；改动集 + log 一起 commit。
- [ ] **watermark 是唯一「已落地」开关**；**watermark 缺失则停下问基线，不猜**。
- [ ] 不擅自改 `SPEC-*.md` / 生产 skill（约定冻结，新约定先经人类批准）。
- [ ] 不创建孤卡（每卡 ≥2 双向链接）；提交/推送/提 MR 只在用户明确要求时做。

## §6 各分类生产 skill 链接表

| 分类 | 内容树 | 生产 skill | 覆盖二级维度 |
|---|---|---|---|
| 1 内存与切块 | `runbooks/memory-sram/` | [`triton-knowledge-memory-ingest`](../triton-knowledge-memory-ingest/SKILL.md) | 1.1 访存效率 · 1.2 Tiling/SRAM 预算 · 1.3 ND↔NZ · 1.4 32B/512B 对齐 |
| 2 计算与流水 | `runbooks/compute-pipeline/` | [`triton-knowledge-compute-ingest`](../triton-knowledge-compute-ingest/SKILL.md) | 2.1 单元利用/Scalar 削减 · 2.2 流水掩盖 · 2.3 多核负载/coreDim · 2.4 融合 |
| 3 算法与数值 | `runbooks/algorithm-numerics/` | [`triton-knowledge-numerics-ingest`](../triton-knowledge-numerics-ingest/SKILL.md) | 3.1 算法重构 · 3.2 混合精度 · 3.3 数值稳定 · 3.4 跨 Block 归约 |
| 4 规范与移植 | `runbooks/portability/` | [`triton-knowledge-portability-ingest`](../triton-knowledge-portability-ingest/SKILL.md) | 4.1 动态 shape/兼容 · 4.2 编译器约束/hints · 4.3 跨芯片适配 |
| 5 调优空间设计 | `runbooks/autotuning/` | [`triton-knowledge-autotuning-ingest`](../triton-knowledge-autotuning-ingest/SKILL.md) | 5.1 搜索空间剪枝/num_stages/autotune key |
| 6 Agent 元技能 | `runbooks/meta-skills/` | [`triton-knowledge-meta-ingest`](../triton-knowledge-meta-ingest/SKILL.md) | 6.1 瓶颈诊断树 · 6.2 修复链 · 6.3 轨迹动作时序 |

> **reference/ 与 ops/ 的生产**：reference 文档蒸馏与 ops 案例卡的写入也由上表按知识主题路由（例如 ND↔NZ 格式的官方文档卡由 memory-ingest 产出到 `reference/`，算子案例卡由主导分类的 skill 产出到 `ops/`）。
>
> **配套工具**：[`triton-knowledge-query`](../triton-knowledge-query/SKILL.md)（检索）、[`triton-knowledge-lint`](../triton-knowledge-lint/SKILL.md)（全库原则体检）。

## §7 路由流程子文档

三路由的详细 pipeline 在 [`references/`](references/) 子文档（渐进式披露）：
- [`new-source-onboarding.md`](references/new-source-onboarding.md) — 新 source 接入 pipeline
- [`incremental-sync.md`](references/incremental-sync.md) — 批次级增量同步 pipeline
- [`version-bump.md`](references/version-bump.md) — 大版本升级 pipeline（摘要，权威见目标知识库根目录的 `SPEC-Version-update.md` 占位）
