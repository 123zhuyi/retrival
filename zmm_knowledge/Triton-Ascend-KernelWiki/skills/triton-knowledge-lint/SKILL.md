---
name: triton-knowledge-lint
description: "Use when 用户需要在摄入、生成、勘误或提交 PR 前，检查 triton-ascend OKF 知识库的结构、溯源、索引或图谱是否合规。"
---

# triton-knowledge-lint — 全库知识基本原则体检（确定性 + 可选语义深审）

对 `reference/` + `ops/` + `runbooks/`（6 大分类）的全部 concept 卡，按 SPEC 的**基本原则**做一次权威体检。安装后的 skill 入口 `scripts/knowledge_lint.py --knowledge-root <知识库根>` 是**确定性、零-LLM、只读只报**的 linter，一条命令叠加全库原则检查 + 聚合现有 `okf_graph/knowledge_query verify`，单一分级报告。

> 复用 `triton-knowledge-ingest/scripts/okf_graph.py`（图簇）+ `triton-knowledge-ingest/scripts/okf_source.py`（源簇）、`triton-knowledge-query/scripts/knowledge_query.py` 的解析器。

## §0 引擎接口（用 Bash 调，cwd=triton-knowledge-lint skill 目录）
```
python3 scripts/knowledge_lint.py --knowledge-root <知识库根> [--bundle B] [--kind K] [--json]
python3 scripts/knowledge_lint.py --knowledge-root <知识库根> --aggregate-only
python3 scripts/knowledge_lint.py --knowledge-root <知识库根> --sample N [--bundle B] [--kind K]
```
- 退出码：**任一 blocker（或聚合 verify 失败）→ 1**；否则 0。`--json` 输出结构化 findings 供 CI/agent 消费。
- 报告按原则分组打印 `✗ blocker / ! warn / ✓ ok`，尾部 `blocker N / warn M` + `OK|FAIL`。
- **溯源是 per-bundle 的**：`ops` / `triton-ascend-docs` = fixed commit（轨迹路径+commit 或 GitCode @sha）；`triton-ascend-official` = 文档站 URL；`runbooks`/`glossary` 允许空 resource。脚本内置该策略表（占位，知识源补充后校准），不会对文档站 URL/空 resource 误报。

## §1 何时用
- 每次摄入/生成**收尾后**（finalize 之后）、**提交前**、或 CI 关口。
- 改动一批卡后只查该范围：`--bundle <bundle>` / `--kind <kind>`。

## §2 确定性检查清单（脚本，零-LLM）

| 组 | 检查 | 级别 |
|---|---|---|
| **A 命名/结构** | 文件名无前导序号；深度 ≤3；snake_case（CJK 名容忍、runbooks/ops 的 ID 约定豁免大写） | blocker / warn |
| **B frontmatter** | okf.v1 基础字段在位；`kind/type/source_family` 合法（含 `meta_skill`）；无 legacy `timestamp`；无重复 tag；无 merge 冲突标记 | blocker |
| **C 多源溯源** | 有 `sources`：**唯一** primary、`resource==primary.url`、role∈受控词表（含 `trajectory`）；`resource`/`sources[].url` 形态按 **per-bundle 策略**（fixed commit / 文档 URL / 空 OK） | blocker |
| **D 正文** | 无嵌图；`okf:related` 块至多一对；相对链接解析到存在文件；未完成标记 `TODO/FIXME/待补充` | 嵌图/多块 blocker；死链/标记 warn |
| **E index/导航** | 每级 index.md 每条目有非空描述；git-bundle 根 index.md 含 pin（watermark） | blocker |
| **G 日志** | `log/<date>.md`：首行 `# <date>`、`[HH:MM]` 合法、条目倒序。仅全库 lint 跑 | blocker |
| **F 聚合** | shell `okf_graph verify` + `knowledge_query verify`，独立成段 | blocker |

## §3 可选语义深审（agent，超出确定性范围时）
确定性脚本管不了的**语义原则**：蒸馏非照搬、description 与正文一致、index 只列本层、**runbook 条目算子无关（占位名泄漏检查）**、**性能数字有出处**。流程：
1. `knowledge_lint.py --sample N [--bundle/--kind]` 取 JSON worklist。
2. 逐卡（量大时派 subagent）审：蒸馏非照搬、description 准确、index 渐进式披露、runbook 占位名纯度、数字可溯源。
3. 输出 `findings`（severity+位置+建议）。**只报不改**。

## §4 报告判读 & 收尾
- **blocker 必清零**：逐条修卡后复跑至 `OK`。
- **warn 酌情**：死链/未完成标记按优先级清理。
- 聚合段报"cards changed since last judge / 索引 stale" = **图谱/索引未重建**：先跑 finalize（`knowledge_query build` / `okf_graph candidates→judge→inject`）再复跑。

## §5 边界
- 脚本**零-LLM、确定性、只读只报不自动改**；修复交人/上层 agent。
- per-bundle 容忍（文档站 URL、runbooks 空 resource、ID 命名豁免）已内置，**勿**改成一刀切。
- **聚合**非替换：`okf_graph/knowledge_query verify` 仍可单独跑。
