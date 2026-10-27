---
name: triton-knowledge-reflux
description: "triton-ascend-kb 回流生产 skill：单算子任务粒度，把 triton-op-generator 一次生成任务的过程与结果（summary/report/迭代日志/知识消费记录）提取为 OKF 卡片回流知识库——ops 实测案例卡 + runbooks 泛化条目（OPT/AP/CT/DG）+ 已消费卡的「已知实例」反链。op-generator Phase 6 的执行体。手动触发或由 op-generator Phase 6 调用。"
disable-model-invocation: false
argument-hint: <op-generator 工作目录> [--output <manifest 路径>]
---

# triton-knowledge-reflux — 生成轨迹 → 知识库回流

单算子任务粒度的知识回流 skill。op-generator 每完成一次算子生成任务，工作目录里就留下了一份**实测验证过**的知识原料（精度全过的 kernel、逐轮 benchmark 数据、错误修复闭环、平台不可用记录、被消费的知识卡清单）——本 skill 负责把这些原料蒸馏成 OKF 卡片，走 lint + staging + 图谱管线落进 triton-ascend-kb。

**与 6 大分类 ingest skill 的关系**：正文规范（runbook 骨架、占位名、dimension、泛化层级、坏实践必填）**完全沿用**各分类生产 skill，本 skill 不另立格式；本 skill 的增量价值在——① 单任务粒度自动触发判定；② 确定性收集脚本产出证据骨架；③ S4 消费反馈反链（ingest 系没有的信号源）。

## §0 定位与数据流

```
triton-ascend-kb ──(op-generator Phase 2/3.4/4.1 knowledge_query 读)──▶ op-generator
       ▲                                                                    │
       └──(本 skill: collect → 蒸馏 → 产卡 → lint → staging → judge → inject)──┘
```

- **输入**：op-generator 工作目录（`triton_ascend_output/op_{index}_{op_name}_{ts}_{rid}/`）
- **输出**：`ops/` 案例卡（confidence: verified）、`runbooks/<category>/` 条目或已知实例反链、`log/` 审计、staging 候选
- **本库最稀缺的就是实测知识**：现有 ops 卡几乎全是 `inferred`（静态分析），本 skill 产出的卡全部带硬件实测数据。

## §1 采集（先跑脚本，后读材料）

```bash
python3 <本skill目录>/scripts/collect_from_workdir.py --work-dir <工作目录>
```

脚本产出 `<工作目录>/reflux_manifest.json`，内容分两区：

- **确定性区（脚本填，直接可信）**：`run_facts`（success/迭代数/speedup/pass 率，带 json 字段路径 evidence）、`phase3_iterations`（逐轮 verify + failures 明细）、`phase4_iterations`（逐轮命中点/基线对比/verdict_hint）、`consumed_cards`（S4 消费清单）、`triggers`（S1-S4 判定）、`warnings`
- **蒸馏区（`agent_fill`，Agent 填）**：`candidates[]` 候选知识条目 + `case_card`

脚本只做收集与判定，**蒸馏是 Agent 的职责**。逐张读以下一手材料（按需深读，session.md 只在细节存疑时查）：

1. `report.md` — 架构决策、Phase 4 优化历程表、失败归因
2. `sketch.txt` + `precheck.json` — 设计意图与 Layer 1 约束（判断「设计 vs 实现」谁错了）
3. `output/opt_iter_*/log.md` — 每轮优化动作与状态机推导（命中点、无提升原因、禁用）
4. `output/iter_*/log.md` + `verify_result.json` failures — 错误原文与修复方向
5. `94_*.py`（任务文件）与 `{op_name}_generated.py` — 代码级证据（提取片段时用）

## §2 触发判定（S1-S4）

以 manifest `triggers` 为准（脚本已判），Agent 复核后执行：

| Source | 触发条件 | 产出 | 落点 |
|---|---|---|---|
| **S1 成功案例** | `success=true` ∧ `passed_cases==total_cases>0` ∧ `speedup_vs_torch≥1.1` | 1 张 ops 实测案例卡 | `ops/<op-name>.md`（骨架见 [STRUCTURE-case-card.md](references/STRUCTURE-case-card.md)） |
| **S2 调试经验** | 存在「错误 → 修复 → 通过」闭环（Phase 3 迭代对 / Phase 4 verify 失败回退） | CT/AP/DG 条目或已知实例 | 按错误主题路由到对应 `runbooks/<category>/`；通用修复链落 `runbooks/meta-skills/` |
| **S3 性能洞察** | 存在 Phase 4 优化轮（**含 no_improvement 与永久禁用**——"点 P 在此类算子无效"、"平台 API 不存在"都是高价值 CT） | OPT/AP/CT 条目或已知实例 | 对应分类 `runbooks/<category>/` |
| **S4 消费反馈** | 本次运行消费了知识库卡（manifest `consumed_cards` 非空） | 被消费卡的「已知实例」反链 / 使用反馈 | 各被消费卡所在文件 |

四源**全部未触发** → 输出 §7 盒装摘要（各源标注未触发原因）后结束，不产任何卡。
**失败轨迹同样回流**（S2/S3），但仅当错误已定位到可泛化原因；纯环境抖动（B 类）不收。

## §3 会话定向（防重复，每次必做）

1. `ops/index.md` — 是否已有同名/同构算子的案例卡（同名算子多次生成 → **增强既有卡**：新 run 作为一节追加或更新实测表，不建第二张）
2. 目标 `runbooks/<category>/index.md` — 候选条目是否已有同类 OPT/AP/CT（有 → 只做已知实例 append，见 [append-known-instance.md](references/append-known-instance.md)）
3. `log/` 最近 1-3 天 — 防重复劳动
4. 大库场景 `grep` 全文确认，不只看 index

## §4 蒸馏（agent_fill.candidates）

逐条候选知识四要素**全部必填且带出处**，无出处不入库：

```json
{
  "id": "cand_001",
  "type": "OPT",                    // OPT 正向优化点 / AP 反模式 / CT 约束陷阱 / DG 诊断链
  "dimension": "2.2",               // 对应分类的二级维度，见各分类 ingest skill 提取点清单
  "title": "K2 循环内 load 提前发射",
  "diagnosis": "wDK/k_t/wED 仅依赖 cc，却晚于循环首条指令发射，DMA 空等（出处: output/opt_iter_0/log.md）",
  "modification": "三 load 提前至循环开头与 wW/wU/wM/eG 一并发射（出处: output/opt_iter_0/optimized_code.py）",
  "verification": "geomean 22.0006 vs 基线 22.2468 → 无提升（出处: output/opt_iter_0/optimized_perf_result.json.speedup_vs_torch vs baseline_perf_result.json.speedup_vs_torch）",
  "generalization": "条件性（失效边界：fw 侧 profiling 噪声与 impl 差值同量级时不可判）",
  "placeholders": "wDK/k_t/wED → loadA/loadB/loadC",
  "bad_practice": "依赖就绪的 load 留在循环后段串行发射",
  "route": {"target": "runbooks/compute-pipeline", "existing_entry": "loop-invariant-code-motion.md"}
}
```

**硬门禁**：
- `verification` 必须指回 manifest/轨迹的 **json 字段路径**（如 `output/opt_iter_0/optimized_perf_result.json.speedup_vs_torch`），**禁止估算编造数字**——本库最硬的溯源纪律
- 无一手可核的断言（如仅凭直觉推测的瓶颈）标 `unverified`，落 `log/` 不落卡
- runbook 条目**算子无关**：占位名（`inputA`/`blockM`/`loadA`…），禁算子业务名；GPU 概念（Warp/Wavefront）先转 AICore/Vector/Cube 语境再入库
- 「坏实践（反例）」必填；提炼非照搬，代码片段 ≤15 行

**S3 特别规则**：`no_improvement` 轮蒸馏为「已知实例」append 到该优化点条目（格式"在 <算子类别>/<条件下> 无效：<数据出处>"）；`hit_optimization_point` 命中但平台不可用/被禁用 → 新 CT 条目（约束 → 违反后果 → 满足方式）。

**S4 特别规则**：只对 `relevance_level == strong` 且实际影响了设计/修复决策的卡回链（manifest `consumed_cards` 里 `recommended_action == read` 的子集，Agent 逐张确认）；append 内容为"<本算子>（<category>，arch）案例：<一句话使用效果>（<ops 卡相对路径#锚点>）"。**弱命中/未实际消费的不回链**，防反馈膨胀。

## §5 产卡与落盘

### S1 案例卡
按 [STRUCTURE-case-card.md](references/STRUCTURE-case-card.md) 写 `ops/<op-name>.md`：
- frontmatter：`kind: operator`、`type: case_study`、`source_family: ops`、`confidence: verified`（硬件实测）、`status: draft`（待人工 review 转 verified）
- `sources:` primary = 轨迹 `report.md` 的 `file://` 绝对路径（工作目录归档后更新为归档路径）；`resource == primary.url`
- 正文含「实测记录」节：arch、日期、shape 数、几何平均加速比、pass 率、基线来源口径（vs torch eager / vs aclnn，取 `framework_latency_source` 判断）
- 与 runbook 条目「实例化 ⇄ 已知实例」双向回链

### 泛化条目（S2/S3）
- 路由到对应分类生产 skill 的格式规范（骨架见各分类 `STRUCTURE-runbook.md`），本 skill 不重复定义
- **增量合并**：既有同类条目 → 只 append 已知实例，**不新建**；确为新点 → 新增编号（ID 不复用不重排）。合并判定由**干净上下文子 Agent**执行（防单算子细节污染泛化判断）
- 多来源融合蒸馏，不按案例复制第二份

### 落盘管线
1. 新卡先写 staging：`<知识库根>/.build/cand/`（**注意 kind 白名单**，见 .build/cand/manifest.json 既有约定），通过 judge 后 inject 进正式树；小步快跑时可与知识库维护者确认直写 + 补 staging 记录
2. **逐层 index 更新**（`ops/index.md`、`runbooks/<cat>/index.md`，按既有"新增条目（日期）"小节风格）
3. **图谱重建**：`okf_graph.py candidates → judge → inject → viz → verify`（脚本在 `triton-knowledge-ingest/scripts/`，`--knowledge-root` 指向知识库根）
4. `log/<YYYY-MM-DD>.md` 顶部插入 `## [HH:MM] <op> | reflux: <一句话>`，记改动集与证据指针
5. 新卡跑 `triton-knowledge-lint`，blocker 清零才算落地

## §6 红线

- [ ] 性能数字必须可复现（指回 json 字段路径），禁止估算编造——** reflux 卡的全部价值在于可信**
- [ ] 一概念一 canonical 卡：已有条目只 append，不复制第二份
- [ ] 正文禁摄入/变更口吻（「本次生成 / 该 run」只进 log/ 与「已知实例」行）
- [ ] 不创建孤卡（每卡 ≥2 双向链接）；不擅自改 SPEC-*.md 与各分类生产 skill
- [ ] status: draft + confidence 如实标注；未人工 review 不自称 verified status
- [ ] 失败轨迹不硬凑：无泛化价值的 B 类环境错误不入库

## §7 盒装摘要（强制输出）

```
┌─ 知识回流 [triton-knowledge-reflux] ─────────────────────
│ S1 成功案例: <已产 ops/<op>.md (draft) / 未触发 (<原因>)>
│ S2 调试经验: <已 append <entry> 已知实例 ×N / 新增 <ID> / 未触发>
│ S3 性能洞察: <已 append <entry> ×N / 新增 CT-<n> / 未触发>
│ S4 消费反馈: <已回链 <卡> ×N / 未触发>
│ 质检/图谱: <lint PASS / staging→judge→inject 完成 / verify PASS>
└────────────────────────────────────────────────────
```

## §8 与 op-generator 的接线（两阶段协议）

op-generator 的 AGENTS.md Phase 6 已改为调用本 skill。skill 通过符号链接分发到插件侧（`.claude/skills/triton-knowledge-reflux -> 本目录`），与 triton-knowledge-query 同机制。两阶段协议：

- **阶段 A（生成侧，插件副本执行，必做）**：collect → 蒸馏 → 在 `<工作目录>/reflux/` 产出 draft bundle：
  - `case_card.md` — ops 案例卡草稿（按 STRUCTURE-case-card.md，`resource`/`sources` 用工作目录绝对路径）
  - `appends.md` — 待 append 的已知实例清单（目标条目路径 + 追加行文本 + 证据）
  - `landing_plan.json` — 每个动作的 {action: create_card|append_instance|crosslink, target, evidence}
- **阶段 B（KB 侧落地）**：agent 可直达知识库根目录时按 §5 直接落地（写卡 → index → 图谱 → lint）；不可达或知识库治理要求人工过闸时，bundle 留存并告警，由知识库维护者按 landing_plan.json 落地。

B 知识库根目录解析：`TRITON_ASCEND_KNOWLEDGE_ROOT(S)` / `OKF_KNOWLEDGE_ROOT(S)` / `~/.config/cannbot/triton-knowledge.env`（与 triton-knowledge-query 同约定），不写死本机绝对路径。

### legacy 过渡

op-generator 原 Phase 6 写 legacy `.claude/skills/triton-knowledge-retrieval/knowledge/raw/experiences/`（旧 KKB YAML）。过渡期由 `TRITON_KKB_LEGACY_INGEST=1` 开关控制双写，默认关闭；本 skill 的触发判定自足于工作目录产物，不依赖 F1-F5 事件日志。
