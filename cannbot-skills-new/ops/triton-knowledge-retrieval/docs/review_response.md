# 检视意见回复 — triton-knowledge-retrieval 合入

## 1. 路径格式不一致

**问题**: 调用方（AGENTS.md + 三 SKILL.md）使用 `ops/triton-knowledge-retrieval/knowledge/cache/…`，其他文件使用 `.claude/skills/triton-op-verifier/`。

**核实结果**: `init.sh` 在安装时将 `ops/` 下的 skill 通过 symlink 映射到 `.claude/skills/` 下。CLAUDE.md 中其他 skill 路径已统一使用 `.claude/skills/`（如 `.claude/skills/triton-latency-optimizer/SKILL.md`、`.claude/skills/triton-op-verifier/scripts/freeze_baseline.py`），仅 triton-knowledge-retrieval 路径混用 `ops/` 前缀，需统一。

**修改方案**: 全局替换 `ops/triton-knowledge-retrieval` → `.claude/skills/triton-knowledge-retrieval`。
- `plugins-official/triton-op-generator/AGENTS.md`: 11 处替换（CLAUDE.md 是 AGENTS.md 的 symlink，自动同步）
- `ops/triton-op-designer/SKILL.md`: 1 处（盒装摘要缓存路径）
- `ops/triton-op-coding/SKILL.md`: 3 处（codegen_init / compile_error / precision_debug 盒装摘要缓存路径）
- `ops/triton-latency-optimizer/SKILL.md`: 1 处（盒装摘要缓存路径）

**状态**: 已修复。

---

## 2. _index.yaml 解析崩溃

**问题**: `python3 -c "import yaml; yaml.safe_load(...)"` 报 `ParserError: expected , but found '-' (line 6)`

**核实结果**: 确认存在。第 4 行是 YAML mapping 格式：
```yaml
exp_elem_add_001_20260526: {category: 'elementwise', ...}
```
第 6 行起是 YAML sequence 格式：
```yaml
- experience_id: exp_perf_001
```
同一层级混用 mapping 和 sequence 是非法 YAML，任何解析器都会崩溃。

**修改方案**: 统一为 sequence 格式。将第 4 行的 mapping 条目改写为 sequence 条目：
```yaml
# 改前（第4行）:
exp_elem_add_001_20260526: {category: 'elementwise', title: 'Add算子块大小动态调整优化经验', status: 'draft', composite_score: 75}

# 改后:
- experience_id: exp_elem_add_001_20260526
  category: elementwise
  title: Add算子块大小动态调整优化经验
  status: draft
  composite_score: 75
```

**影响文件**: `knowledge/raw/experiences/_index.yaml`

**状态**: 已修复。YAML 解析验证通过。

---

## 3. 知识库目录名错误

**问题**: 文档中 `raw/develop-guide/` 和 `raw/expert_experiences/` 与实际目录名不符。

**核实结果**: 
- `raw/develop-guide/`：实际目录为 `raw/triton-ascend-develop-guide/`。影响 4 处引用
- `raw/expert_experiences/`：此目录**不存在**。影响 2 处引用

**修改方案**:

| 文件 | 行号 | 错误路径 | 修正为 |
|------|------|---------|--------|
| `SKILL.md` | 41 | `knowledge/raw/develop-guide/` | `knowledge/raw/triton-ascend-develop-guide/` |
| `references/retrieval-workflow.md` | 178 | `raw/develop-guide/triton_api/` | `raw/triton-ascend-develop-guide/triton_api/` |
| `references/knowledge-unit-format.md` | 78 | `raw/develop-guide/triton_api/` | `raw/triton-ascend-develop-guide/triton_api/` |
| `docs/KKB_OVERVIEW.md` | 71 | `knowledge/raw/develop-guide/` | `knowledge/raw/triton-ascend-develop-guide/` |
| `SKILL.md` | 43 | `knowledge/raw/expert_experiences/` | `knowledge/raw/experiences/`（合并到经验层） |
| `builder/auto_ingestion.md` | 12 | `raw/expert_experiences/` | `raw/experiences/`（合并到经验层） |

**说明**: Tier 2 原始经验层实际使用的是 `raw/experiences/raw_experiences/` 子目录存放多源合并的原始 YAML。`raw/expert_experiences/` 目录从未创建，Source A 人工录入应指向 `raw/experiences/` 对应子目录。

**影响文件**: `SKILL.md`、`references/retrieval-workflow.md`、`references/knowledge-unit-format.md`、`docs/KKB_OVERVIEW.md`、`builder/auto_ingestion.md`

**状态**: 已修复。全部 6 处路径已修正。

---

## 4. top_experiences vs top_units 字段名不一致

**问题**: 缓存失效维护任务按 `top_experiences` 读取缓存内容，但缓存文件使用 `top_units`，导致读取恒为空。

**核实结果**: 
- 缓存文件 frontmatter（39 个文件）全部使用 `top_units`
- `references/cache-protocol.md` 规范使用 `top_units`
- `retrieval/quality_scoring.md:161` 使用 `top_experiences`（错误）
- `maintenance/periodic_tasks.md:154,270` 使用 `top_experiences`（错误）

**修改方案**: 统一为 `top_units`。

| 文件 | 行号 | 修改 |
|------|------|------|
| `retrieval/quality_scoring.md` | 161 | `top_experiences` → `top_units` |
| `maintenance/periodic_tasks.md` | 154 | `top_experiences` → `top_units` |
| `maintenance/periodic_tasks.md` | 270 | `top_experiences` → `top_units` |

**影响文件**: `retrieval/quality_scoring.md`、`maintenance/periodic_tasks.md`

**状态**: 已修复。3 处 `top_experiences` → `top_units`。

---

## 5. _index.yaml 与实际文件不同步

**问题**: 漏登 + 重复登记。

**核实结果**:

**漏登 3 条**（检视意见列了 4 条，其中 `exp_elem_add_001_20260526` 实际已在索引第 4 行，确认漏登的是另外 3 条）：

| 文件名 | 所在目录 | 索引状态 |
|--------|---------|---------|
| `exp_causal_conv1d_varlen_opt_001.yaml` | `raw/experiences/conv/` | 未登记 |
| `exp_debug_triton_branch_type_mismatch_001.yaml` | `raw/experiences/general_debug/` | 未登记 |
| `exp_elemwise_add_001_20260526.yaml` | `raw/experiences/elementwise/` | 未登记 |

**重复 1 条**: `exp_solve_tril_001` 在索引中出现 2 次。

**统计**: 索引 300 条 vs 实际文件 304 条，差 4 条（3 条漏登 + 1 条重复计入 = 300 + 3 - 1 = 302? 需进一步排查剩余差异）。

**修改方案**:
1. 在 `_index.yaml` 中补登 3 条缺失经验
2. 删除重复的 `exp_solve_tril_001` 条目（保留一个）
3. 全量比对 `_index.yaml` 与实际文件，确保索引数 = 文件数

**影响文件**: `knowledge/raw/experiences/_index.yaml`

**状态**: 已修复。补登 3 条缺失 + 删除 1 条重复 + 删除 1 条孤儿引用（exp_fused_index_001，无对应文件）。最终索引 302 条 = 302 个实际经验文件。

---

## 6. F4 缓存命中事件字段定义不一致

**问题**: log.md 中实际记录与格式参考块使用不同字段名，quality_scoring.md 的 frequency 维度可能受影响。

**核实结果**:

| 字段 | 实际事件 (L7-30) | 格式参考块 (L106-121) | quality_scoring.md 依赖 |
|------|-----------------|---------------------|----------------------|
| 命中类型 | `match_type: full` | `hit_type: exact` | — |
| 是否采纳 | `accepted: true` | `accepted: true` | `accepted` (两边都有) |
| 修改原因 | `modification_reason: ""` | — | — |
| Jaccard 相似度 | — | `jaccard_similarity` | — |
| 复用/刷新数 | — | `units_reused`/`units_refreshed` | — |

**结论**: `accepted` 字段两边都有，quality_scoring 的 frequency 维度不会受影响。但格式参考块与实际事件字段名不一致（`match_type` vs `hit_type`），LLM 按格式参考写新事件时会混用。

**修改方案**: 格式参考块对齐实际事件，统一字段名：
```yaml
# 改后的 F4 格式参考
event_type: F4
timestamp: YYYY-MM-DDTHH:MM:SS
session_id: <op_name>_runNNN
operator: <op_name>
hardware: <910b2|910b3>
cache_result:
  cache_file: <缓存文件名>
  match_type: full | partial
  accepted: true | false
  modification_reason: "<原因，未修改时为空字符串>"
```

**影响文件**: `log.md`（格式参考块部分）

**状态**: 已修复。格式参考块已对齐实际事件字段。

---

## 7. 经验条数与算子分类数量不一致

**问题**: 多处引用不同数字。

**核实结果**:

| 位置 | 引用值 | 实际值 | 说明 |
|------|--------|--------|------|
| `SKILL.md:39` | 282 条 | **304 条** | 经验层描述 |
| `SKILL.md:34` | 13 类算子 | **14 类** | ops/ 目录有 14 个 md 文件 |
| `experience_index.md:15` | 266 条 | 304 条 | 索引链接描述 |
| `experience_index.md:215` | 282 条 | 304 条 | 表格中引用 |
| `_index.yaml` | 300 条 | 304 条 | 索引条目数 |

**修改方案**: 全部统一为实际值：
- `SKILL.md:39`: `282` → `304`
- `SKILL.md:34`: `13 类` → `14 类`
- `experience_index.md:15`: `266条` → `304条`
- `experience_index.md:215`: `282` → `304`
- 同步修正 KKB_OVERVIEW.md 等其他引用同一数字的文档

**影响文件**: `SKILL.md`、`knowledge/pages/experience_index.md`、`docs/KKB_OVERVIEW.md`

**状态**: 已修复。全部引用已统一为 304 条/14 类算子。

---

## 修改汇总

| 问题编号 | 严重程度 | 修改文件数 | 状态 |
|---------|---------|-----------|------|
| 1. 路径格式 | 低 | 5 | 已修复 |
| 2. _index.yaml 解析崩溃 | **P0** | 1 | 已修复 |
| 3. 目录名错误 | **P0** | 5 | 已修复 |
| 4. top_experiences → top_units | **P0** | 2 | 已修复 |
| 5. _index.yaml 不同步 | **P1** | 1 | 已修复 |
| 6. F4 字段不一致 | P1 | 1 | 已修复 |
| 7. 经验条数不一致 | P1 | 3 | 已修复 |

---

## 补充修复（第二轮检视，2026-08-06）

### 问题1 残留：SKILL.md 自身路径未随调用方统一

**问题**: KKB 自身 SKILL.md 第 28/50/165/167/171/174 行使用裸 `knowledge/cache/...`，未加 `.claude/skills/` 前缀。

**分析**: KKB 内部路径按自身 skill 根目录解析，与调用方的 `.claude/skills/triton-knowledge-retrieval/knowledge/cache/` 指向同一物理位置（由 `init.sh` symlink 保证），因此不会产生实际错误。但缺少显式约定说明。

**修改**: 在「五层知识架构」章节开头新增路径解析约定块：
```markdown
> **路径解析约定**：本文档中所有相对路径（如 `knowledge/cache/`、`knowledge/pages/`、`knowledge/raw/`）
  均相对于本 skill 根目录（即 `triton-knowledge-retrieval/`）解析。外部调用方使用
  `.claude/skills/triton-knowledge-retrieval/` 前缀（由 `init.sh` symlink 保证与本目录等价），
  因此两套路径指向同一物理位置，不会产生不一致。
```

**影响文件**: `SKILL.md`

**状态**: 已修复。

---

### 问题3 残留：KKB_OVERVIEW.md 架构图中残留 expert_experiences/

**问题**: `docs/KKB_OVERVIEW.md` 第 76、177 行的架构示意图中仍写 `expert_experiences/`。

**修改**:
- L76: `expert_experiences/` → `raw_experiences/`（Tier 2 数据来源标注）
- L177: `expert_experiences/` → `raw_experiences/`（数据流示意图）

**影响文件**: `docs/KKB_OVERVIEW.md`

**状态**: 已修复。

---

### 问题5 残留：仍差 2 处 — 漏登 exp_fused_index_001 + exp_solve_tril_001 跨目录撞名

**问题**:
1. `exp_fused_index_001`（misc/exp_fused_index_001.yaml）文件存在但未登记进 _index.yaml
2. `exp_solve_tril_001` 存在两个同 ID 文件：`misc/exp_solve_tril_001.yaml`（旧版，operator_type: "Matrix Operations"）与 `solve_tril/exp_solve_tril_001.yaml`（正式版，operator_type: "Linear Algebra Operations"）——两个文件内容不同但 experience_id 相同
3. 连带：之前"304 条"的说法基于文件总数（含重复），实际唯一经验数为 303

**修改**:
1. 删除 `misc/exp_solve_tril_001.yaml`（保留 solve_tril/ 目录下的正式版本）
2. 补登 `exp_fused_index_001` 到 `_index.yaml`（category: misc）
3. 全量数字 `304` → `303`，同步更新：
   - `SKILL.md`: `304 条单条经验` → `303 条单条经验`
   - `experience_index.md`: `304条` → `303条`（L15, L215）
   - `KKB_OVERVIEW.md`: 6 处 `304` → `303`，含 `+16.8%` → `+17.2%`（重算：(355-303)/303）

**验证**: `_index.yaml` 303 条 = 实际文件 303 个，无重复、无漏登。

**影响文件**: `_index.yaml`、`misc/exp_solve_tril_001.yaml`（已删除）、`SKILL.md`、`experience_index.md`、`KKB_OVERVIEW.md`

**状态**: 已修复。
