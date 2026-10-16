# builder/ — 知识构建引擎

> **用途**：负责知识库的自动化扩充与 Wiki 更新建议生成。
> **关联**：`auto_ingestion.md`（入库流程）、`maintenance/periodic_tasks.md`（触发时机）

---

## 目录结构

```
builder/
├── README.md                    # 本文件：目录说明 + wiki_distill 工作流
├── auto_ingestion.md            # 知识自动构建流水线（Source B/C/D 提取规则）
└── wiki_distill_drafts/         # Wiki 提炼草稿目录（自动生成，人工审核后合入）
    └── <草稿文件，命名见下方>
```

`wiki_distill_drafts/` 目录不预创建，在首次触发 Wiki Distiller（`auto_ingestion.md §4`）时自动创建。

---

## wiki_distill_drafts/ 工作流

### 文件命名规则

```
wiki_distill_<category>_<YYYYMMDD>.md

示例:
  wiki_distill_matmul_20260510.md
  wiki_distill_normalization_20260510.md
```

若同一类目在同一天触发多次，追加后缀 `_v2`、`_v3`。

### 文件格式

```markdown
---
category: <op_category>
triggered_by: <触发本次提炼的 session_id>
created: YYYY-MM-DDTHH:MM:SS
status: DRAFT            # DRAFT | APPROVED | REJECTED | EXPIRED
reviewed_by: null        # 审核人，APPROVED/REJECTED 时填写
reviewed_at: null        # 审核时间
expires_at: YYYY-MM-DD   # 创建时间 + 30天，超期未审核自动变为 EXPIRED
---

# Wiki Distill Draft: <category>

**触发条件**: <stable经验数> 条 stable 经验，上次 Wiki 更新 <N> 天前，新增 <M> 条 stable 经验

## 候选更新

### U1 更新 pages/ops/<op>.md §常见优化
<具体新增内容，含经验ID引用>

### U4 新增 experience_index.md 条目
<新增行内容>

## 来源经验
- [<经验ID>](<经验文件路径>): <一句话描述>

## 审核说明
> 本草稿需人工审核后方可合入正式 Wiki 页面。
> 审核方式：修改本文件 frontmatter 中的 status 字段：
>   - APPROVED → 将「候选更新」内容手动合入对应 Wiki 页面后删除本草稿文件
>   - REJECTED → 记录拒绝理由后删除本草稿文件（经验仍保留，仅不更新 Wiki）
```

### 审核流程

```
1. Wiki Distiller 生成草稿文件（status: DRAFT）
2. 人工阅读草稿，判断更新建议是否合理
3a. 接受更新：
    → 将「候选更新」内容手动合入对应 Wiki 页面（pages/ops/<op>.md 等）
    → 将草稿文件 status 改为 APPROVED，记录 reviewed_by/reviewed_at
    → 删除草稿文件
3b. 拒绝更新：
    → 将草稿文件 status 改为 REJECTED，记录拒绝原因
    → 删除草稿文件
4. 删除草稿文件后，在 log.md 追加 WIKI_DISTILL_REVIEWED 事件
```

### 超期规则

```
expires_at = created + 30 天

periodic_tasks.md Task 4（覆盖率统计）中会统计待审核草稿数量并报告。
若草稿 status=DRAFT 且当前时间 > expires_at:
  → 将 status 改为 EXPIRED（不删除，仅标记）
  → 在 MAINTENANCE_REPORT 中提醒人工处理或删除

EXPIRED 草稿不影响知识库正常运行，可随时手动删除。
```

### WIKI_DISTILL_REVIEWED 事件格式

```yaml
---
event_type: WIKI_DISTILL_REVIEWED
timestamp: YYYY-MM-DDTHH:MM:SS
category: <op_category>
draft_file: wiki_distill_<category>_<YYYYMMDD>.md
decision: APPROVED | REJECTED | EXPIRED
wiki_pages_updated:       # APPROVED 时填写
  - pages/ops/<op>.md
rejection_reason: null    # REJECTED 时填写
---
```
