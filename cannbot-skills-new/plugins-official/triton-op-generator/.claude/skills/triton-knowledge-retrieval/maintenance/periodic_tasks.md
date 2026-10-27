---
# 知识库定期维护任务规范

> **用途**：定义知识库的自动化定期维护任务，包括质量重评估、冲突检测、缓存失效检测和覆盖率统计。
> **执行时机**：每完成 10 次算子生成后自动触发（由 Stage I 尾部检查驱动）。
> **关联文档**：
> - 反馈事件格式：[log.md](../log.md)
> - 质量评分规则：[retrieval/quality_scoring.md](../retrieval/quality_scoring.md)
> - 自动构建流水线：[builder/auto_ingestion.md](../builder/auto_ingestion.md)

---

## §1 触发规则

```
触发条件: 每完成第 N×10 次算子生成（N=1,2,3,...）
检查方法: 从 log.md 统计 event_type=F3 的事件数量 mod 10 == 0

紧急触发条件（不等待10次，立即触发）:
  - deprecated 经验占比 > 20%
  - 发现 review 经验数 > 30
  - 任意 F5 用户纠正同一经验 ≥ 3 次
```

---

## §2 Task 1：质量重评估（Quality Re-evaluation）

### 执行步骤

**Step Q1：收集反馈事件**

```
读取 log.md 中上次维护以来的所有事件:
  F1 事件 → 更新 compile_fail_count / compile_success_count
  F2 事件 → 更新 precision_pass_count / precision_fail_count  
  F3 事件 → 更新 high_impact / low_impact 贡献记录
  F4 事件 → 更新 usage_count（accepted:true 时）
  F5 事件 → 更新 user_rejection_count
```

**Step Q2：重新计算 composite_score**

```
对每条 status ≠ deprecated 的经验:

frequency_value = usage_count / max(total_retrievals, 1) × 100
  # total_retrievals 为该类目被检索的总次数（近似为 F3 事件数 × 5）

performance_value = 所有 high_impact F3 事件中 contribution 的平均值
  # 若无 F3 记录，保持初始值

precision_value = precision_pass_count / max(precision_pass_count + precision_fail_count, 1) × 100

coverage_value = 保持当前值（不自动更新，需人工判断）

composite = frequency×0.25 + performance×0.40 + precision×0.25 + coverage×0.10
```

**Step Q3：状态迁移判断**

```
按 quality_scoring.md §2.1 规则执行状态迁移:

draft → stable:  composite ≥ 60 且 usage_count ≥ 2
draft → review:  usage_count ≥ 10 且 composite < 30
stable → review: composite < 30（连续5次 low_impact）或 F5 纠正 ≥ 3次
review → deprecated: 人工标注（本任务只标记 review，不自动 deprecated）
```

**Step Q4：更新 _index.yaml**

```
更新变更的经验条目:
  status: <新状态>
  composite_score: <新分值>
  usage_count: <累计值>
  last_used: <最近F4/F3日期>
```

---

## §3 Task 2：冲突检测（Conflict Detection）

### 检测范围

扫描同类目下经验间的矛盾和歧义：

**检测规则 CD1：参数建议冲突**

```
扫描对象: 同类目（category 相同）+ 同算子类型（operator_type 相同）的经验对

冲突信号:
  - 两条经验对同一参数（如 BLOCK_K）给出不同推荐值
  - 两条经验的 conditions 覆盖范围有重叠，但 key_technique 建议相反
  - 一条经验的 performance_boost 为正值，另一条为负值（避坑）

冲突处理:
  Step 1: 检查两条经验的 conditions 是否真正重叠（shape/dtype/hardware）
    → 若条件不重叠：无冲突，仅添加「条件区分」说明
    → 若条件重叠：标记为真实冲突
  
  Step 2: 对真实冲突:
    → 保留 composite_score 更高的经验（主经验）
    → 在另一条经验添加 conflict_with 字段，标注主经验ID
    → 在 experience_index.md 中插入 [矛盾提示] 知识单元
    → 在 log.md 记录 CONFLICT_DETECTED 事件
```

**检测规则 CD2：与硬件约束冲突**

```
扫描对象: 所有 key_code_pattern 不为空的经验

冲突信号（以下任一触发）:
  - BLOCK_M × BLOCK_N × dtype_bytes 超过 64KB（UB 限制的 1/3）
  - BLOCK_N × dtype_bytes 不是 512 的倍数（行对齐约束）
  - 代码中出现 tl.swizzle2d（禁用 API）
  - 代码中使用 int64 做比较或加法（scalar fallback）

冲突处理:
  → 在该经验添加 hardware_conflict_warning 字段
  → 降低 composite_score（performance 维度 ×0.5）
  → 在 log.md 记录 HARDWARE_CONFLICT 事件
```

**检测规则 CD3：与 Wiki 内容分歧**

```
扫描对象: composite_score ≥ 60 的 stable 经验 vs 对应 Wiki 页面

分歧信号:
  - 经验的 performance_boost 远高于 Wiki 中记录的提升%（差异 > 15%）
  - 经验建议的参数值与 Wiki 推荐值不同

分歧处理:
  → 记录 WIKI_DIVERGENCE 事件到 log.md
  → 加入 wiki_distill_drafts 候选（见 auto_ingestion.md §4）
  → 不自动修改 Wiki，等待人工审核
```

---

## §4 Task 3：缓存失效检测（Cache Invalidation）

### 检测对象

`cache/` 目录下所有缓存文件（各 `{query_type}__*.md`）。

### 检测规则

```
Step CI1: 读取每个 cache/{query_type}__*.md 的 top_units 列表

Step CI2: 在 _index.yaml 中检查每条经验的当前状态:
  若任意 experience 状态为 deprecated
    → 在该缓存文件 frontmatter 添加: stale: true, stale_reason: "contains deprecated experience"
    → 在 log.md 记录 CACHE_STALE 事件

  若超过 30% 的 experience 状态为 review
    → 在该缓存文件 frontmatter 添加: stale: partial, stale_reason: "contains review experiences"
    → 下次命中时提示用户考虑重新检索

Step CI3: 检查缓存时效性（可选）:
  若缓存 created 日期距今 > 90 天，且未被使用
    → 标记为: stale: stale_by_age
    → 不删除，仅标记（等待下次命中时决定是否刷新）
```

---

## §5 Task 4：覆盖率统计（Coverage Statistics）

### 统计维度

```
统计1: 经验状态分布
  总条数 / stable条数 / draft条数 / review条数 / deprecated条数

统计2: 知识盲区识别
  stable经验数 < 3 的类目 → 知识盲区
  输出: 类目名 + 当前 stable 数 + 建议补充方式

统计3: 高质量经验覆盖
  composite_score ≥ 80 的经验数（高质量经验）
  各类目平均 composite_score

统计4: 问题类型覆盖
  general_debug: N条 stable（编译问题覆盖度）
  general_stability: N条 stable（精度问题覆盖度）
  general_perf: N条 stable（性能问题覆盖度）
  各类目专项经验数

统计5: Wiki 同步状态
  有 stable 经验但 Wiki 未更新的类目
  待审核的 wiki_distill_drafts 数量
```

### 输出格式

写入 `log.md`：

```yaml
---
event_type: MAINTENANCE_REPORT
timestamp: YYYY-MM-DDTHH:MM:SS
trigger: 第N次算子生成完成（累计）
tasks_executed: [Q, CD, CI, COV]

quality_reeval:
  experiences_updated: 12
  status_changes:
    draft_to_stable: 3
    stable_to_review: 1
  composite_score_avg_change: +2.3

conflict_detection:
  conflicts_found: 2
  hardware_conflicts: 1
  wiki_divergences: 0

cache_invalidation:
  caches_checked: 5
  stale_marked: 1
  stale_by_age: 0

coverage_stats:
  total_experiences: 267
  by_status:
    stable: 180 (67%)
    draft: 70 (26%)
    review: 15 (6%)
    deprecated: 2 (1%)
  blind_spots:
    - {category: reduction, stable_count: 2, action: "需补充"}
    - {category: conv, stable_count: 1, action: "需补充"}
  high_quality_experiences: 45  # composite >= 80
  problem_coverage:
    general_debug_stable: 5
    general_stability_stable: 4
    general_perf_stable: 7
  wiki_distill_pending: 2       # 待审核的提炼草稿

recommendations:
  - "reduction 类目只有 2 条 stable 经验，建议下次 reduction 算子生成后重点提取"
  - "attention 类目有 2 个待审核 wiki_distill_drafts，建议安排人工审核"
---
```

---

## §6 维护执行检查清单

每次触发维护任务时，LLM 执行以下自检：

```
□ Task 1 质量重评估:
  □ 是否读取了上次维护以来的所有 F1-F5 事件？
  □ 是否重新计算了所有变动经验的 composite_score？
  □ 是否执行了状态迁移（draft→stable / stable→review）？
  □ 是否更新了 _index.yaml 中变动的条目？

□ Task 2 冲突检测:
  □ 是否扫描了新增 stable 经验与同类目经验的参数冲突？
  □ 是否检查了新增经验的 key_code_pattern 是否违反硬件约束？
  □ 冲突经验是否已在 experience_index.md 中标注 [矛盾提示]？

□ Task 3 缓存失效:
  □ 是否检查了所有 cache/ 缓存文件的 top_units 状态？
  □ 含 deprecated 经验的缓存是否标记了 stale: true？

□ Task 4 覆盖率统计:
  □ 是否识别了 stable 经验 < 3 的盲区类目？
  □ 是否在 log.md 追加了 MAINTENANCE_REPORT 事件？
  □ 是否有待审核的 wiki_distill_drafts 需要提醒？
```
