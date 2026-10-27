---
# 知识单元质量评分系统

> **用途**：为每条知识单元/经验打四维质量分，驱动检索排序、状态升降级和淘汰决策。
> **关联**：评分由 [knowledge/log.md](../../log.md) 中的反馈事件（F1-F5）自动更新，由 [periodic_tasks](../maintenance/periodic_tasks.md) 定期执行状态迁移。

---

## §1 四维质量评分模型

### 1.1 评分维度定义

每条经验（experience YAML）包含以下质量评分字段：

```yaml
quality_score:
  frequency:     # 维度1: 使用频率（被检索命中并实际采用的比率）
    value: 0     # 0-100，初始为0
    weight: 0.25
    source: F4事件（缓存复用accepted:true）+ F3事件（high_impact_units）
    
  performance:   # 维度2: 实际性能提升（实测均值，%）
    value: 0     # 0-100，初始为声称提升%的50%（保守估计）
    weight: 0.40
    source: F3事件（high_impact_units.contribution）
    
  precision:     # 维度3: 使用后精度通过率（%）
    value: 100   # 0-100，初始假设100%（乐观估计，需反馈验证）
    weight: 0.25
    source: F2事件（precision_result.status）
    
  coverage:      # 维度4: 适用的 shape×dtype 组合覆盖度（%）
    value: 50    # 0-100，根据经验中 conditions 字段的约束范围估算
    weight: 0.10
    source: 人工标注（experience YAML 中 conditions 的宽泛程度）

composite_score: null   # 计算值，不手动填写
# composite = frequency*0.25 + performance*0.40 + precision*0.25 + coverage*0.10
```

### 1.2 初始值设置规则

| 经验来源 | frequency初始值 | performance初始值 | precision初始值 | coverage初始值 |
|---------|---------------|-----------------|----------------|---------------|
| 专家人工录入（Tier 2提炼） | 30（已有背书） | 声称值×0.7（折扣） | 90 | 按条件宽泛度 |
| 算子生成成功案例（Source B） | 10（新录入） | 实测值×0.8 | 100（本次通过） | 20（条件严格） |
| 问题排查记录（Source C） | 10 | 20（调试类通常无提升%） | 100 | 30 |
| Profiling分析（Source D） | 5 | 实测值 | 100 | 20 |

### 1.3 复合分计算示例

```yaml
# 示例：exp_matmul_001（对角tiling）
quality_score:
  frequency:
    value: 85    # 被引用38次，接受35次 = 92%
    weight: 0.25
  performance:
    value: 40    # 声称40%，实测均值40%
    weight: 0.40
  precision:
    value: 100   # 使用后从未触发精度问题
    weight: 0.25
  coverage:
    value: 60    # 适用 M,N≥256 的大多数场景
    weight: 0.10
# composite = 85*0.25 + 40*0.40 + 100*0.25 + 60*0.10 = 62.25
```

---

## §2 知识单元状态机

### 2.1 状态定义

```
draft      新录入，未经反馈验证
  │ composite >= 60 且 被使用 >= 2次
  ▼
stable     经验证，正常使用
  │ composite < 30 且 被使用 >= 5次
  ▼
review     待审查，可能失效
  │ 人工确认问题 或 composite < 15
  ▼
deprecated 已弃用，不再检索命中
```

**状态迁移规则**：

| 当前状态 | 条件 | 新状态 | 触发事件 |
|---------|------|-------|---------|
| draft | composite ≥ 60 且 使用次数 ≥ 2 | stable | F3/F4 |
| draft | 使用次数 ≥ 10 且 composite < 30 | review | F3/F4 |
| stable | composite < 30（连续5次低贡献） | review | F3 low_impact |
| stable | F5 用户纠正 ≥ 3次 | review | F5 |
| draft/stable | periodic_tasks Task 2 检测到 conflict_with 被写入（参数冲突，且本经验为低分方） | review | CD 冲突检测 |
| review | 人工确认有效 | stable | 人工操作 |
| review | 人工确认无效 | deprecated | 人工操作 |
| deprecated | — | — | 不可恢复（需新建经验） |

> **冲突迁移规则说明**：periodic_tasks Task 2（CD）检测到真实参数冲突后，低分方（composite_score 较低的一方）自动迁移到 `review` 状态，高分方保持原状态不变。具体冲突检测规则见 `maintenance/periodic_tasks.md §3 CD1`。

### 2.2 状态字段在 YAML 中的位置

```yaml
# 在每条经验 YAML 文件中新增以下字段：
metadata:
  status: draft       # draft | stable | review | deprecated
  usage_count: 0      # 被检索命中并采用的次数（F3 high_impact + F4 accepted）
  last_used: null     # 最后一次被采用的日期 YYYY-MM-DD
  created: YYYY-MM-DD
  updated: YYYY-MM-DD # 质量分最后更新时间
  quality_score:      # 见 §1.1
    frequency:
      value: 10
      weight: 0.25
    performance:
      value: 35
      weight: 0.40
    precision:
      value: 100
      weight: 0.25
    coverage:
      value: 40
      weight: 0.10
    composite_score: null   # 由系统计算
  feedback_events:          # 关联的反馈事件记录（最近5条）
    - event_type: F3
      session_id: matmul_1024x1024_fp16_run042
      contribution: 25%
      timestamp: 2026-05-05
```

---

## §3 检索时的评分应用

### 3.1 经验匹配排序

在 Stage C Step 4 匹配调优经验时，综合考虑质量分：

```
最终排序分 = 条件匹配分（§2 Step4规则）× 质量权重系数

质量权重系数:
  stable  + composite ≥ 80: 1.3（高质量经验，优先推荐）
  stable  + composite ≥ 60: 1.0（正常权重）
  draft   + composite ≥ 50: 0.8（新经验，略降权）
  review  + 任意:           0.4（待审查，大幅降权）
  deprecated:               0.0（不出现在检索结果中）
```

### 3.2 缓存失效检测

算子缓存（Tier -1）中若包含 deprecated 或 review 状态的知识单元：

```
检测时机: Stage C Step 0 缓存命中时
检测规则:
  若缓存文件中任意 top_units 中的经验状态为 deprecated
    → 将该缓存标记为 stale: true
    → 触发增量更新（仅重新检索 deprecated 经验对应的 Step 4/5/6）
  
  若缓存文件中超过 30% 的经验状态为 review
    → 建议完整重新检索（提示用户）
```

---

## §4 _index.yaml Schema 扩展

在 `raw/experiences/_index.yaml` 中，每条经验新增以下字段：

```yaml
# 现有字段（保留）
- experience_id: exp_matmul_001
  category: matmul
  experience_type: 对角tiling
  operator_type: matmul
  performance_boost: 40%
  hardware: [910b2, 910b3]

  # 新增字段
  status: stable              # draft | stable | review | deprecated
  composite_score: 62.25      # 由系统计算填写
  usage_count: 35             # 被采用次数
  last_used: "2026-05-01"
```

_index.yaml 同时新增文件头部的 schema 说明：

```yaml
# _index.yaml Schema Version 2
# 新增字段说明（相对 v1）:
# - status: 经验状态，draft/stable/review/deprecated，影响检索命中权重
# - composite_score: 四维质量复合分（0-100），由 periodic_tasks 定期更新
# - usage_count: 累计被检索采用次数
# - last_used: 最后一次被采用的日期
#
# 状态迁移规则: 见 retrieval/quality_scoring.md §2
# 评分更新规则: 见 log.md 反馈事件格式
# 定期维护: 见 maintenance/periodic_tasks.md
```

---

## §5 质量覆盖率统计

每完成 10 次算子生成后，统计以下覆盖率指标（由 periodic_tasks 执行）：

| 指标 | 计算方式 | 预警阈值 |
|------|---------|---------|
| stable经验占比 | stable数 / 总数 | < 60% 时预警 |
| deprecated经验占比 | deprecated数 / 总数 | > 20% 时清理 |
| 知识盲区类目 | stable经验数 < 3 的类目列表 | 任意类目 < 3 条 |
| 高质量经验（composite≥80）覆盖率 | composite≥80的类目数 / 总类目数 | < 50% 时预警 |

输出格式（写入 log.md）：
```yaml
---
event_type: COVERAGE_REPORT
timestamp: YYYY-MM-DDTHH:MM:SS
trigger: 第N次算子生成完成
stats:
  total_experiences: 267
  stable: 180 (67%)
  draft: 70 (26%)
  review: 15 (6%)
  deprecated: 2 (1%)
blind_spots:
  - category: reduction
    stable_count: 2
    recommendation: "需补充 reduction 类目经验（当前 stable < 3）"
  - category: conv
    stable_count: 1
    recommendation: "需补充 conv 类目经验"
high_quality_categories:
  - matmul: composite_avg=72
  - attention: composite_avg=68
---
```
