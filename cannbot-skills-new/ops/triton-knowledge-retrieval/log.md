# 知识库反馈事件日志

> 记录所有 F1-F5 反馈事件、INGESTION 摄入事件、MAINTENANCE_REPORT 维护报告。
> 由各 triton-* skill 在执行过程中自动追加，由 `maintenance/periodic_tasks.md` 定期读取分析。

---
event_type: F4
timestamp: 2026-08-05T23:07:00Z
session_id: 3_Add_multidtype_run1
operator: 3_Add
hardware: ascend910b2
cache_result:
  cache_file: knowledge/cache/codegen_init__elementwise_add_float32_float16_bfloat16_broadcasting.md
  match_type: full
  accepted: true
  modification_reason: ""
---

---
event_type: F4
timestamp: 2026-08-05T23:30:00Z
session_id: 3_Add_multidtype_run1
operator: 3_Add
hardware: ascend910b2
cache_result:
  cache_file: knowledge/cache/perf_tuning__VecUtil_low_elemwise_broadcast_dual_path.md
  match_type: partial
  accepted: true
  modification_reason: "语义高度匹配但关键词不完全一致(broadcast vs broadcasting)，直接复用"
---

---

## 事件索引

| 事件ID | 时间 | 类型 | Session | 摘要 |
|--------|------|------|---------|------|
| — | — | — | — | — |

---

## 事件格式参考

### F1 编译修复事件

```yaml
---
event_type: F1
timestamp: YYYY-MM-DDTHH:MM:SS
session_id: <op_name>_runNNN
operator: <op_name>
hardware: <910b2|910b3>
compile_result:
  status: fail | success
  failed_unit_id: <知识单元ID，如 C3>
  error_message: "<错误信息摘要>"
  fix_applied: "<修复方法>"
---
```

### F2 精度验证事件

```yaml
---
event_type: F2
timestamp: YYYY-MM-DDTHH:MM:SS
session_id: <op_name>_runNNN
operator: <op_name>
hardware: <910b2|910b3>
precision_result:
  max_err: <float>
  cosine_sim: <float>
  status: pass | fail
  dtype: <fp16|bf16|fp32>
  failure_unit_ids: [<知识单元ID列表>]
---
```

### F3 性能收敛事件

```yaml
---
event_type: F3
timestamp: YYYY-MM-DDTHH:MM:SS
session_id: <op_name>_runNNN
operator: <op_name>
hardware: <910b2|910b3>
profiling_result:
  gen_time_before: <float>
  gen_time_after: <float>
  speedup_vs_baseline: <float>
  cube_util: <percentage>
  vector_util: <percentage>
  bottleneck_predicted: "<瓶颈类型>"
  bottleneck_actual: "<瓶颈类型>"
  high_impact_units:
    - id: <知识单元ID>
      contribution: <percentage>
  low_impact_units:
    - id: <知识单元ID>
      contribution: 0%
      note: "<原因>"
---
```

### F4 缓存命中事件

```yaml
---
event_type: F4
timestamp: YYYY-MM-DDTHH:MM:SS
session_id: <op_name>_runNNN
cache_result:
  cache_file: <缓存文件名>
  match_type: exact | partial
  accepted: true | false
  modification_reason: "<原因，未修改时为空字符串>"
---
```

### F5 用户纠正事件

```yaml
---
event_type: F5
timestamp: YYYY-MM-DDTHH:MM:SS
session_id: <op_name>_runNNN
correction:
  unit_id: <知识单元ID>
  correction_type: <deprecate|update|conflict>
  reason: "<纠正原因>"
---
```

### INGESTION 摄入事件

```yaml
---
event_type: INGESTION
source_type: source_b | source_c | source_d
timestamp: YYYY-MM-DDTHH:MM:SS
session_id: <op_name>_runNNN
new_experiences:
  - id: <experience_id>
    category: <类目>
    status: draft
    composite_score_initial: <float>
    performance_boost_claimed: <percentage>
    key_technique: "<1-2句描述>"
wiki_distill_triggered: false
_index_updated: true
---
```

### MAINTENANCE_REPORT 维护报告

```yaml
---
event_type: MAINTENANCE_REPORT
timestamp: YYYY-MM-DDTHH:MM:SS
trigger: 第N次算子生成完成（累计）
tasks_executed: [Q, CD, CI, COV]

quality_reeval:
  experiences_updated: <int>
  status_changes:
    draft_to_stable: <int>
    stable_to_review: <int>
  composite_score_avg_change: <float>

conflict_detection:
  conflicts_found: <int>
  hardware_conflicts: <int>
  wiki_divergences: <int>

cache_invalidation:
  caches_checked: <int>
  stale_marked: <int>
  stale_by_age: <int>

coverage_stats:
  total_experiences: <int>
  by_status:
    stable: <int> (<percentage>)
    draft: <int> (<percentage>)
    review: <int> (<percentage>)
    deprecated: <int> (<percentage>)
  blind_spots: []
  high_quality_experiences: <int>
  problem_coverage:
    general_debug_stable: <int>
    general_stability_stable: <int>
    general_perf_stable: <int>
  wiki_distill_pending: <int>

recommendations: []
---
```

---

## 事件记录

（事件按时间顺序追加在下方）

- event_type: INGESTION
  timestamp: 2026-09-14
  session_id: op_94_94_Gdn2Fwd_20260911_1314_5897
  source: source_c
  experience_id: exp_debug_006_20260914
  target: knowledge/raw/experiences/general_debug/exp_debug_006_20260914.yaml
  trigger: F1(fail)→F1(success) 事件对（iter_0 fp32 mask 精度失败 → iter_1 int32 mask + padded store 修复 50/50）

- event_type: INGESTION
  timestamp: 2026-09-14
  session_id: op_94_94_Gdn2Fwd_20260911_1314_5897
  source: source_d
  experience_id: exp_perf_018_20260914
  target: knowledge/raw/experiences/general_perf/exp_perf_018_20260914.yaml
  trigger: bottleneck_predicted（IR 推断 l2norm 标量链）≠ bottleneck_actual（[BK,BV] tile 主链向量运算），rsqrt 实测无提升证伪

- event_type: F3
  timestamp: 2026-09-14
  session_id: op_94_94_Gdn2Fwd_20260911_1314_5897
  op_name: 94_Gdn2Fwd
  bottleneck_predicted: l2norm 标量归一化链（vdiv + S↔V 同步）
  bottleneck_actual: "[BK,BV] tile 主链向量运算（vmul_2d/reduce_sum_ra）"
  speedup_vs_baseline: 1.0
  high_impact_units: []

---
event_type: INGESTION
timestamp: 2026-09-14 11:35:08
session_id: op_94_94_Gdn2Fwd_20260911_1314_5897
source: source_d
file: knowledge/raw/experiences/general_perf/exp_perf_018_20260914.yaml
action: update
summary: 追加 opt_iter_4 simulator 驱动轮证伪：msprof 指令级采集确认 per-token pipe 同步（BAR+SET/WAIT≈56%）为性能地板；点 7 Pass 合并（净指令数不变的代数改写）实测 19.679<19.914 无提升，与 rsqrt 同根因——每向量指令附 ~712cyc BAR，仅减少指令条数的改写才有效

---
event_type: INGESTION
timestamp: 2026-09-14
source: cross-session-archive
experience_id: exp_chunk_gdn2_wy_dispatch_001
action: create
target: knowledge/raw/experiences/ssm/exp_chunk_gdn2_wy_dispatch_001.yaml
summary: 94_Gdn2Fwd 两次独立会话对照实验（serial vs chunked WY）沉淀：serial/chunked 交叉点
  (n_seg·H≲15 且 T≳500)、fla 双 kernel WY 骨架（K1 并行 cumsum→Gram→Neumann 求逆 + K2 串行
  4 dot）、BT=32 编译上限、host 预转置消除 gather、cost-model dispatch、5 条证伪方向。
  同步修订 template/linear-recurrent.md（L1.1 降级为默认骨架 + Layer 2 新增骨架 B）。
---
