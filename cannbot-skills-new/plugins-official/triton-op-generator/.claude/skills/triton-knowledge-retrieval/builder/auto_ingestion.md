# 知识自动构建流水线（Auto Ingestion Pipeline）

> **用途**：从算子生成成功案例、问题排查记录和 Profiling 分析结果中自动提取知识，构建新的经验条目，扩充知识库。
> **目标**：打破"只有专家人工维护知识"的瓶颈，实现知识库的自我扩充能力。

---

## §1 知识来源类型

| 来源ID | 来源类型 | 触发时机 | 写入目标 | ID 格式 |
|-------|---------|---------|---------|---------|
| **Source A** | 专家人工录入（现有） | 人工操作 | `raw/experiences/` | `exp_<cat>_<3位序号>` 如 `exp_matmul_001` |
| **Source B** | 算子生成成功案例 | Stage I 最终交付后 | `raw/experiences/<op_category>/` | `exp_<cat>_<3位序号>_<YYYYMMDD>` 如 `exp_matmul_039_20260505` |
| **Source C** | 问题排查记录 | 编译/精度问题解决后 | `raw/experiences/general_debug/` 或 `general_stability/` | `exp_debug_<3位序号>_<YYYYMMDD>` 如 `exp_debug_012_20260505` |
| **Source D** | Profiling 分析结果 | Stage H 收敛后 | `raw/experiences/general_perf/` 或 `<op_category>/` | `exp_perf_<3位序号>_<YYYYMMDD>` 如 `exp_perf_007_20260505` |

**ID 命名说明**：
- Source A（专家录入）不带日期后缀，序号三位补零（001, 002, ...）
- Source B/C/D（自动提取）统一带 `_<YYYYMMDD>` 后缀，便于追溯生成时间和区分来源
- `<3位序号>` 为同类目下当前最大序号 +1（查询 `_index.yaml` 中该类目最大 seq 得出）
- 写入前先检查 `_index.yaml`，若已存在相同 experience_id 则跳过（防止重复录入）

> Source B/C/D 为新增的自动化来源，全部从实战运行结果中提取，无需人工录入。

---

## §2 Source B：算子生成成功案例提取

### 触发条件

Stage I 最终交付时，同时满足：
1. 精度验证通过（cosine_sim ≥ 0.999，max_err 在容许范围）
2. 至少完成 1 轮 Profiling，speedup_vs_baseline ≥ 1.1x（即相对 PyTorch 基线加速 ≥ 10%）
3. evolution chain 中有至少 1 次有效的调优改进（F3 high_impact_units 非空）

> **speedup 阈值说明**：`speedup_vs_baseline = pytorch_baseline_time / gen_time_after`，取自 F3 事件的 `speedup_vs_baseline` 字段。≥ 1.1x 意味着 Triton kernel 比 PyTorch 基线至少快 10%，低于此阈值说明优化效果不显著，不值得录入。

### 提取步骤

**Step B1：从最优 kernel 提取关键实现模式**

```
提取内容:
  - tiling 配置（BLOCK_M, BLOCK_N, BLOCK_K 及选择依据）
  - 关键 API 使用方式（是否使用 block_ptr / care_padding / async_task 等）
  - 特殊处理模式（对角tiling / persistent kernel / fp32 accumulator 等）
  - constexpr 参数列表及其作用
```

**Step B2：从 evolution chain 提取有效调优历史**

```
提取内容:
  - 哪些基因（知识单元ID）在哪一代产生了显著提升（贡献度 > 10%）
  - 初始版本 vs 收敛版本的关键差异
  - 无效尝试（贡献度 = 0% 的基因，标记为负例）
```

**Step B3：从 profiling 报告提取性能数据**

```
提取内容:
  - 最终 gen_time vs baseline_time → 计算 speedup
  - CubeUtil / VectorUtil 终态值
  - 实际瓶颈（与 Step 4 预测瓶颈对比，记录偏差）
  - 每个关键优化步骤的性能贡献
```

**Step B4：知识单元化**

将提取信息转化为标准经验 YAML 格式：

```yaml
experience_id: exp_<category>_<auto_seq>_<YYYYMMDD>
  # 命名规则: auto_seq 为该类目当前最大序号+1，标记来源为 source_b
category: <op_category>          # 与现有类目保持一致
experience_type: <优化技术名称>
operator_type: <matmul|attention|normalization|...>
hardware: [910b2, 910b3]
dtype: <fp16|bf16|fp32>

# 触发条件（从最优kernel的参数中提取）
conditions:
  shape: "M>=<m_threshold>, N>=<n_threshold>"  # 使条件尽量精确
  dtype: <dtype>
  pattern: "<触发此优化的计算模式>"

# 性能数据（从 profiling 提取，真实值）
performance_boost: <speedup%>    # 相对 PyTorch 基线
cube_util_final: <CubeUtil%>
vector_util_final: <VectorUtil%>

# 核心优化内容（从 Step B1 提取）
key_technique: "<1-2句核心技术描述>"
key_code_pattern: |
  # 从最优kernel中提取的关键代码片段（10-20行以内）
  <关键代码>

# 知识来源追溯
source_type: source_b            # 标记为自动提取
source_session: <session_id>     # 对应 log.md 中的 F3 事件
source_kernel_file: <kernel文件路径>  # 可选，便于溯源
evolution_contribution: <贡献度%>  # 在 evolution 中的贡献度

# 质量评分（初始值，见 quality_scoring.md §1.2）
metadata:
  status: draft
  usage_count: 0
  last_used: null
  created: <YYYY-MM-DD>
  updated: <YYYY-MM-DD>
  quality_score:
    frequency:
      value: 10
      weight: 0.25
    performance:
      value: <speedup%×0.8>    # 折扣系数0.8，保守估计
      weight: 0.40
    precision:
      value: 100
      weight: 0.25
    coverage:
      value: 20                # 新提取经验初始覆盖度较低
      weight: 0.10
    composite_score: null
```

**Step B5：写入和索引更新**

```
写入位置: raw/experiences/<category>/exp_<category>_<seq>.yaml
索引更新: 
  1. 在 raw/experiences/_index.yaml 末尾追加新条目
  2. 若该类目经验 ≥ 5 条 stable → 触发 Wiki Distiller（见 §4）
反馈写入:
  在 log.md 追加 Source B 入库记录（见 §5 日志格式）
```

---

## §3 Source C & D：问题排查记录 / Profiling 分析提取

### Source C：问题排查记录

**触发条件**：本次 session 中存在**至少一组完整的修复事件对**，即：
- 编译修复：同一 session_id 下出现 `F1(status=fail)` 后紧接 `F1(status=success)`
- 精度修复：同一 session_id 下出现 `F2(status=fail)` 后紧接 `F2(status=pass)`

> **事件对计数规则**：一个 session 中无论出现多少次 fail → success 循环（如 fail/fail/success），只计为**一组**修复记录，生成**一条** Source C 经验。不重复录入同一 session 的多次失败尝试。

```yaml
# Source C YAML 模板
experience_id: exp_debug_<seq>_<YYYYMMDD>
category: general_debug    # 或 general_stability（精度问题）
experience_type: <编译修复|精度调试>
operator_type: <涉及的算子类型>

# 问题描述
problem:
  type: compile | precision
  error_message: "<错误信息摘要>"
  trigger_condition: "<触发条件：shape/dtype/API>"

# 解决方案
solution:
  root_cause: "<根本原因1-2句>"
  fix_code_before: |
    <修复前的问题代码片段>
  fix_code_after: |
    <修复后的正确代码片段>
  knowledge_unit_ref: <关联的知识单元ID，如 C3（512B对齐）>

# 来源
source_type: source_c
source_session: <session_id>
source_event: <F1或F2事件ID>

metadata:
  status: draft
  quality_score:
    frequency: {value: 10, weight: 0.25}
    performance: {value: 20, weight: 0.40}  # 调试类主要价值是正确性
    precision: {value: 100, weight: 0.25}
    coverage: {value: 30, weight: 0.10}
    composite_score: null
```

### Source D：Profiling 分析提取

**触发条件**：Stage H 收敛后，F3 事件中满足以下任一条件：
1. `bottleneck_actual` 与 `bottleneck_predicted` 的**瓶颈类型完全不同**（如预测 CubeUtil_low 实测 MTE2_high）
2. 同一瓶颈类型下，实测的利用率数值偏差 **> 20个百分点**（如预测 CubeUtil=60% 实测 CubeUtil=85%）

> **偏差计算说明**：以利用率绝对值差判断（|predicted_util - actual_util| > 20%），而非相对差。这意味着知识库对瓶颈预测出现明显错误时才录入，避免噪声数据。

```yaml
# Source D YAML 模板
experience_id: exp_perf_<seq>_<YYYYMMDD>
category: general_perf    # 或具体的 op_category
experience_type: profiling分析

# 瓶颈发现
bottleneck_finding:
  predicted: "<Stage C 预测的瓶颈>"
  actual: "<Profiling 实测的真实瓶颈>"
  deviation: "<偏差描述>"

# 修复方法
fix_applied: "<采用的修复方法>"
performance_delta: "<修复后的性能变化%>"

# 适用条件（从本次 session 提取）
conditions:
  shape: "<触发此瓶颈的 shape 范围>"
  dtype: "<dtype>"
  hardware: [910b2, 910b3]

source_type: source_d
source_session: <session_id>

metadata:
  status: draft
  quality_score:
    frequency: {value: 5, weight: 0.25}
    performance: {value: <实测值>, weight: 0.40}
    precision: {value: 100, weight: 0.25}
    coverage: {value: 20, weight: 0.10}
    composite_score: null
```

---

## §4 Wiki 提炼器（Wiki Distiller）

将大量零散 experiences 提炼为 Tier 0 Wiki 页面的更新建议。

### 触发条件

```
同时满足以下全部条件:
  1. 某类目下 stable 状态的经验数量 ≥ 5 条
  2. 距上次该类目 Wiki 更新时间 > 30 天
  3. 新增 stable 经验数量 ≥ 2 条（上次更新后）
```

### 提炼流程

**Step W1：收集候选经验**

```
读取该类目所有 status=stable 的经验
筛选: composite_score ≥ 60 且 usage_count ≥ 2
排序: 按 composite_score 降序
```

**Step W2：提取共性模式**

```
分析所有候选经验，识别:
  1. 重复出现的优化技术（出现 ≥ 3 次）→ 候选"常见优化"小节
  2. 新发现的瓶颈类型（现有 Wiki 无对应段落）→ 候选新瓶颈说明
  3. 新的 API 使用模式（现有 cheatsheet 未收录）→ 候选 API 补充
  4. 跨算子通用的新技术（适用 ≥ 2 种算子类型）→ 候选新 techniques 页
```

**Step W3：生成更新建议（草稿）**

```
更新类型 U1: 更新 pages/ops/<op>.md 的"常见优化"小节
  → 补充 ≥ 3 次出现的优化技术，附经验ID列表

更新类型 U2: 更新 pages/techniques/<t>.md
  → 补充新发现的适用场景或参数范围

更新类型 U3: 新建 pages/techniques/<new_technique>.md
  → 仅当发现完全新的、现有wiki无对应页面的技术模式

更新类型 U4: 更新 pages/experience_index.md
  → 将新 stable 经验加入快速匹配区
```

**Step W4：写入草稿，等待人工确认**

```
草稿写入位置: builder/wiki_distill_drafts/
文件命名: wiki_distill_<category>_<YYYYMMDD>.md

草稿格式:
  # Wiki Distill Draft: <category>
  触发条件: <stable经验数> 条 stable 经验，上次更新 <N> 天前
  候选更新:
    U1 更新 ops/<op>.md §常见优化:
      <具体新增内容>
    U4 新增 experience_index.md 条目:
      <新增行>
  来源经验: [<经验ID列表>]
  状态: DRAFT（需人工审核后方可合入）
```

> **注意**：Wiki 提炼为**建议型更新**，需人工确认后才能合入正式 Wiki 页面，防止低质量内容污染主知识库。

---

## §5 入库日志格式

每次 Source B/C/D 完成入库后，在 `log.md` 追加以下记录：

```yaml
---
event_type: INGESTION
source_type: source_b | source_c | source_d
timestamp: YYYY-MM-DDTHH:MM:SS
session_id: <对应的算子生成session>
new_experiences:
  - id: exp_matmul_039_20260505
    category: matmul
    status: draft
    composite_score_initial: 46.0
    performance_boost_claimed: 28%
    key_technique: "BLOCK_K=128 对 K<512 的小矩阵显著提升"
wiki_distill_triggered: false   # 是否触发了 Wiki 提炼
_index_updated: true
---
```

---

## §6 自动入库执行检查清单

每次 Stage I 后执行以下检查（LLM 自检）：

```
□ 是否有性能提升 > 15% 的优化技术，且现有经验库中无类似条目？
  → 是: 触发 Source B 提取
  
□ 本次生成是否修复了编译错误或精度问题？
  → 是: 触发 Source C 提取

□ Profiling 实测瓶颈与 Stage C 预测是否有 ≥ 2 个维度的偏差？
  → 是: 触发 Source D 提取

□ 提取的经验 ID 是否已在 _index.yaml 中存在？
  → 是: 跳过（不重复录入）
  → 否: 写入新条目

□ 写入后是否更新了 _index.yaml？
  → 必须更新

□ 是否在 log.md 追加了 INGESTION 事件？
  → 必须追加
```
