# 知识库快速接入指南

> 适用于：任何需要接入本知识库的算子生成 SKILL
> 在 cannbot 中，本知识库位于 `ops/triton-knowledge-retrieval/`，已通过 triton-knowledge-retrieval skill 提供标准查询接口。
> 本文档描述知识检索协议和反馈事件规范，供 cannbot 子 Agent 和各 triton-* skill 参考。

---

## 概述

本知识库（`triton-ascend-kkb/`）是一个独立的、可插拔的算子优化知识系统，通过标准查询接口向任意 SKILL 提供：

- **硬件约束知识**（对齐、UB容量、Grid限制等）
- **优化经验知识**（调优策略、代码骨架、潜在瓶颈）
- **调试知识**（编译错误修复、精度问题定位）
- **自动缓存与演化**（每次检索结果自动保存，知识库随使用持续增长）

接入只需 3 步：**放置知识库 → 配置路径前缀 → 在工作流关键节点注入检索 prompt**。

---

## Step 1：知识库位置

在 cannbot 中，知识库已内嵌于 `ops/triton-knowledge-retrieval/`，目录结构如下：

```
ops/triton-knowledge-retrieval/
├── SKILL.md                  # Skill 入口 + 七步检索流程
├── log.md                    # 反馈事件日志
├── references/               # 检索协议文档
│   ├── retrieval-workflow.md
│   ├── cache-protocol.md
│   └── knowledge-unit-format.md
├── knowledge/
│   ├── cache/                # 查询缓存（自动管理）
│   ├── pages/                # 结构化 Wiki
│   └── raw/
│       ├── experiences/      # 结构化经验 YAML
│       └── triton-ascend-develop-guide/
├── retrieval/                # 检索引擎
│   ├── quality_scoring.md
│   └── problem_decomposer.md
├── builder/                  # 知识摄入引擎
│   └── auto_ingestion.md
└── maintenance/              # 定期维护
    ├── periodic_tasks.md
    └── integration_quickstart.md
```

> **路径前缀**：在 cannbot 内，知识库根路径为 `ops/triton-knowledge-retrieval/`。

---

## Step 2：在生成开始前执行检索

在你的 SKILL 工作流中，**生成代码之前**加入以下检索步骤：

### 2a. 缓存命中检查（必须最先执行）

```
读 `knowledge/cache/README.md`，按以下规则查找缓存：
- 精确匹配：query_type 相同 AND query_keywords 完全一致 → 直接复用，写 F4 事件
- 相似匹配：query_keywords 重叠度 ≥ 60% → 部分复用 + 增量检索
- 无匹配：执行下方完整检索
```

### 2b. 发送 codegen_init 检索请求

```yaml
query_type: codegen_init
query_keywords: [<算子类型关键词>, <dtype>, <主要组件关键词...>]
context:
  problem: "为 <算子描述> 生成高性能 kernel"
  trigger: "代码生成初始化，从零构建算子知识包"
  scope: "<算子类型+形状+dtype>"
```

按 `SKILL.md` 中定义的七步检索流程执行完整检索，得到**知识单元列表**（C/OP/A/B/E/T/S/X 系列）。

### 2c. 保存检索结果（强制）

```
将知识单元列表写入：
  <知识库路径>/cache/codegen_init__<keyword1>_<keyword2>_....md

更新：
  <知识库路径>/cache/README.md §5 §6（加索引行 + 相似度条目）
```

---

## Step 3：在子 Agent prompt 中注入知识接入协议

将以下模板复制到每个子 Agent（执行具体代码生成的 Agent）的 prompt 末尾：

```
## 知识接入协议（必须遵守）

在本路线执行过程中，遇到以下情况时，主动向知识库发送检索请求并等待结果：

1. 编译失败（任意步骤）：
   → 发送 compile_error 检索请求（见 SKILL.md §检索模式 §compile_error）
   → 检索完成后再修复代码（禁止在未检索的情况下盲目修改）
   → 写入 F1(fail) 事件到 log.md
   → 修复成功后写 F1(success) 事件

2. 精度验证失败（max_err超标 / cosine_sim<0.999 / NaN / inf）：
   → 发送 precision_debug 检索请求（见 SKILL.md §检索模式 §precision_debug）
   → 检索完成后再修复代码
   → 写入 F2(fail) 事件，精度通过后写 F2(pass) 事件

3. Profiling 瓶颈分析完成后（每轮迭代）：
   → 发送 perf_tuning 检索请求（见 SKILL.md §检索模式 §perf_tuning）
   → Top-3 经验追加为 P2 级知识单元（不替换原有 P0/P1 单元）

4. 所有内部迭代完成后：
   → 写入 F3 Profiling收敛事件（含 high_impact_units 和 low_impact_units）

5. 用户否定某知识单元时：
   → 写入 F5 用户纠正事件

所有事件追加写入：log.md（不覆盖已有内容）

检索请求格式：
  query_type: compile_error | precision_debug | perf_tuning
  query_keywords: [<关键词列表>]
  context:
    problem: "<一句话描述本次检索要解决的具体问题>"
    trigger: "<触发上下文>"
    scope: "<检索范围>"
```

---

## Step 4（可选）：生成完成后执行知识摄入

每次成功完成算子生成后，检查以下摄入条件（详见 `builder/auto_ingestion.md`）：

| 条件 | 摄入操作 |
|------|---------|
| 精度通过 + speedup ≥ 1.1x + 完成≥1轮有效 profiling | **Source B**：提取优化模式 → `raw/experiences/<category>/` |
| 存在 F1(fail→success) 或 F2(fail→pass) 事件对 | **Source C**：提取调试记录 → `raw/experiences/general_debug/` |
| F3.bottleneck_actual 与 predicted 偏差 > 20% | **Source D**：提取瓶颈分析 → `raw/experiences/general_perf/` |

摄入完成后写入 INGESTION 事件到 `log.md`。

> 每累计 10 次 F3 事件后，触发 `maintenance/periodic_tasks.md` 定期维护任务（质量重评 + 冲突检测 + 缓存失效 + 覆盖统计）。

---

## 接入核对清单

### 生成前
- [ ] 按 `cache/README.md §4` 查找缓存文件（命中→写F4事件）
- [ ] 无缓存时执行 `SKILL.md` 七步检索流程
- [ ] 检索结果保存到 `cache/codegen_init__<keywords>.md`
- [ ] 更新 `cache/README.md §5 §6`

### 生成中（子Agent）
- [ ] 编译失败 → `compile_error` 检索 → F1(fail) → 修复 → F1(success)
- [ ] 精度失败 → `precision_debug` 检索 → F2(fail) → 修复 → F2(pass)
- [ ] 瓶颈分析后 → `perf_tuning` 检索 → 追加 P2 知识单元
- [ ] **禁止未检索就盲目修改代码**

### 生成后
- [ ] 写入 F3 事件（含 high/low_impact_units）
- [ ] 检查 Source B/C/D 触发条件并执行摄入
- [ ] 写入 INGESTION 事件

### 定期
- [ ] F3 count mod 10 == 0 → 触发 `maintenance/periodic_tasks.md`
- [ ] deprecated > 20% / review > 30 / F5 同单元 ≥ 3 → 紧急维护

---

## 端到端示例：split_qkv_rmsnorm_rope 算子生成全流程

以下是一次完整的接入示例，演示从检索到摄入的全链路执行过程。

### 场景设定

- **算子**：split_qkv + rmsnorm + rope 融合算子
- **规格**：M=4096, H=32, D=128, dtype=fp16
- **知识库路径**：`ops/triton-knowledge-retrieval`（cannbot 内嵌）

---

### 阶段 1：生成前检索（codegen_init）

**Step 0：缓存命中检查**

读 `cache/README.md §4`，扫描 `cache/` 目录：

```
检索关键词: [split_qkv, rmsnorm, rope, fp16]

发现缓存文件: codegen_init__matmul_rmsnorm_fp16.md
  query_keywords: [matmul, rmsnorm, fp16]

计算 Jaccard 相似度:
  A = {split_qkv, rmsnorm, rope, fp16}
  B = {matmul, rmsnorm, fp16}
  A ∩ B = {rmsnorm, fp16}  → |A ∩ B| = 2
  A ∪ B = {split_qkv, rmsnorm, rope, fp16, matmul}  → |A ∪ B| = 5
  Jaccard = 2/5 = 0.40 < 0.60

结论：无命中，执行完整检索
```

**Step 1-6：执行 SKILL.md 七步检索流程**

```
Step 1 约束预检 → 输出 C1/C2/C3/C4/C5 硬件约束知识单元
Step 2 算子分类 → 识别为 attention+normalization+rope 融合
         输出 OP 知识单元：split_qkv（attention类）+ rmsnorm（normalization类）+ rope（rope类）
Step 3 API 知识单元 → 输出 A1(block_ptr), A2(tl.dot), A3(tl.math.rsqrt)
Step 4 瓶颈预判 → 预判主瓶颈：VectorUtil_low（rope + rmsnorm 向量密集）
         输出 B1(向量计算瓶颈), E1(exp_rope_003 RoPE向量化经验, composite_score=71)
Step 5 调优策略 → 输出 T1(BLOCK_N=64 fp16对齐策略), T2(fp32 accumulator策略)
Step 6 代码骨架 → 输出 S1(split_qkv+rmsnorm+rope 融合骨架)
```

**保存缓存文件**

```
写入: cache/codegen_init__split_qkv_rmsnorm_rope_fp16.md
更新: cache/README.md §5 §6（新增索引行）
```

---

### 阶段 2：kernel 生成（子 Agent 执行）

子 Agent 收到知识单元列表 + 标准注入 prompt，开始生成代码。

**事件 1：编译失败（F1 fail）**

第一版 kernel 编译失败：

```
错误：AlignmentError: BLOCK_N=48, fp16 → 48×2=96 bytes，不是512的倍数
```

触发 `compile_error` 检索：

```yaml
query_type: compile_error
query_keywords: [AlignmentError, BLOCK_N, fp16, alignment]
context:
  problem: "编译失败，AlignmentError：BLOCK_N=48 × 2bytes = 96B，不满足512B对齐"
  trigger: "编译失败 — AlignmentError: BLOCK_N=48 not 512B-aligned"
  scope: "编译错误：AlignmentError"
```

检索结果：命中 C3 约束单元（BLOCK_N × dtype_bytes 必须是512的倍数）

修复：BLOCK_N=48 → BLOCK_N=128（128×2=256B，不满足） → BLOCK_N=256（256×2=512B，满足）

写入 F1(fail) 事件，修复后写 F1(success) 事件：

```yaml
---
event_type: F1
timestamp: 2026-05-05T14:23:11
session_id: split_qkv_rmsnorm_rope_fp16_M4096_run001
operator: split_qkv_rmsnorm_rope
hardware: 910b3
compile_result:
  status: success
  failed_unit_id: C3
  error_message: "BLOCK_N=48 不满足512B对齐要求，改为BLOCK_N=256后编译通过"
  fix_applied: "BLOCK_N: 48 → 256"
---
```

**事件 2：精度通过（F2 pass）**

精度验证：max_err=2.1e-4, cosine_sim=0.9998（通过）

```yaml
---
event_type: F2
timestamp: 2026-05-05T14:31:05
session_id: split_qkv_rmsnorm_rope_fp16_M4096_run001
operator: split_qkv_rmsnorm_rope
hardware: 910b3
precision_result:
  max_err: 2.1e-4
  cosine_sim: 0.9998
  status: pass
  dtype: fp16
  failure_unit_ids: []
---
```

---

### 阶段 3：Profiling 迭代

**第 1 轮 Profiling**

```
测量结果: gen_time=0.82ms, pytorch_baseline=1.10ms
speedup_vs_baseline = 1.10/0.82 = 1.34x
CubeUtil=28%, VectorUtil=71%
实测瓶颈: CubeUtil_low（与预判 VectorUtil_low 不同）
```

触发 `perf_tuning` 检索：

```yaml
query_type: perf_tuning
query_keywords: [CubeUtil_low, split_qkv, matmul, tiling]
context:
  problem: "性能瓶颈：CubeUtil仅28%，矩阵乘计算效率偏低"
  trigger: "瓶颈分析 — bottleneck=CubeUtil_low，CubeUtil=28%/VectorUtil=71%"
  scope: "瓶颈类别：CubeUtil_low"
```

检索路径：→ `pages/techniques/core_mapping.md`（对角 tiling）

命中经验 exp_matmul_001（对角tiling，composite_score=62，status=stable）

应用对角 tiling 优化，第 2 轮 gen_time=0.61ms，CubeUtil 提升至 68%。

---

### 阶段 4：迭代收敛，写入 F3 事件

所有优化迭代完成后写入 F3：

```yaml
---
event_type: F3
timestamp: 2026-05-05T15:44:20
session_id: split_qkv_rmsnorm_rope_fp16_M4096_run001
operator: split_qkv_rmsnorm_rope
hardware: 910b3
profiling_result:
  gen_time_before: 0.82
  gen_time_after: 0.61
  speedup_vs_baseline: 1.80
  cube_util: 68%
  vector_util: 74%
  bottleneck_predicted: "VectorUtil_low"
  bottleneck_actual: "CubeUtil_low"
  high_impact_units:
    - id: exp_matmul_001
      contribution: 77%    # (0.82-0.61)/(0.82-0.61) * 100%
  low_impact_units:
    - id: E1
      contribution: 0%
      note: "预判 VectorUtil 为瓶颈，实测 CubeUtil 才是主要瓶颈，该单元未发挥作用"
---
```

---

### 阶段 5：生成后知识摄入检查

| 检查项 | 结果 | 操作 |
|-------|------|------|
| cosine_sim ≥ 0.999 AND speedup ≥ 1.1x AND 有效 profiling ≥ 1 轮 | ✓（sim=0.9998, speedup=1.80x） | **触发 Source B**：提取对角 tiling 经验 → `raw/experiences/matmul/` |
| 存在 F1(fail→success) 事件对 | ✓（AlignmentError 修复） | **触发 Source C**：提取对齐修复调试记录 → `raw/experiences/general_debug/` |
| 实测瓶颈与预判偏差 > 20% | ✓（VectorUtil_low vs CubeUtil_low，类型完全不同） | **触发 Source D**：提取瓶颈预判偏差记录 → `raw/experiences/general_perf/` |

写入 INGESTION 事件到 `log.md`。

---

### 流程总结

```
codegen_init 检索（缓存miss → 7步检索 → 保存缓存）
    ↓
知识单元注入子Agent（C1-C5 + OP + A + B + E + T + S）
    ↓
编译失败 → compile_error 检索 → F1(fail) → 修复 → F1(success)
    ↓
精度通过 → F2(pass)
    ↓
第1轮 Profiling → perf_tuning 检索 → 对角tiling优化
    ↓
第2轮 Profiling → 收敛 → 写 F3（speedup=1.80x）
    ↓
Source B 摄入（成功经验）+ Source C 摄入（调试记录）+ Source D 摄入（瓶颈偏差）
    ↓
写 INGESTION 事件
```

---

## 相关文档

- `SKILL.md` — Skill 入口 + 七步检索流程
- `references/retrieval-workflow.md` — 完整检索工作流
- `references/cache-protocol.md` — 缓存接口规范 + 命名规则
- `references/knowledge-unit-format.md` — 知识单元类型定义
- `retrieval/quality_scoring.md` — 四维质量评分
- `builder/auto_ingestion.md` — Source B/C/D 自动摄入
- `maintenance/periodic_tasks.md` — 定期维护任务
- `log.md` — 反馈事件日志（所有事件写入此文件）
