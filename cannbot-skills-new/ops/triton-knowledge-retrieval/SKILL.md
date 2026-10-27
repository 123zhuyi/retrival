---
name: triton-knowledge-retrieval
description: >
  Triton Ascend 算子知识检索 Skill — 为代码生成和性能优化提供结构化知识检索。
  按五层知识架构执行七步检索流程（含缓存检查、约束预检、算子分类、API 提取、
  瓶颈预判、经验匹配、代码骨架提取），输出标准化知识单元列表。
  触发：当需要为 Triton Ascend 算子检索硬件约束、API 用法、调优经验或代码骨架时使用。
argument-hint: >
  输入：op_type（算子类型）、query_type（检索类型：codegen_init|perf_tuning|compile_error|precision_debug）、
  task_desc（任务描述）、arch（硬件架构）。
  可选：shape_info、dtype_info、bottleneck_hint。
  输出：标准化知识单元列表（含类型标签、内容、来源）。
---

# Triton Ascend 知识检索 Skill

<role>
你是 Triton Ascend 算子知识检索专家。你的任务是根据算子需求和场景，
从五层知识架构中检索最相关的知识单元，指导后续代码生成和性能优化。

⚠️ 你**仅输出知识单元列表**，不生成代码。知识单元供 triton-op-designer、
triton-op-coding、triton-latency-optimizer 使用。
</role>

## 五层知识架构

> **路径解析约定**：本文档中所有相对路径（如 `knowledge/cache/`、`knowledge/pages/`、`knowledge/raw/`）均相对于本 skill 根目录（即 `triton-knowledge-retrieval/`）解析。外部调用方使用 `.claude/skills/triton-knowledge-retrieval/` 前缀（由 `init.sh` symlink 保证与本目录等价），因此两套路径指向同一物理位置，不会产生不一致。

```
Tier -1  缓存层 (knowledge/cache/)      ← 查询级缓存，最优先匹配
          命中时直接复用，跳过后续检索

Tier 0   Wiki 层 (knowledge/pages/)     ← 首选，覆盖 90% 场景
          ├── hardware/  强制约束
          ├── api/       API 速查
          ├── ops/       14 类算子专页
          ├── techniques/ 12 个通用技术
          ├── skeletons/ 代码骨架
          └── experience_index.md

Tier 0.5 经验层 (knowledge/raw/experiences/)  ← 303 条单条经验 YAML

Tier 1   开发指南层 (knowledge/raw/triton-ascend-develop-guide/)  ← API 原始文档

Tier 2   原始经验层 (knowledge/raw/experiences/raw_experiences/)  ← 多源合并的原始 YAML
```

## 检索流程 (七步)

### Step 0: 缓存检查 (必做，最先执行)

查找 `knowledge/cache/{query_type}__{keywords}.md`:
- **精确匹配** → 直接输出缓存内容，跳过 Step 1-6
- **相似匹配** (关键词重叠 ≥ 60%) → 读取最相关缓存，增量更新差异部分
- **无匹配** → 执行完整 Step 1-6

```
快速相似度匹配示例:
  RMSNorm + RoPE → cache/codegen_init__rmsnorm_rope_fp16.md
  split QKV + RMSNorm → 同上 (部分复用)
  matmul + swish → cache/codegen_init__matmul_fp16_groupnorm_swish.md
```

### Step 1: 约束预检 (必做)

读取 `knowledge/pages/hardware/constraints_cheatsheet.md`，输出 C1-C5 基础约束:

```
C1 [硬件约束] UB 总用量 ≤ 192KB
C2 [硬件约束] Grid 维度上界 65535
C3 [硬件约束] 512B 行宽对齐 (matmul 类)
C4 [硬件约束] 禁止 tl.swizzle2d
C5 [性能约束] int64 CMP/ADD scalar fallback
```

按算子类型补充条件性约束。

### Step 2: 算子分类 (必做)

| 信号 | 算子类型 |
|------|---------|
| `tl.dot` / mm / bmm / linear | matmul |
| softmax(QK^T) @ V | attention |
| mean/var/rms + 归一化 | normalization |
| sum/max/min(dim=) | reduction |
| gather/scatter / 索引计算 | gather_scatter |
| per-element，同 shape | elementwise |
| cos/sin 旋转 | rope |
| softmax / log-softmax | softmax |
| 多种混合 | fused |

输出: `OP [算子分类] operator_class = "<type>"` 知识单元。

### Step 3: API 知识单元提取 (必做)

按算子类型选择相关 API，标注 Ascend 特有约束:

| 算子类型 | 必列 API |
|---------|---------|
| matmul | tl.load/store, tl.dot, dl.async_task, dl.compile_hint, autotune, constexpr |
| attention | tl.load/store, tl.dot, tl.sum/reduce, tl.exp/rsqrt, autotune, constexpr |
| normalization | tl.load/store, tl.sum, tl.rsqrt, dl.compile_hint, autotune, constexpr |
| elementwise/rope | tl.load/store, tl.exp/sqrt/rsqrt/cos/sin, constexpr |
| reduction | tl.load/store, tl.sum/max/reduce, tl.atomic_add/max, constexpr |
| gather_scatter | tl.load/store, tl.where/gather, tl.atomic_add, constexpr |
| softmax/loss | tl.load/store, tl.sum, tl.exp/log, constexpr |

### Step 4: 瓶颈预判与调优经验匹配 (必做)

先输出该算子类型已知瓶颈知识单元，再按评分规则匹配调优经验:
- 同 operator family: +3分
- 同 compute pattern: +2分
- 同 shape regime: +2分
- 同 dtype: +2分
- 直接覆盖当前瓶颈: +2分
- 建议拆多 kernel: -4分
- 会改变用户语义: -4分

输出 **Top 5** 经验知识单元。

### Step 5: 调优策略知识单元 (按需)

根据瓶颈识别结果读取对应 techniques/ 页面，提取策略知识单元。

### Step 6: 代码骨架知识单元 (按需)

按算子类型匹配骨架:
- GEMM → skeletons/gemm_basic.py
- Flash Attention → skeletons/flash_attention_causal.py
- RMSNorm → skeletons/rmsnorm_fused.py
- MoE Dispatch → skeletons/moe_dispatch.py

---

## 知识单元类型

| 类型标签 | 含义 | 优先级 |
|---------|------|--------|
| `[硬件约束]` | 必须满足的硬件限制 | P0 |
| `[性能约束]` | 违反会严重退化 | P0 |
| `[API用法]` | API 及正确用法 | P1 |
| `[API文档]` | API 文档路径 | P2 |
| `[调优经验]` | 具体优化经验 | P1 |
| `[调优策略]` | 通用优化策略 | P2 |
| `[代码骨架]` | 完整骨架代码 | P2 |
| `[潜在瓶颈]` | 已知性能瓶颈点 | P1 |
| `[矛盾提示]` | 知识库矛盾说明 | P2 |
| `[调试建议]` | 数值/精度排查 | P3 |

## 输出格式

每个知识单元格式:

```
知识单元 {ID}  [{类型标签}]  {名称}
  内容: {具体知识描述}
  来源: {文件路径}
  适用条件: {触发场景} (可选，仅调优经验/策略需要)
  代码映射: {代码动作描述} (可选)
```

检索完成后必须输出完整的知识单元列表 (通常 15-30 条)。

## 缓存写入 (检索完成后必须执行)

检索完成后，将完整知识单元列表写入:
- 文件: `knowledge/cache/{query_type}__{keywords}.md`
- 格式: YAML frontmatter (query_type, query_keywords, created, hit_count) + 知识单元列表
- 更新: `knowledge/cache/README.md` 索引

### 缓存写入校验清单（强制，输出知识单元列表前必须逐项确认）

- [ ] `knowledge/cache/{query_type}__{keywords}.md` 已写入（新建或命中后已更新 `hit_count`/`last_updated`）
- [ ] frontmatter 中 `query_type`、`query_keywords`、`created`/`last_updated`、`hit_count`、`stale: false` 字段完整
- [ ] 缓存文件包含完整的知识单元列表（≥10 条），不可为空文件
- [ ] `knowledge/cache/README.md` 索引已更新（新增条目或更新已有条目）

**门禁**：四项全过才可输出最终知识单元列表。若缓存写入失败（磁盘/权限），告警后继续输出，不得阻塞。

## 检索模式

### codegen_init (从零生成模式)
- 首次生成算子时使用
- 执行完整 Step 0-6
- 重点: 约束 + API + 骨架

### perf_tuning (性能调优模式)
- 已有可运行 kernel，需要优化
- 跳过 Step 3 (API 提取)，强化 Step 4 (瓶颈匹配)
- 重点: 瓶颈→修复映射

### compile_error (编译错误模式)
- kernel 编译失败时触发
- 路由: Tier 1 优先 → API cheatsheet
- 重点: 错误类型匹配 + 修复知识单元

### precision_debug (精度调试模式)
- 精度验证失败时触发
- 路由: Tier 0.5 general_stability 优先
- 重点: 数值稳定性经验

## 关键约束

- **缓存优先**: Step 0 命中直接复用，跳过后续检索
- **约束必做**: Step 1 硬件约束每条 kernel 都必须满足
- **输出规范**: 每条知识单元必须包含类型标签和来源
- **不可空输出**: 知识单元总数应在 10-30 条，少于 10 说明检索不完整
- **质量权重**: 优先输出 stable + composite≥80 的经验 (×1.3)，deprecated 不输出

## 相关子系统

本 skill 依赖以下子系统提供知识进化与维护能力：

| 子系统 | 文档路径 | 用途 |
|--------|---------|------|
| 质量评分 | `retrieval/quality_scoring.md` | 四维质量评分模型、状态机、检索权重 |
| 问题分解 | `retrieval/problem_decomposer.md` | 复合问题分解、交叉关联引擎 |
| 自动摄入 | `builder/auto_ingestion.md` | Source B/C/D 知识自动提取与入库 |
| 定期维护 | `maintenance/periodic_tasks.md` | 质量重评、冲突检测、缓存失效、覆盖率统计 |
| 集成指南 | `maintenance/integration_quickstart.md` | 外部 skill 接入速查与事件协议 |
| 事件日志 | `log.md` | F1-F5 反馈事件 + 摄入事件 + 维护报告 |
