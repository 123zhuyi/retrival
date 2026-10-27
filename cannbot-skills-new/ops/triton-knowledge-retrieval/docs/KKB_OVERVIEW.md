# Triton-Ascend KKB 知识库

> KKB (Kernel Knowledge Base) 是面向 Triton-Ascend 算子自动生成的领域知识基础设施。
> 本文档从设计理念、核心架构、cannbot 接入方案三个维度，阐述 KKB 作为 cannbot
> 现有知识体系的**增强层**，而非替代层，如何提升算子生成的成功率和性能。

---

## 一、设计理念

### 1.1 问题的起点

在 Triton-Ascend 算子自动生成场景中，LLM 的代码生成能力很强，但缺乏 Ascend NPU 领域的专业知识：

- **硬件约束盲区**：LLM 不知道 Ascend 910B 的 UB 只有 192KB、512B 行宽对齐要求、禁止 tl.swizzle2d 等硬性约束，生成的代码编译失败率高
- **经验无法复用**：每次算子生成都是从零开始，上一次成功探索出的优化技巧（如对角 tiling 消除 bank conflict）无法自动传递给下一次任务
- **知识检索低效**：平铺式知识库需要遍历大量文件，LLM context 利用率低
- **缺乏设计阶段防错**：约束冲突（如 GroupNorm 融合时 BLOCK_N=256 与 channels_per_group=64 的矛盾）到编译阶段才发现，导致无效迭代

### 1.2 四个核心设计原则

**原则 1：分层检索，高频优先。** 知识按访问频率分层组织。最常用知识（硬件约束、API 用法）最小、最结构化，确保 LLM 最少 context 获取。不常用知识（原始 API 文档、专家原始录入）更大、更详细，放在低优先级层。

**原则 2：知识单元化，直接可执行。** 知识不能停留在"自然语言描述"层面。每条知识必须能直接映射到一个代码动作——"知道 UB 是 192KB"不够，必须能映射为"BLOCK_M=128, BLOCK_N=256"。

**原则 3：自我进化，越用越好。** 知识库不应是静态文档集合。每次算子生成成功或失败，都应自动沉淀为新的知识条目。知识库随使用次数增加而持续增值——缓存命中率上升、经验质量提高、知识盲区减少。

**原则 4：设计阶段防错，而非编译阶段发现。** 约束冲突应在设计阶段就被识别和提示。与其让 LLM 生成一个注定失败的 kernel 再返工，不如在代码生成前就告诉它"这两个约束冲突，需要拆成两个 kernel"。

---

## 二、核心架构与技术创新点

### 2.1 总体架构：三层知识架构

KKB 采用三层知识架构，按访问频率从高到低、文件粒度从小到大的原则组织：

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                          KKB 三层知识架构                                     │
│                      访问频率：高 ◄────────────── 低                           │
│                      文件大小：小 ◄────────────── 大                           │
├─────────────────────────────────────────────────────────────────────────────┤
│                                                                             │
│  ┌─────────────────────────────────────────────────────────────────────┐   │
│  │ Tier 0  缓存层 (knowledge/cache/)                                    │   │
│  │ ┌──────────┐ ┌──────────┐ ┌──────────┐ ┌──────────┐ ┌──────────┐   │   │
│  │ │ codegen  │ │ compile  │ │ perf     │ │ precision│ │    ...   │   │   │
│  │ │ _init__  │ │ _error__ │ │ _tuning__│ │ _debug__ │ │          │   │   │
│  │ │ matmul.. │ │ matmul.. │ │ matmul.. │ │ rmsnorm..│ │          │   │   │
│  │ └──────────┘ └──────────┘ └──────────┘ └──────────┘ └──────────┘   │   │
│  │ 单文件 ≤300 行 · Jaccard 相似度匹配 · 命中直接复用                     │   │
│  └─────────────────────────────────────────────────────────────────────┘   │
│                                    │ 未命中                                  │
│                                    ▼                                         │
│  ┌─────────────────────────────────────────────────────────────────────┐   │
│  │ Tier 1  Wiki 层 (knowledge/pages/)         28 页 · 单页 ≤200 行      │   │
│  │ ┌──────────┐ ┌──────────┐ ┌──────────┐ ┌──────────┐ ┌──────────┐   │   │
│  │ │hardware/ │ │  api/    │ │  ops/    │ │techniques│ │skeletons/│   │   │
│  │ │强制约束  │ │API 速查  │ │14 类算子 │ │  /       │ │代码骨架  │   │   │
│  │ │          │ │          │ │ 专页     │ │12 通用   │ │          │   │   │
│  │ │          │ │          │ │          │ │ 技术     │ │          │   │   │
│  │ └──────────┘ └──────────┘ └──────────┘ └──────────┘ └──────────┘   │   │
│  │ 首选检索层 · 覆盖 90% 场景 · 双维度交叉引用                             │   │
│  └───────────────────────────┬─────────────────────────────────────────┘   │
│                              │ 深度查询                                      │
│              ┌───────────────┴───────────────┐                               │
│              ▼                               ▼                               │
│  ┌──────────────────────────┐  ┌──────────────────────────────────────┐   │
│  │ Tier 2  结构化经验层                      │  │ Tier 2  开发指南层                     │   │
│  │ (knowledge/raw/                           │  │ (knowledge/raw/triton-ascend-develop-guide/)        │   │
│  │  experiences/)                            │  │ 154 个 API 原始文档                   │   │
│  │                           │  │                                      │   │
│  │  ┌─ 数据来源 ──────────┐  │  │ 仅在 Tier 1 Wiki 无法覆盖时           │   │
│  │  │  扁平 MD 文档        │  │  │ 回退查阅                              │   │
│  │  │  raw_experiences/ │  │  │                                      │   │
│  │  │      │ 结构化抽取     │  │  │                                      │   │
│  │  │      ▼               │  │  │                                      │   │
│  │  │  结构化 YAML 经验     │  │  │                                      │   │
│  │  │  · match_cond 触发条件│  │  │                                      │   │
│  │  │  · code_snippet 代码  │  │  │                                      │   │
│  │  │  · performance_boost  │  │  │                                      │   │
│  │  │  · composite_score    │  │  │                                      │   │
│  │  └──────────────────────┘  │  │                                      │   │
│  │                           │  │                                      │   │
│  │  ┌──────────┐ ┌──────────┐│  │                                      │   │
│  │  │ matmul   │ │attention ││  │                                      │   │
│  │  │ 38 条    │ │ 47 条    ││  │                                      │   │
│  │  └──────────┘ └──────────┘│  │                                      │   │
│  │  303 条独立 YAML · 19 类目│  │                                      │   │
│  └──────────────────────────┘  └──────────────────────────────────────┘   │
│                                                                             │
│  ┌─────────────────────────────────────────────────────────────────────┐   │
│  │ 维护闭环：Source B/C/D 自动摄入 → _index.yaml 更新 → 定期维护        │   │
│  │ → 质量重评 → 冲突检测 → 缓存失效 → Wiki Distiller → Tier 1 更新      │   │
│  └─────────────────────────────────────────────────────────────────────┘   │
└─────────────────────────────────────────────────────────────────────────────┘
```

**三层架构的设计逻辑**：

- **Tier 0 缓存层**：查询级缓存，Jaccard 语义相似度匹配。命中直接复用知识单元，context 节省 60-95%。这是"最快路径"——同类算子首次检索后，后续零额外 context 成本。
- **Tier 1 Wiki 层**：首选检索层，28 页、单页 ≤200 行。按 hardware/api/ops/techniques/skeletons 五个维度组织，通过经验路由表和双向引用链形成知识图谱，覆盖 90% 的检索场景。
- **Tier 2 经验层 + 开发指南层**：深度查询层。经验层提供 303 条结构化 YAML（含精确 match_cond + code_snippet + performance_boost），填补 Wiki 抽象与原始数据之间的精确匹配空白；开发指南层作为 API 原始文档，仅在 Wiki 无法覆盖时回退。

三层架构使检索从"遍历 260 个文件"降为"Tier 0 命中 → 直接复用；未命中 → Tier 1 单页路由 → 按需下钻 Tier 2"的渐进式路径，平均 context 消耗降低 59.3%。

### 2.2 七步检索流程与四种检索模式

在三层架构之上，KKB 定义了七步检索流程，按需在不同架构层之间路由：

```
Step 0: 缓存命中检查 (必做) → 精确/相似/无匹配 三级策略 (Tier 0)
Step 0.5: 问题分解 (可选) → 复合问题时拆解为独立子问题，各子问题独立检索后交叉关联
Step 1: 约束预检 (必做) → C1-C5 硬件约束 + 条件性约束（按算子类型）(Tier 1 hardware/)
Step 2: 算子分类 (必做) → 14 类算子决策表，输出 OP 知识单元 (Tier 1 ops/)
Step 3: API 知识单元提取 (必做) → 按算子类型关联必列 API，标注 Ascend 特有约束 (Tier 1 api/ → Tier 2 develop-guide/)
Step 4: 瓶颈预判与经验匹配 (必做) → 条件匹配评分 + 质量权重 × 系数，输出 Top 5 经验 (Tier 2 experiences/)
Step 5: 调优策略知识单元 (按需) → 根据瓶颈从 techniques/ 页面提取 (Tier 1 techniques/)
Step 6: 代码骨架知识单元 (按需) → 按算子类型匹配骨架 (Tier 1 skeletons/)
```

四种检索模式按需裁剪七步流程：

| 检索模式 | 触发场景 | 重点 Step | 弱化/跳过 |
|---------|---------|----------|----------|
| `codegen_init` | 从零生成 Triton Ascend kernel | Step 0-6 全量 | — |
| `perf_tuning` | 已有正确 kernel，优化性能 | Step 0,0.5,1,4,5 | Step 3 (API 已知), Step 6 (骨架已知) |
| `compile_error` | 编译/语法/API 报错 | Step 0 → 0.5 → 错误路由 | Step 2-6 按需 |
| `precision_debug` | 精度/NaN/cos_sim 不达标 | Step 0 → 0.5 → 精度路由 | Step 2-6 按需 |

### 2.3 创新一：结构化 YAML 经验与 10 种知识单元类型

**问题背景**

平铺知识库的优化经验以 markdown 文档散落，存在三个致命缺陷：

**缺陷 1：无法精确条件匹配。** 平铺文档中写"大矩阵场景下推荐 128×128 tiling，性能提升约 40%"，LLM 需自行理解"大矩阵"的含义——M≥256? N≥256? K≥1024? 这种模糊性导致经验在错误场景被套用，或正确场景下被忽略。

**缺陷 2：知识无法直接映射为代码。** 平铺文档中的经验是自然语言叙述——"建议用对角 tiling 消除 bank conflict"。LLM 需要把这句话翻译成具体的 `pid_m`/`pid_n` 计算公式和 grid 调度逻辑，翻译过程引入新的出错可能。

**缺陷 3：缺乏精确匹配层。** Tier 1 的 Wiki 页面（如 `matmul.md`）提供"GEMM 应该怎么优化"的通用知识，但它无法回答"给定 M=512, N=512, K=4096, fp16，当前应该用哪条经验、预期提升多少"。平铺知识库既没有 Wiki 层的抽象，也没有精确匹配层——两端优势都拿不到。

**创新方法：10 种知识单元类型体系**

KKB 将知识从"自然语言叙述"升级为"可直接映射到代码动作的结构化单元"，共 10 种类型：

| 类型标签 | 含义 | 优先级 | 代码映射动作 | 示例 |
|---------|------|--------|------------|------|
| `[硬件约束]` C | 必须满足的硬件限制 | P0 | 约束 BLOCK/Grid 参数 | UB≤192KB → BLOCK_M=128,N=256 |
| `[性能约束]` P | 违反会严重退化 | P0 | 指导融合/tiling 方案 | 大矩阵行主序 → 对角 tiling |
| `[API用法]` A | API 及正确用法 | P1 | 防止运行时错误 | tl.load care_padding=False |
| `[调优经验]` E | 具体已验证优化经验 | P1 | 复用关键代码模式 | GEMM+Swish Fusion, +30% |
| `[调优策略]` T | 通用优化策略 | P2 | kernel 骨架级指导 | 对角 tiling 映射公式 |
| `[代码骨架]` S | 完整骨架代码 | P2 | 减少从零构建风险 | gemm_basic.py 完整骨架 |
| `[潜在瓶颈]` B | 已知性能瓶颈点 | P1 | 预判+设计阶段选方案 | L2 Bank Conflict (M,N≥512) |
| `[矛盾提示]` X | 知识库矛盾说明 | P2 | 标出不可调和冲突 | GroupNorm 融合→拆两 kernel |
| `[调试建议]` — | 数值/精度排查 | P3 | 故障诊断路径 | NaN 排查→fp32 accumulator |
| `[API文档]` — | API 文档路径 | P2 | 文档路径引用 | autotune 搜索空间配置 |

**Tier 2 的桥接设计** — 经验层与开发指南层并列于 Wiki 层之下，各司其职：

```
Tier 1 Wiki (pages/)                  — 抽象，通用，无具体条件匹配
              │
    ┌─────────┴─────────┐
    ▼                   ▼
Tier 2 经验层          Tier 2 开发指南层
(experiences/)         (develop-guide/)
— 结构化 YAML          — API 原始文档
— 精确条件匹配          — 回退查阅
— 直接可执行

其中 Tier 2 的 YAML 经验并非凭空产生，而是从扁平 MD 文档中结构化抽取而来：

  扁平 MD 文档                      结构化 YAML 经验
  (raw_experiences/)             (experiences/)
  ┌──────────────────┐              ┌──────────────────────────┐
  │ # GEMM 优化经验   │              │ exp_matmul_001:           │
  │ 大矩阵场景下推荐  │   结构化     │   match_cond:             │
  │ 128×128 tiling,  │ ──抽取──▶   │     m: [>=256]           │
  │ 性能提升约 40%.  │              │     n: [>=256]            │
  │ 代码示例:        │              │     k: [>=1024]          │
  │ for m in range.. │              │   code_snippet: |        │
  │   ...模糊条件... │              │     pid_m = ...          │
  └──────────────────┘              │   performance_boost: 40% │
                                    │   composite_score: 85    │
                                    └──────────────────────────┘
```

同一条经验在两种知识库中的形态对比：

| 维度 | 平铺知识库 (MD) | KKB (YAML) |
|------|-----------------|-----------|
| 条件匹配 | LLM 需自行理解"大矩阵"的含义 | `match_cond: {m: [>=256], n: [>=256], k: [>=1024]}` 精确限定 |
| 代码可用性 | 代码混在自然语言中，难以提取 | `code_snippet` 字段直接提供可用代码 |
| 性能预期 | 模糊描述 "约 40%" | `performance_boost: 40%` 精确数值，可供排序 |
| 可查询性 | 只能全文搜索 | LLM 可按 `match_cond` / `operator_type` / `data_type` 精确筛选 |
| 排序依据 | 无 | `composite_score` 四维质量评分（频次×0.25 + 性能×0.40 + 精度×0.25 + 覆盖×0.10） |
| 可进化性 | 需人工编辑文档 | Source B/C/D 自动摄入，按模板写入新 YAML，`_index.yaml` 自动追加 |

**价值**

1. **首次编译通过率 +34pp**（48%→82%）：知识从"需要 LLM 理解的叙述"变为"可直接映射的代码约束"，消除了理解-翻译环节的出错
2. **端到端首次通过率 +40pp**（30%→70%）：E 系列经验提供的 `code_snippet` 和 T 系列策略提供的骨架，使 LLM 从"从零编写"变为"适配已有模板"
3. **知识可自动进化**：YAML 的模板化结构使 Source B/C/D 自动摄入成为可能——一次成功的算子生成完成后，`match_cond` + `performance_boost` + `code_snippet` 直接写入新 YAML，无需人工干预

### 2.4 创新二：Jaccard 相似度缓存与跨算子复用

**问题背景**

平铺知识库每次检索都需要遍历全部 ~260 个文件，消耗 ~40K tokens。更关键的是，**相似算子之间的检索结果没有复用**——生成了 `matmul_512x512_fp16` 后，紧接着生成 `matmul_1024x1024_fp16`，硬件约束和通用优化技巧完全相同，但仍需从头检索一遍。此外，Phase 3 迭代中的修复检索成本不应等同于首发检索成本——编译失败后的修复通常只需增量知识（如"512B 对齐怎么修复"），平铺知识库却每次都要完整遍历全部文件。

**创新方法**

KKB 引入 **Jaccard 相似度缓存**，用关键词集合的交并比判断语义相似度，而非按"文件名相同"的粗糙匹配：

```
Jaccard 相似度 = |A ∩ B| / |A ∪ B|

查询 A: {split_qkv, rmsnorm, rope, fp16}
缓存 B: {split_qkv, rmsnorm, fp16}
Jaccard = 3/4 = 0.75 → 相似命中 (≥0.60)
```

**三级命中策略**：

| 命中级别 | 条件 | 策略 | Context 节省 |
|---------|------|------|-------------|
| 精确命中 | Jaccard = 1.00 | 直接复用所有知识单元 | ~95% |
| 相似命中 | Jaccard ≥ 0.60 | 复用交集部分 + 差异增量检索 | 60-80% |
| 无命中 | Jaccard < 0.60 | 完整七步检索 | 0% |

**增量合并规则**（相似命中时按知识单元系列分别处理）：

| 系列 | 合并策略 | 原因 |
|------|---------|------|
| C 系列 (硬件约束) | 去重合并 | 硬件相同，约束相同 |
| E 系列 (调优经验) | 交集复用，差异重新匹配 | 部分经验算子专属 |
| T 系列 (策略) / S 系列 (骨架) | 覆盖更新 | 总是取最新 |
| B 系列 (瓶颈) / X 系列 (矛盾) | 追加 | 新的瓶颈/矛盾信息有增量价值 |

**跨算子类型复用**。传统缓存按"文件名完全相同"命中，但不同算子类型之间文件名不可能相同。KKB 的缓存按**知识单元系列**拆分（C 系列硬件约束、E 系列调优经验等），即使算子类型不同，硬件约束和通用优化技巧仍可直接复用：

```
查询 A (matmul + RoPE + fp16) 的 C 系列(硬件约束) → 直接命中（硬件相同，约束相同）
查询 A 的 E 系列(调优经验) → 部分命中（matmul 专属不可用，但通用标量消除经验可用）
```

Stale 检测：deprecated 经验 / 90 天未命中 → 自动跳过并覆写。

**价值**

1. **缓存总体命中率 80%**，平均每次检索节省 **66.2%** context；命中率随使用从 50% 持续提升至 90%
2. **失败修复成本大幅低于首发**：compile_error 场景缓存命中率 72%、precision_debug 场景 78%，修复检索仅需 ~2K tokens
3. **跨算子知识复用**：硬件约束 C 系列在所有算子间共享，通用标量消除技巧跨 matmul/attention/normalization 复用

### 2.5 创新三：Wiki 层知识图谱与双维度交叉引用

**问题背景**

平铺知识库的知识组织有三个结构性问题：

1. **知识孤岛**：`matmul_optimization.md` 不知道 `attention_notes.md` 使用了相同的对角 tiling 技巧。一条通用技术（如 512B 行宽对齐）在多个文件中重复出现但互不链接。
2. **按实例存放导致文件膨胀**：每增加一个算子变体就建一个新文件（如 `matmul_512x512.md`、`bmm_batch8.md`），类目间无抽象，N 个变体需要 N 个文件。
3. **搜索空间大且无导航**：LLM 需遍历全部 ~260 个文件逐一判断是否相关，无路由机制也无"顺藤摸瓜"的探索能力。

**创新方法：ops × techniques 双维度抽象**

ops 页面按**算子类目**抽象（13 个类目），而非按算子实例存放：

| | 平铺知识库 | KKB (pages/ops/matmul.md) |
|---|---|---|
| **组织维度** | 按算子实例：N 个变体 N 个文件 | 按算子类目：1 个页面覆盖所有 GEMM 变体 |
| **内容** | 具体某个 kernel 的完整实现 | 检测标准 + 适用条件表 + **经验路由表** + 关键技术 + 代码模板 + 常见陷阱 |
| **检索方式** | LLM 遍历目录逐一判断 | LLM 一次性读取页面，通过经验路由表找到匹配项 |

ops 页面的核心定位是**路由，而非存储**。它不直接存放所有经验细节，而是通过经验路由表告诉 LLM："你的条件是 M=512, N=512，匹配 `exp_matmul_diag_tiling_001`，预期提升 25%，详细代码去读对应 YAML"。单页 ≤200 行即可覆盖整个类目。

techniques 页面则集中维护跨算子类型的通用技术（12 个页面），避免在每个 ops 页面重复。一条 512B 行宽对齐知识被 12 个页面引用，一次更新全局生效。

**多层索引构建知识图谱**：

```
                      ┌─────────────────────────────────┐
                      │   index.md (统一检索入口)         │
                      │   路由表: matmul → ops/matmul.md  │
                      └──────────────┬──────────────────┘
                                     │
                                     ▼
                      ┌─────────────────────────────────┐
                      │   ops/matmul.md (Tier 1)         │
                      │   经验路由表:                     │
                      │   "M,N≥512 → exp_matmul_diag_    │
                      │    tiling_001 (25%) →             │
                      │    experiences/matmul/            │
                      │    exp_matmul_diag_tiling_001     │
                      │    .yaml"                        │
                      │   相关页面:                       │
                      │   → techniques/tiling.md  ────────┼──────────┐
                      │   → techniques/core_mapping.md ──┼──────┐   │
                      │   → hardware/ascend_910b.md ─────┼──┐   │   │
                      └──────────────┬───────────────────┘  │   │   │
                                     │                      │   │   │
                                     ▼                      │   │   │
┌────────────────────────────────────────────────────────┐  │   │   │
│  experience_index.md (经验速查索引)                      │  │   │   │
│  "M,N≥512 → exp_matmul_diag_tiling_001 (25-35%)        │  │   │   │
│   → ops/matmul.md + techniques/core_mapping.md         │  │   │   │
│   → experiences/matmul/exp_matmul_diag_tiling_001.yaml │  │   │   │
└────────────────────┬───────────────────────────────────┘  │   │   │
                     │                                      │   │   │
                     ▼                                      ▼   ▼   ▼
┌────────────────────────────────────────────────────────────────────┐
│  _index.yaml (全量元数据索引, 303条)                                 │
│  exp_matmul_diag_tiling_001: category=matmul, boost=25%             │
└────────────────────────────────────────────────────────────────────┘
```

**双向引用 + 按需渐进加载**：

每个 Wiki 页面都包含"相关页面"（正向引用）和"被以下页面引用"（反向引用）。图谱结构使知识按层级渐进加载，而非一次性塞入 context：

```
第 1 层加载: index.md → 路由到 ops/matmul.md           (~2K tokens)
第 2 层加载: matmul.md 经验路由表                       (~3K tokens)
第 3 层加载: 匹配的 YAML 经验 + 引用的 technique 页面    (~10K tokens)
                                                       ─────────────
                                                        ~15K tokens
```

**价值**

1. **搜索空间从 O(260 文件) 降到 O(1 页面 + 1-3 条经验)**：路由边代替了 LLM 的逐个判断
2. **平均 context 消耗降低 59.3%**（40,870 → 16,620 tokens），文件读取数减少 62.9%（13.2 → 4.9 个）
3. **O(1) 维护**：硬件约束变更只需修改 1 个 hardware/ 页面，所有引用页自动反映；新技术只需更新 1 个 techniques 页面，12 个 ops 页面通过引用自动受益

### 2.6 创新四：知识自动进化闭环

**问题背景**

平铺知识库是静态文档集合。每次算子生成任务中探索出的优化技巧（如发现新的 tiling 策略、修复编译错误的经验、精度调试的过程），全部散落在各次任务日志中无法复用。知识库不随使用增长——100 次任务后和 1 次任务后，知识库的内容完全相同。

**创新方法**

KKB 设计了从反馈事件到知识沉淀的完整闭环：

```
反馈事件 (F1-F5) → 自动摄入 (Source B/C/D) → _index.yaml 更新
→ 定期维护 (每 10 次 F3) → 质量重评 + 冲突检测 + 缓存失效 + 覆盖统计
→ Wiki Distiller → Tier 1 Wiki 提炼更新
```

**五类反馈事件**：

| 事件 | 触发时机 | 记录内容 |
|------|---------|---------|
| F1 编译修复 | 编译失败 → 修复成功 | failed_unit_id, error_message, fix_applied |
| F2 精度修复 | 精度失败 → 精度通过 | max_err, cosine_sim, failure_unit_ids |
| F3 性能收敛 | 每轮优化完成后 | bottleneck_predicted/actual, high/low_impact_units |
| F4 缓存命中 | 检索时命中缓存 | cache_file, hit_type, jaccard_similarity |
| F5 用户纠正 | 用户否定某知识单元 | unit_id, correction_type, reason |

**三种自动摄入来源**：

| 来源 | 触发条件 | 写入目标 |
|------|---------|---------|
| Source B (成功经验) | 精度全过 + speedup≥1.1 + 有效 profiling≥1 轮 | raw/experiences/<category>/ |
| Source C (调试记录) | 存在 F1(fail→success) 或 F2(fail→pass) 事件对 | raw/experiences/general_debug/ 或 general_stability/ |
| Source D (性能洞察) | 实测瓶颈与预判偏差 > 20% | raw/experiences/general_perf/ 或 <category>/ |

**四维质量评分**决定经验是否进入 stable 状态：

```
composite = frequency×0.25 + performance×0.40 + precision×0.25 + coverage×0.10
```

其中 `performance` 权重最高（0.40），因为性能提升是经验的最终价值体现。

**状态机**：`draft → stable (composite≥60, 使用≥2次) → review → deprecated`

**价值**

在 50 次连续任务的模拟测试中：

| 指标 | 初始 | 50次后 | 变化 |
|------|------|--------|------|
| 经验总数 | 303 | 355 | **+17.2%** |
| 高质量经验 (composite≥80) | 45 | 85 | **+88.9%** |
| 缓存命中率 | 31% | 55% | **+24pp** |
| 知识盲区 (stable<3 类目) | 3 | 0 | **清零** |

平铺知识库在同样 50 次任务后完全不变——经验散落在任务日志中，无法被后续任务复用。

### 2.7 创新五：问题分解器与交叉关联引擎

**问题背景**

复杂算子生成场景中，LLM 面临的问题往往是复合的——"kernel 精度异常 + 性能低于基线 40%"这类输入包含了精度和性能两个独立子问题。平铺知识库要求 LLM 自行将复合问题分解为子问题并分别检索，但 LLM 在没有领域知识引导时难以正确分解。更关键的是，子问题的解之间可能存在冲突（如"使用 fp32 accumulator 保证精度"和"UB 容量上限 192KB"不可兼得），平铺知识库无法在设计阶段发现这些冲突——它们到编译阶段才暴露，导致无效迭代。

**创新方法**

KKB 引入**问题分解器**和**交叉关联引擎**两个互补机制：

```
输入: "kernel 精度异常 + 性能低于基线 40%"

┌──────────────────────────────────────────────────────────────────┐
│                    问题分解器                                     │
│  ┌─────────────────────┐     ┌─────────────────────┐             │
│  │  子问题 1: 精度异常  │     │  子问题 2: 性能低下  │             │
│  │  路由: precision_   │     │  路由: perf_tuning   │             │
│  │        debug         │     │                      │             │
│  │  检索: general_     │     │  检索: ops+techniques│             │
│  │        stability/    │     │                      │             │
│  └──────────┬──────────┘     └──────────┬──────────┘             │
└─────────────┼──────────────────────────┼─────────────────────────┘
              │                          │
              └──────────┬───────────────┘
                         │
                         ▼
┌──────────────────────────────────────────────────────────────────┐
│                 交叉关联引擎                                      │
│                                                                  │
│  因果关联: 精度低 ← fp16 累加溢出 ← 未使用 fp32 accumulator         │
│  冲突关联: 使用 fp32 acc ⊗ UB 翻倍 ← 可能超出 192KB                 │
│  互补关联: fp32 acc + persistent kernel = 精度保证 + 延迟降低       │
│  泛化关联: 精度修复模式 → general_stability 通用经验                 │
└──────────────────────────────────────────────────────────────────┘
```

**四类关联类型**：

- **因果关联**：从症状追溯到根因。如"精度低 ← fp16 累加溢出 ← 未使用 fp32 accumulator"，帮助 LLM 理解"为什么"而不仅是"怎么办"
- **冲突关联**：在设计阶段就标记不可兼得的策略组合。如 GroupNorm + Swish 融合场景中，`BLOCK_N=256`（512B 对齐要求）与 `channels_per_group=64`（GroupNorm tile 限制）的冲突，通过 X 系列（矛盾提示）直接告诉 LLM"这两个约束不可调和，必须拆成两个 kernel"
- **互补关联**：识别可叠加收益的策略组合。如 `fp32 acc + persistent kernel`，两者独立使用时各有代价，但组合使用可同时获得精度保证和延迟降低
- **泛化关联**：将特定算子的成功经验抽象为通用技术模式，写入 techniques 页面供其他算子类目复用

**价值**

1. **冲突场景迭代次数减少 58.3%**（2.9 → 1.2 轮）：约束冲突在设计阶段就被识别和提示，而非到编译阶段才发现
2. **冲突场景首次成功率从 20% 提升到 80%**（+60pp）：典型如 GroupNorm+Swish 融合场景，平铺知识库需要 3 轮失败探索才能发现"必须拆 kernel"，KKB 第一轮就直接设计两 kernel 方案
3. **交叉关联引擎使知识图谱中的"边"从静态引用升级为动态推理**：LLM 不仅知道 A 页面引用了 B 页面，还知道在特定场景下 A 和 B 之间的因果/冲突/互补关系

---

## 三、cannbot 接入方案：增强而非替代

### 3.1 核心接入原则

cannbot 已有一套完整的、经过验证的知识体系：

| cannbot 现有知识层 | 内容 | 管理方式 |
|-------------------|------|---------|
| `.claude/template/{category}.md` | Layer 1 设计约束 + Layer 2 算法骨架 + Layer 3 关键代码片段 | Phase 7 人工提炼归档，跨会话复用 |
| `ops/triton-op-designer/references/` | UnifiedSketch DSL 规范 + 22 个手写优化案例 | Skill 级别，随开发迭代更新 |
| `ops/triton-op-coding/references/` | API 编程基础 + 示例代码 + 6 类算子参考文档 | Skill 级别 |
| `ops/triton-latency-optimizer/references/` | 30 个优化点 + checklist + block_size_scaling | Skill 级别 |
| `ops/npu-arch/references/` | NPU 架构指南 + 硬件参数 | 全局共享 |

**KKB 的角色是增强层，不是替代层**。KKB 在上述现有知识体系之上，提供三个 cannbot 原本不具备的核心能力：

1. **结构化跨会话经验检索** — template 只沉淀过去经验的抽象（Layer 1-3），KKB 可以检索 303 条具体的、条件匹配的、已验证的调优经验（含精确的 match_cond + code_snippet + boost 数据）
2. **自动知识进化** — template 依赖 Phase 7 人工提炼，KKB 通过 Source B/C/D 在每次成功生成后自动摄入新经验
3. **设计阶段冲突检测** — template Layer 1 列出禁止项，但 KKB 的 X 系列（矛盾提示）能识别两条约束之间的动态冲突并在设计阶段预警

### 3.2 接入方式：子 Skill 自主检索

**核心决策：检索流程下放到各子 Skill，在动作前触发。**

采用**子 Skill 自主检索**模式：每个子 Skill 在执行核心动作之前，自行调用 `triton-knowledge-retrieval` 完成检索并立即消费结果。检索→消费紧密衔接，无中间环节。Jaccard 缓存机制保证同类算子后续检索零额外 context 成本。

```
新接入后的调用链路（子 Skill 各自独立触发检索）:

  triton-op-designer (Phase 2 Step 2)
  ├── [新] Step 0: 调用 triton-knowledge-retrieval (codegen_init, Step 0-2)
  │     获取: 缓存命中 / 硬件约束 C1-C5 / 算子分类 OP 知识单元
  ├── [现有] 加载 template/{category}.md (Layer 1/2)
  ├── [现有] 加载硬件规格、手写优化案例
  └── 产出: sketch (含知识单元约束引用)

  triton-op-coding (Phase 3.1)
  ├── [新] Step 0: 调用 triton-knowledge-retrieval (codegen_init, Step 0-6 全量)
  │     获取: API 知识单元 / 瓶颈预判 / Top 5 调优经验 / 策略 / 代码骨架
  │     若 designer 已检索过同类算子，Step 0 缓存命中 → 直接复用（零额外耗时）
  ├── [现有] 加载 template/{category}.md (Layer 1/3)
  ├── [现有] 加载硬件规格、API 参考
  └── 产出: 代码 (含结构化注释头 + 知识-代码映射表)

  triton-op-coding (Phase 3, 模式 3 — 编译修复)
  ├── [新] 调用 triton-knowledge-retrieval (compile_error)
  │     获取: Tier 1 API cheatsheet → 错误类型匹配 + 修复知识单元
  │     cache 命中率 72%，平均 ~2K tokens
  ├── [现有] Conductor 错误分析
  └── 产出: 修复后代码

  triton-op-coding (Phase 3, 模式 4 — 精度调试)
  ├── [新] 调用 triton-knowledge-retrieval (precision_debug)
  │     获取: 经验层 general_stability 优先 → 数值稳定性经验
  │     cache 命中率 78%，平均 ~2K tokens
  ├── [现有] Conductor 精度分析
  └── 产出: 修复后代码

  triton-latency-optimizer (Phase 4, 每轮优化)
  ├── [新] Step 0: 调用 triton-knowledge-retrieval (perf_tuning)
  │     获取: 瓶颈→修复映射 / 算子类别高频优化点 / Top-3 调优经验
  │     帮助精准跳过不相关优化点，聚焦有效优化
  ├── [现有] 按顺序检查 30 个优化点
  └── 产出: 优化后代码 (含优化策略说明)
```

**检索先后顺序的价值**：designer → coding → optimizer 三级流水线共享 Jaccard 缓存。designer 的 `codegen_init` 检索写入缓存后，coding 的 `codegen_init` 检索立即精确命中，零额外 context 成本。compile_error 和 precision_debug 使用独立缓存 key，互不干扰。

### 3.3 KKB 与 cannbot 现有知识的协作关系

```
cannbot 现有知识（主路径，所有场景必用）
│
├── .claude/template/{category}.md
│   Layer 1: 设计约束（硬性规则）
│   Layer 2: 算法骨架（参考方向）
│   Layer 3: 关键代码片段（可参考不可复制）
│
├── ops/npu-arch/references/     ← 硬件规格（UB 大小、核数、带宽）
├── designer/references/         ← Sketch DSL + 22 个手写案例
├── coding/references/           ← API 基础 + 6 类算子参考
└── optimizer/references/        ← 30 个优化点 + checklist

                    +
                    │
    ╔══════════════════════════════════════════════════╗
    ║         KKB 增强层（子 Skill 动作前自主检索）     ║
    ║                                                  ║
    ║  ┌─────────────────────────────────────────┐    ║
    ║  │ 1. 缓存层：Jaccard 相似度命中，90% 场景  │    ║
    ║  │    首次检索后同类算子零额外 context 复用 │    ║
    ║  ├─────────────────────────────────────────┤    ║
    ║  │ 2. Wiki 层：C1-C5 硬件约束（补充验证     │    ║
    ║  │    template Layer 1 的完整性）           │    ║
    ║  │    + 14 类算子经验路由表                 │    ║
    ║  ├─────────────────────────────────────────┤    ║
    ║  │ 3. 经验层：303 条结构化 YAML             │    ║
    ║  │    精确条件匹配 → LLM 直接注入代码       │    ║
    ║  ├─────────────────────────────────────────┤    ║
    ║  │ 4. 进化层：F1-F5 反馈 → Source B/C/D    │    ║
    ║  │    自动摄入 → 知识库越用越好             │    ║
    ║  └─────────────────────────────────────────┘    ║
    ╚══════════════════════════════════════════════════╝
```

**各子 Skill 的 KKB 检索时机与消费方式**：

| 子 Skill | 检索时机 | 检索模式 | 消费方式 |
|---------|---------|---------|---------|
| designer | 设计草图前 | `codegen_init` (Step 0-2) | C 系列（硬件约束）补充验证 template Layer 1；OP 分类结果指导算子路由；B 系列（瓶颈）预判性能风险；X 系列（矛盾提示）预警设计阶段冲突 |
| coding | 代码生成前 | `codegen_init` (Step 0-6 全量) | A 系列（API 用法）补充 Ascend 特有约束；E 系列（调优经验）提供条件匹配代码片段；S 系列（骨架）减少从零构建 |
| coding (模式 3) | 编译失败后、修复前 | `compile_error` | 错误类型匹配 + 修复知识单元，cache 命中率 72% |
| coding (模式 4) | 精度失败后、调试前 | `precision_debug` | general_stability 经验优先 + 数值稳定性经验，cache 命中率 78% |
| optimizer | 每轮优化开始前 | `perf_tuning` | 瓶颈→优化点映射，优先尝试 Top-3 高收益优化点，跳过不相关优化点 |

**关键协作规则**：

| 场景 | cannbot 现有知识 | KKB 增强 |
|------|----------------|---------|
| 设计阶段 | template Layer 1 为硬性边界 | KKB C 系列（硬件约束）补充验证；KKB B 系列（潜在瓶颈）预判性能风险；KKB X 系列（矛盾提示）在设计阶段预警冲突 |
| 代码生成 | coding/references/ API 参考为主 | KKB A 系列（API 用法）补充 Ascend 特有约束；KKB E 系列（调优经验）提供条件匹配的代码片段 |
| 编译修复 | Conductor 分析 + coding 模式 3 | KKB compile_error 检索：错误类型匹配 + 修复知识单元（cache 命中率 72%） |
| 精度修复 | Conductor 分析 + coding 模式 3 | KKB precision_debug 检索：general_stability 经验优先（cache 命中率 78%） |
| 性能调优 | optimizer 30 个优化点按序检查 | KKB perf_tuning 检索：瓶颈→优化点映射，优先尝试高收益优化点 |
| **冲突时** | **template Layer 1 优先** | KKB 建议如与 template Layer 1 冲突，以 Layer 1 为准；KKB 矛盾提示仅标记"两 KKB 约束间冲突" |

### 3.4 接入后各阶段的 KKB 注入点

```
Phase 0: 参数确认
Phase 1: 任务构建 (triton-task-extractor)

Phase 2: 算法设计
  ├── Step 1: template 前置检查 (precheck.json)          ← cannbot 现有
  │     读取 .claude/template/{category}.md Layer 1 约束
  │
  └── Step 2: triton-op-designer 设计草图                ← cannbot 现有 + KKB 增强
        ├── [designer 自主检索] codegen_init (Step 0-2)
        │     获取: 缓存命中 / C1-C5 硬件约束 / 算子分类 OP / 瓶颈预判 B / 矛盾提示 X
        ├── [现有] 加载 template Layer 1/2 + 手写优化案例
        └── 产出: sketch (template Layer 1 硬性边界 + KKB C/B/X 补充验证)

Phase 3: 代码生成与验证 (迭代)
  ├── 3.1: triton-op-coding (初始生成)                    ← cannbot 现有 + KKB 增强
  │     ├── [coding 自主检索] codegen_init (Step 0-6 全量)
  │     │     获取: API 知识单元 A / 瓶颈预判 B / Top 5 调优经验 E / 策略 T / 骨架 S
  │     │     若 designer 已检索同类算子 → 缓存精确命中，零额外 context
  │     ├── [现有] 加载 template Layer 1/3 + API 参考
  │     └── 产出: 初始代码
  │
  ├── 3.2: AST 预检查
  │
  ├── 3.3: triton-op-verifier (verify.py)
  │
  └── 3.4: Conductor 分析与迭代修复                        ← cannbot 现有 + KKB 增强
        ├── [Conductor 错误分类] A/B/C 三类
        │
        ├── 编译失败 → coding 模式 3
        │     ├── [coding 自主检索] compile_error
        │     │     获取: 错误类型匹配 + 修复知识单元 (cache 命中率 72%)
        │     └── [现有] 修复代码
        │
        └── 精度失败 → coding 模式 4
              ├── [coding 自主检索] precision_debug
              │     获取: general_stability 经验 + 数值稳定性经验 (cache 命中率 78%)
              └── [现有] 修复代码

Phase 4: 性能优化与验证 (迭代)
  └── triton-latency-optimizer                            ← cannbot 现有 + KKB 增强
        ├── [optimizer 自主检索] perf_tuning (每轮优化前)
        │     获取: 瓶颈→优化点映射 / Top-3 调优经验 / 类别高频优化点
        ├── [现有] 按序检查 30 个优化点，优先尝试 KKB 指向的高收益点
        └── 产出: 优化后代码

Phase 5: 输出报告 (report.md + summary.json)               ← cannbot 现有

Phase 6: 知识库摄入与维护                                  ← 主 Agent 负责
  ├── F1-F5 事件日志写入 log.md
  ├── Source B/C/D 触发检查 → YAML 写入 → _index.yaml 更新
  └── 定期维护 (每 10 次 F3): 质量重评 + 冲突检测 + 缓存失效 + 覆盖统计

Phase 7: 会话导出 (session.jsonl + session.md)             ← cannbot 现有
```

### 3.5 子 Skill 检索的强制执行

每个子 Skill 在 SKILL.md 的对应流程步骤中，将 KKB 检索设为**前置必选步骤**（非可选），确保在实际动作前完成检索：

| 子 Skill | 修改位置 | 强制规则 |
|---------|---------|---------|
| designer | "知识加载规则" 章节，在加载 template 之前 | 调用 `codegen_init` (Step 0-2)，获取 C/OP/B/X 知识单元。检索失败或 skill 不可用时回退到现有路径 |
| coding | "知识加载规则" 章节，在加载 template 之前 | 调用 `codegen_init` (Step 0-6 全量)，获取 A/B/E/T/S 知识单元 |
| coding (模式 3) | 模式 3 修复流程，在错误分析之后、写代码之前 | 调用 `compile_error`，获取错误匹配 + 修复知识单元 |
| coding (模式 4) | 模式 4 调试流程，在精度分析之后、写代码之前 | 调用 `precision_debug`，获取 stability 经验 + 修复方案 |
| optimizer | 优化点检查之前，新增强制前置步骤 | 每轮调用 `perf_tuning`，获取瓶颈→优化点映射 + Top-3 经验 |

**回退保障**：每个子 Skill 均保留原有知识加载路径。若 `triton-knowledge-retrieval` skill 不可用、检索超时或返回空结果，自动回退到现有的 template + 参考文档路径，不阻塞算子生成。

**缓存联动**：designer → coding → optimizer 三级流水线共享 Jaccard 缓存。designer 首次检索写入缓存后，coding 的 `codegen_init` 检索精确命中（Jaccard=1.00），直接复用全部知识单元，context 节省 ~95%。

### 3.6 KKB 与 cannbot 知识冲突的优先级

当 KKB 检索结果与 cannbot 现有知识出现分歧时，以以下优先级为准：

```
1. cannbot template Layer 1        ← 硬性约束，绝对优先
2. cannbot npu-arch 硬件规格        ← 硬件事实，不可违背
3. KKB C 系列 [硬件约束]            ← 补充验证，不与 1/2 冲突时可参考
4. cannbot coding/optimizer 参考    ← Skill 级别最佳实践
5. KKB E/A/T/S 系列                 ← 补充建议，与 4 冲突时以 4 为准
6. KKB X 系列 [矛盾提示]            ← 仅标记 KKB 内部约束冲突，不影响 cannbot 逻辑
```

**简化为两条铁律**：
- `template Layer 1` 与 KKB 冲突 → **Layer 1 胜出**
- KKB 两条经验之间冲突 → X 系列预警，LLM 自行判断

### 3.7 知识摄入与进化（主 Agent 负责）

检索由子 Skill 自主完成，但**知识摄入（写入）仍由主 Agent 在 Phase 6 统一负责**：

```
Phase 6: 知识库摄入与维护 (主 Agent, CLAUDE.md)

  输入: 各子 Skill 的 session 日志中的 F1-F5 事件
  ├── F1 编译修复事件 → Source C (调试记录)
  ├── F2 精度修复事件 → Source C (调试记录)
  ├── F3 性能收敛事件 → Source B (成功经验) + Source D (性能洞察)
  ├── F4 缓存命中事件 → 命中率统计
  └── F5 用户纠正事件 → review/deprecated 状态迁移

  处理:
  ├── Source B/C/D 触发检查 → YAML 写入 → _index.yaml 更新
  ├── 定期维护 (F3 mod 10 == 0): 质量重评 + 冲突检测 + 缓存失效 + 覆盖统计
  └── 终端输出: KKB 知识摄入摘要
```

**职责分离**：
- **检索（读）** → 各子 Skill 自主触发，动作前完成，结果立即消费
- **摄入（写）** → 主 Agent Phase 6 统一处理，汇总全流程事件后批量写入

### 3.8 目录结构与修改文件清单

```
ops/triton-knowledge-retrieval/          ← KKB 增强层完整目录
├── SKILL.md                             ← Skill 入口 + 七步检索流程 + 相关子系统
├── log.md                               ← 反馈事件日志 (F1-F5 + INGESTION + MAINTENANCE)
├── references/                          ← 检索协议文档
│   ├── retrieval-workflow.md            ← 七步检索流程 + Step 0.5 问题分解
│   ├── cache-protocol.md
│   └── knowledge-unit-format.md
├── knowledge/                           ← 知识库内容
│   ├── cache/                           ← 查询级缓存 (~52 个)
│   ├── pages/                           ← Wiki 层 (hardware/api/ops/techniques/skeletons)
│   └── raw/                             ← 经验 + 开发指南 + 原始数据
├── retrieval/                           ← 检索引擎
│   ├── quality_scoring.md               ← 四维质量评分
│   └── problem_decomposer.md            ← 问题分解器 + 交叉关联引擎
├── builder/                             ← 知识摄入引擎
│   └── auto_ingestion.md                ← Source B/C/D 自动摄入
└── maintenance/                         ← 定期维护
    ├── periodic_tasks.md                ← 4 个维护任务
    └── integration_quickstart.md        ← 接入速查 + 事件协议
```

**cannbot 侧修改**（已执行）：

| 文件 | 改动 | 说明 |
|------|------|------|
| `ops/triton-op-designer/SKILL.md` | 知识加载规则新增前置步骤 | 在加载 template 前调用 `codegen_init` (Step 0-2)，获取 C/OP/B/X 知识单元 |
| `ops/triton-op-coding/SKILL.md` | 知识加载规则新增前置步骤 + 模式 3/4 新增检索 | 初始生成前调用 `codegen_init` (全量)；模式 3 调用 `compile_error`；模式 4 调用 `precision_debug` |
| `ops/triton-latency-optimizer/SKILL.md` | 优化点检查前新增前置步骤 | 每轮优化前调用 `perf_tuning`，将 Top-3 经验与 30 个优化点检查结合 |
| `plugins-official/triton-op-generator/CLAUDE.md` | 移除集中检索，保留 Phase 6 知识摄入 | 移除 Phase 2.0/3.4/4.0 的集中检索逻辑，Phase 6 知识摄入与维护保持不变 |

---

## 四、关键约束

| 约束 | 说明 |
|------|------|
| 增强而非替代 | KKB 是 cannbot 现有知识的增强层；template Layer 1 与 KKB 冲突时 Layer 1 胜出 |
| 检索归属 | 检索（读）由各子 Skill 自主触发（designer/coding/optimizer），在核心动作前完成；摄入（写）由主 Agent Phase 6 统一负责 |
| 检索失败不阻塞 | 子 Skill 检索失败或 skill 不可用时，自动回退到现有 template + 参考文档路径，不阻塞算子生成 |
| 日志归属 | F1-F5 事件在子 Skill 执行过程中产生，由主 Agent 在 Phase 6 统一写入 log.md；子 Skill 不直接写 log.md |
| session_id 命名 | 统一格式 `<op_name>_<dtype>_run<n>` |
| 知识库路径 | 固定使用 `ops/triton-knowledge-retrieval/`，cache/log.md/raw/experiences 跨任务累积 |
| 失败不阻塞 | cache 写入失败 / log.md append 失败告警但不阻塞算子生成 |
