# triton-ascend-knowledge

`triton-ascend-knowledge` 是面向 **Triton-Ascend DSL 算子开发**的知识库社区插件。它提供知识编译（Agent 轨迹/官方文档蒸馏）、治理、检索、Issue 材料整理和勘误流程；真实知识内容维护在独立知识库仓库中。

本插件与 AscendC 版 [`cannbot-knowledge`](../cannbot-knowledge/) 同构：双仓分离、OKF 卡片、三内容树 + 跨树共享层。差异在于生产层——按 **6 大知识分类**（共 18 个二级维度，详见下文「知识分类体系」）各设一个生产 skill，知识主要从 **Agent 轨迹**蒸馏"诊断-修改-验证"技能。

## 一、内容边界

本插件包含：

- `skills/`：知识编译、治理、检索相关 skills（1 编排器 + 6 分类生产 + query + lint + issue-report）。
- `skills/*/scripts/`：随对应 Skill 安装的唯一脚本入口；生产脚本统一接受 `--knowledge-root <path>`，`triton-knowledge-query` 还会读取安装时持久化的 root 配置。
- `examples/knowledge-base-skeleton/`：知识库目录骨架示例（占位，非真实知识）。
- `CONTRIBUTING.md`：外部贡献和勘误合入规则。

本插件不包含：

- 真实 `reference/`、`ops/`、`runbooks/` 知识正文。
- 真实 `graph/` 判定缓存和 `log/` 审计日志。
- 私有轨迹、密钥、内网路径或不可公开日志。

## 二、知识分类体系（6 大一级分类 / 18 个二级维度）

知识库全集整理为 **6 大一级分类、18 个二级维度**。该体系既覆盖硬件底层的物理约束，又契合 Agent 从轨迹（Trajectory）中抽取"诊断-修改-验证"技能的需求。每个一级分类对应一个生产 skill，知识正文落在知识库 `runbooks/` 的同名义目录下（见下文「知识库内容树」）。

```
Agent4Kernel (Triton-Ascend) 优化知识库体系
                                                  │
 ┌──────────────────┬──────────────────┬──────────┴───────┬──────────────────┬──────────────────┐
 │                  │                  │                  │                  │                  │
▼                  ▼                  ▼                  ▼                  ▼                  ▼
1. 内存与切块     2. 计算与指令      3. 算法与数值      4. 规范与移植      5. 调优空间设计    6. Agent 元技能
(Memory & SRAM)   (Compute Engine)   (Algorithmic)      (Portability)      (Autotuning)       (Meta-Skills)
```

| 一级分类 (Category) | 二级维度 (Dimension) | 核心知识/Skill 提取点与示例 | 硬件与系统映射依据 |
| --- | --- | --- | --- |
| **1. 内存与切块** *(Memory & SRAM Management)* | **1.1 Memory Access Efficiency** | GM→UB 连续访存 Coalescing、Burst 搬运合并、UB 内存复用 | 减少高延迟 HBM/GM 访问 |
| | **1.2 Tiling & SRAM Budgeting** | `BLOCK_M/N/K` 自适应划分公式、SRAM/UB 容量溢出约束求解、Tail Tile 余数块处理 | UB (Unified Buffer) 物理空间极度受限，片上 SRAM 资源约束 Tiling 策略 |
| | **1.3 Data Layout & Format Transformation** | ND ↔ NZ 格式转换消除、Transposed Load、Stride 计算优化 | Cube 单元强制依赖 NZ / Fractal 张量排布 |
| | **1.4 Hardware Memory Alignment** | **32-Byte**（VV 场景）与 **512-Byte**（CV 场景）内存对齐、自动 Padding、Sub-byte/FP8 Pack & Unpack | Ascend 硬件 Direct Memory Access 硬性校验 |
| **2. 计算与流水** *(Compute Engine & Execution)* | **2.1 AICore Compute Unit Utilization** | Vector/Cube 任务分工、Mask/Repeat 优化、Broadcast 消除、**Scalar 标量开销削减与高延迟 div/mod 索引化简** | 兼顾 Vector、Cube 与算力较弱的 Scalar 单元 |
| | **2.2 Compute & DMA Overlap** | Double/Triple Buffering 实现、Async Copy 异步搬运、**细粒度 Event Sync 替代粗粒度 PipeBarrier** | DMA 搬运与 Vector/Cube 计算的三级流水掩盖 |
| | **2.3 Multi-Core Parallelism & Load Balancing** | Grid 设计 (1D/2D/3D)、Tail Core 空闲均分策略、**coreDim ≤ 65535 限制规避** | 20~48 个 AICore 间的 Task 调度与并发限制 |
| | **2.4 Fusion Intelligence** | Element-wise / Reduction / Transpose 融合进 Reshape/Matmul 的 Prologue/Epilogue 融合模式 | 减少中间 Tensor 写回 GM 的 Round-trip 开销 |
| **3. 算法与数值** *(Algorithmic & Numerics)* | **3.1 Algorithmic Complexity Optimization** | Tree Reduction 树形归约、Online Softmax、FlashAttention 分块重构 | 降低计算复杂度，适合片上单 Pass 计算 |
| | **3.2 Precision Strategy & Mixed Precision** | FP32 Accumulator 提升、BF16/FP8 选择策略、Cast 与 Compute 算子融合 | 兼顾 Ascend 原生硬件数据类型与精度要求 |
| | **3.3 Numerical Stability** | Exponent Clamp、防止 Overflow/Underflow、Log-Sum-Exp 变换 | 避免低精度（FP16/BF16）下的数值溢出风险 |
| | **3.4 Cross-Block Reduction & Communication** | Two-pass Reduction vs Online Reduction、Atomic 边界控制、Workspace Tensor 复用 | 弥补 Ascend 缺少 CUDA `__shfl_sync` Warp 同步机制 |
| **4. 规范与移植** *(Portability & System Bounds)* | **4.1 Correctness & Portability** | 动态 Shape 泛化支持、跨 Triton-Ascend 版本 API 兼容性 | 保证代码在不同输入规模下的鲁棒性 |
| | **4.2 Compiler & Backend Constraint Handling** | `tl.multiple_of`/`tl.max_contiguous` 提示插入、Triton-Ascend 临时 Bug 的 Workaround、环境变量开关 | 配合并引导 Triton 编译器生成高效 CCEC/IR 代码 |
| | **4.3 Cross-Hardware Adaptation** | 910B / 910C / 310P 芯片代际能力探测与差异化分发 | 不同 NPU 芯片在 UB 容量与 Vector 宽度上的差异 |
| **5. 调优空间设计** *(Autotuning Design)* | **5.1 Autotuning Search Space Design** | 候选 `BLOCK_SIZE` 剪枝、`num_stages` 搜索策略、autotune key 关联设计 | 避免盲目暴力搜索导致编译耗时爆炸 |
| **6. Agent 元技能** *(Meta-Action Strategies)* | **6.1 Bottleneck Diagnosis Tree** | Profiling 驱动的 Memory-bound vs Compute-bound 判定链、Roofline 模型定位 | 引导 Agent 快速精准定位瓶颈 |
| | **6.2 Compilation & Accuracy Repair Strategy** | 编译报错修复链（对齐/越界）、精度降级时的自动 Step-up 提升策略 | 提高 Agent 自动 Debug 与 self-correction 成功率 |
| | **6.3 Trajectory-Level Action Sequence** | 多步优化动作组合规则（如：先调整 Tiling 解决 OOM → 再开 Double Buffering 掩盖 Latency → 最后加 Align Hints） | 提取 Agent 轨迹中的高阶决策时序模式 |

## 三、知识库内容树（独立仓库）

```
<知识库根>/
├── reference/            # triton-ascend API/文档蒸馏卡
├── ops/                  # 算子案例卡（成功探索 → 设计 wiki）
├── runbooks/             # 6 大分类优化点库（OPT-*/AP-*/CT-*，单一共享、增量合并）
│   ├── memory-sram/          # 分类1：访存效率 / Tiling·SRAM预算 / ND↔NZ / 32B·512B对齐
│   ├── compute-pipeline/     # 分类2：单元利用·Scalar削减 / 流水掩盖 / 多核负载 / 融合
│   ├── algorithm-numerics/   # 分类3：算法重构 / 混合精度 / 数值稳定 / 跨Block归约
│   ├── portability/          # 分类4：动态shape / 编译器约束·hints / 910B·910C·310P
│   ├── autotuning/           # 分类5：搜索空间剪枝 / num_stages / autotune key
│   └── meta-skills/          # 分类6：瓶颈诊断树 / 修复链 / 轨迹动作时序
├── graph/  search/  log/     # 跨树共享层
├── index.md  SPEC-*.md       # 导航与规范
└── metadata/                 # type_map / tag_registry / aliases
```

## 四、知识卡 frontmatter 约定

知识卡采用 OKF v1 路由字段，以下字段**必填非空**：

```yaml
schema_version: okf.v1
kind: concept            # 卡片种类
type: <type_map 注册类型>
source_family: trajectory | upstream-doc | golden | other
title: ...
description: ...
tags: [...]
created_at: YYYY-MM-DD
updated_at: YYYY-MM-DD
resource: <== primary.url>
sources:                 # ≥1，有且仅一 primary
  - url: ...
    role: primary
dimension: "1.2"         # runbook 条目卡另需：二级维度编号
```

字段含义：

| 字段 | 含义 |
| --- | --- |
| `schema_version` | OKF 规范版本，固定 `okf.v1`；解析器据此选择路由规则 |
| `kind` | 卡片种类（如 `concept`），区分知识卡与索引、日志等非知识文档 |
| `type` | 卡片类型，须在 `metadata/type_map` 中注册，用于按类型检索与治理 |
| `source_family` | 知识来源类别，四选一：`trajectory`（Agent 算子生成轨迹）/ `upstream-doc`（官方文档）/ `golden`（golden 参考实现）/ `other`（社区贡献、Issue、调试记录等） |
| `title` | 卡片标题，人读与检索展示用 |
| `description` | 一句话摘要，index 与检索结果展示用 |
| `tags` | 主题标签，须在 `metadata/tag_registry` 注册；主题层级进 tags，不靠目录嵌套（扁平目录原则） |
| `created_at` / `updated_at` | 创建 / 最近更新日期（`YYYY-MM-DD`），stale 治理依据 |
| `resource` | 主溯源指针，**恒等于** primary source 的 `url` |
| `sources` | 溯源列表：≥1 条且有且仅一条 `role: primary`；轨迹源带仓内路径+commit，文档源带 @sha URL，golden 源带固定 commit+源码路径，其他来源须附可核证据 |
| `dimension` | 二级维度编号（如 `"1.2"`），仅 runbook 条目卡必填，对齐「二、知识分类体系」 |

约束（与 RFC「六、知识卡 frontmatter 约定」一致）：

- 不维护静态 source 总表；source pin 在内容根 `index.md` frontmatter（= watermark，唯一"已落地"开关）。
- `metadata/` 存轻量 registry（type_map / tag_registry / aliases），不替代卡片级 frontmatter。
- 矛盾绝不静默覆盖：并存 + 日期/来源 + `> ⚠️ 矛盾标记`；每卡自包含，正文禁摄入/变更口吻（"本次批次 / 留待后续"只进 `log/`）。

## 五、快速使用

```bash
bash init.sh project claude /path/to/project --profile consumer --knowledge-root /path/to/knowledge-base
python3 skills/triton-knowledge-query/scripts/knowledge_query.py discover
python3 skills/triton-knowledge-query/scripts/knowledge_query.py search --query "BLOCK_SIZE UB 溢出"
```

安装 profile：

- `consumer`：只安装 `triton-knowledge-query`，用于只读检索和知识消费。
- `issue`：只安装 `triton-knowledge-issue-report`，用于提交 Issue、整理 `needs-info` 和复现附件。
- `contributor` / `all`：安装全部 10 个 skill，用于知识编译、治理、检索、Issue 和贡献门禁。
- `--skills triton-knowledge-query,triton-knowledge-issue-report`：高级用法，精确安装指定 skill 子集。

安装时传入 `--knowledge-root /path/to/knowledge-base` 后，插件会把 root 持久化为：

- `TRITON_ASCEND_KNOWLEDGE_ROOT`
- `TRITON_ASCEND_KNOWLEDGE_ROOTS`
- `OKF_KNOWLEDGE_ROOT`
- `OKF_KNOWLEDGE_ROOTS`

`skills/triton-knowledge-query/scripts/knowledge_query.py` 会按以下顺序解析上述变量（优先级从高到低）：

1. `--knowledge-root` / `--knowledge-roots` 参数
2. `TRITON_ASCEND_KNOWLEDGE_ROOT`、`TRITON_ASCEND_KNOWLEDGE_ROOTS`、`OKF_KNOWLEDGE_ROOT`、`OKF_KNOWLEDGE_ROOTS`
3. `~/.config/cannbot/knowledge.env`
4. 有限目录结构自动探测

更多流程见 [quickstart.md](./quickstart.md) 和 [CONTRIBUTING.md](./CONTRIBUTING.md)。
