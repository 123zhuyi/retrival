---
description: triton-optimizer-multi-agent 的 Fusion-Search 算法 Per-Round Worker 规范。每轮都以基因编码驱动路线构造——基因 = 知识编码（优化技术），所有路线共用 B2 选出的唯一融合方案。Round 1: B1 原子分解 → B2 自动融合分析 → C0 基因编码空间设计 → D0 并行多路线搜索 → 构建初始基因池。Round 2+: 基因选择/交叉/变异 → profiling 驱动基因新增 → 并行派发进化路线 → 更新基因适应度。与 naive / mcts 是对等的独立算法，每轮对外仍然只产生一个 round_result.json。
---

> **本文件是 Fusion-Search 算法子 Per-Round Worker Agent 的权威执行规范。**
> - 由主 Orchestrator Agent 通过 `dispatch_round.py` 每轮启动 1 个 fusion-search 子 CLI。
> - 子 Agent 上下文**仅包含**：本文件、`task_manifest.json`、当前 `work_dir`、`baseline_dir`、标准 skill/reference。
> - 子 Agent **禁止**读取历史 `round_index.json`、禁止做跨 round 停止/继续判定、禁止更新 template、禁止修改 `.triton-agent/state-{op_name}-{algorithm}-{run_tag}.json`。
> - 子 Agent 内部通过 `dispatch_parallel.py` 并行启动多个 **route 子 CLI**，每个 route 子 CLI 使用 `.claude/references/algorithms/fusion-search/route.md` 作为参考规范。

---

## 一、Fusion-Search Loop 全景

```
Orchestrator 每轮派发 1 个 fusion-search 子 Agent
              │
              ▼
┌──────────────────────────────────────────────────────────────┐
│  Fusion-Search Worker (本轮)                                  │
│                                                              │
│  ┌─ Round 1 (基因编码驱动的初始搜索) ─────────────────────┐  │
│  │ Phase 0: 环境验证 + baseline 冻结                       │  │
│  │ Phase 1 (B1): 计算原子分解 → atom_decomposition.md      │  │
│  │ Phase 2 (B2): 融合方案分析 → 选唯一最优方案              │  │
│  │              → fusion_plan.json（只有一个）              │  │
│  │ Phase 2.5 (C0): 基因编码空间设计                         │  │
│  │   ┌─ 从 develop_guide / expert_experiences /              │  │
│  │   │  加载知识编码 → 构建基因池                             │  │
│  │   └─ 路线 = 单一融合方案 × 不同基因组合:                  │  │
│  │      所有路线的融合方案相同，只有基因组合不同               │  │
│  │ Phase 3 (D0): dispatch_parallel.py 并行派发所有路线       │  │
│  │ Phase 4: 收集结果 → 选最优 → 基因适应度赋值              │  │
│  │ Phase 5-7: 报告 + 导出 + round_result.json               │  │
│  └────────────────────────────────────────────────────────┘  │
│                              │                               │
│                              ▼                               │
│  ┌─ Round 2+ (遗传/进化循环) ──────────────────────────────┐ │
│  │ Step G0: 加载基因池 (gene_pool.json) + profiling 数据    │ │
│  │          + 加载 fusion_plan.json（融合方案不变）           │ │
│  │ Step G1: 基因选择 (fitness-based selection + 精英保留)   │ │
│  │ Step G2: 交叉 (crossover) — 重组成功基因                  │ │
│  │ Step G3: 变异 (mutation) — 扰动基因参数                   │ │
│  │ Step G4: Profiling 驱动基因新增                           │ │
│  │ Step G5: 进化后基因组合 × 融合方案 → dispatch_parallel     │ │
│  │ Step G6: 收集结果 → 更新基因适应度 → 选最优               │ │
│  │ Step G7: 报告 + 导出 + round_result.json                  │ │
│  └──────────────────────────────────────────────────────────┘ │
│                              │                               │
│                              ▼ 本轮最优 → 成为下一轮 baseline  │
│                        Orchestrator C1/C2 判定                │
└──────────────────────────────────────────────────────────────┘
```

**核心循环逻辑**：

- **Round 1**：baseline = 原始代码 → B1 原子分解 → B2 融合方案分析（**只选 1 个最优方案**）→ **C0 基因编码空间设计**（从 develop_guide / expert_experiences 加载知识编码作为基因）→ 同一融合方案 × 不同基因组合 → 生成多条路线 → D0 并行派发 → 实测性能赋值基因 fitness → 选出最优

- **Round 2+**：baseline = 上一轮最优 → 从基因池中做选择/交叉/变异 → profiling 驱动新增基因 → 进化后的基因组合 × **同一融合方案**（不变）→ 生成路线 → 并行派发 → 更新基因适应度 → 选最优

- **...** 循环直到 C1/C2 触发停止

**核心公式**：

```
路线 = 融合方案 × 基因组
       ↑            ↑
   B2 选出         基因池中的知识编码
  只有1个，不变      每轮进化

所有路线的融合方案相同，差异仅在于基因组合
```

- 融合方案由 B2 在 Round 1 一次性选出，后续 round 中不变
- 基因（知识编码）来自外部知识源，参与选择/交叉/变异，每轮进化
- 路线数 = 基因组合数，不是融合方案数

**为什么 Round 2+ 不重跑 B1/B2**：
- B2 选出的融合方案只有一个，后续无需重新选择
- 后续 round 的核心问题是"用哪些优化技术来实现这个融合"，即基因进化
- Profiling 驱动的新基因发现是创新来源

---

## 二、子 Agent 边界

你是 **Fusion-Search Worker**。你的任务：

**Round 1**：
1. 读取 `task_manifest.json`。
2. 执行 Phase 0 → Phase 1 (B1) → Phase 2 (B2) → Phase 2.5 (C0) → Phase 3 (D0) → Phase 4 → Phase 5 → Phase 6 → Phase 7。
3. **B2**：选出唯一最优融合方案。
4. **C0**：从 develop_guide / expert_experiences 加载知识编码构建基因池。所有路线共用同一融合方案，差异仅在基因组合。
5. 在 Phase 3 并行启动所有 route 子 CLI。
6. 收集结果，为基因赋初始 fitness，选出最优。
7. 写入 `round_result.json`，退出。

**Round 2+**：
1. 读取 `task_manifest.json`，确认 `baseline_dir`。
2. 执行 Phase 0（快速检查）→ Step G0-G4（基因进化）→ Step G5（并行派发）→ Step G6（收集+更新）→ Phase 5-7。
3. 写入 `round_result.json`，退出。

**禁止行为**：
- 读取 `triton_ascend_output/{op_name}-{algorithm}-{run_tag}/round_index.json` 以外的历史信息。
- 执行 C1/C2/C3/D4 停止/继续判定（交给主 Orchestrator）。
- 调用 `transition_next_round.py`。
- 更新 `.claude/template/{category}.md`。
- 修改 `.triton-agent/state-{op_name}-{algorithm}-{run_tag}.json`。
- **在 B2 阶段或基因选择阶段向用户提问或等待用户确认**（全自动）。

---

## 三、Phase 0：参数确认 + 基线就绪检查

### 3.1 读取 manifest

读取 `task_manifest.json`，获取：
- `op_name`, `round_index`, `work_dir`, `algorithm`（应为 `"fusion-search"`）
- `baseline_dir`, `round_strategy`, `analysis_policy`, `hypothesis`, `direction`, `evidence_sources`
- `lineage_summary`（Orchestrator 提供的历史摘要， enriched 后包含 `design_summary`、`key_fixes` 等）
- `experience_file`（算子级私有经验文件，可读可追加）
- `parent_round_journal`（若提供，父 round 详细日志，只读）
- `mode`, `input_files`, `config`
- `algorithm_params`（当前算法 `fusion-search` 的参数，如 `max_routes`、`crossover_rate` 等）
- `genetic_config`（Round 2+ 时由 Orchestrator 注入，包含上轮基因池摘要）

### 3.2 工作目录检查

1. 确认 `work_dir` 存在。
2. 确认 `{work_dir}/{op_name}_generated.py` 存在：
   - **Round 1**：`baseline_dir` 为空或首次创建。Phase 0 做环境验证（`npu-smi info` 检测 arch、验证 `torch_npu` 可用等）并调用 `freeze_baseline.py`。
   - **Round > 1**：Phase 0 **只做快速检查**——确认文件存在即可，不重复环境验证和 freeze。
3. 快速读取 `{work_dir}/{op_name}_generated.py` 确认代码可解析。

### 3.3 硬件架构

Round 1 通过 `npu-smi info` 检测，失败则用默认 `ascend910b1`。Round > 1 跳过。

### 3.4 基线冻结

仅 Round 1 执行 `freeze_baseline.py`。Round > 1 时跳过。

### 3.5 Round 类型判断

- `round_index == 1` → 执行完整 Phase 1-7（含 B1/B2）
- `round_index > 1` → 跳过 Phase 1/2，直接进入 Step G0-G7（遗传进化循环）

---

## 四、Phase 1 (B1)：计算原子分解（仅 Round 1）

### 4.1 目标

将 PyTorch 基线中的 forward() 计算图拆解为最细粒度的计算原子序列。仅 Round 1 执行。

### 4.2 执行步骤

1. 读取 `{work_dir}/{op_name}.py` 中的 `Model.forward()` 方法。
2. 逐行分析计算图，识别每个独立计算操作。
3. 输出原子序列表格：

```text
| atom_id | op_type    | formula              | inputs         | outputs        | numerical_risk |
|---------|------------|----------------------|----------------|----------------|----------------|
| A1      | load       | X = input[..., :]    | input          | X              | low            |
| A2      | reduce     | mean = sum(X)/N      | X              | mean           | low            |
| A3      | broadcast  | X_centered = X - mean| X, mean        | X_centered     | low            |
| ...     | ...        | ...                  | ...            | ...            | ...            |
```

4. 识别**可融合的相邻原子对/组**，标注 eliminated_intermediates。

### 4.3 产物

`{work_dir}/atom_decomposition.md` — 后续 round 中作为只读参考。

---

## 五、Phase 2 (B2)：融合方案分析 → 选唯一最优（仅 Round 1）

### 5.1 目标

基于 Phase 1 的原子分解结果，枚举所有等价融合方案，自动评分，**只选出得分最高的 1 个方案**。

**本阶段全自动，不等待用户确认。仅 Round 1 执行。**

### 5.2 融合方案生成规则

1. **枚举候选融合组**：基于相邻可融合原子对，生成所有合法的融合组合。
2. **过滤不可行方案**：排除存在 Triton-Ascend 已知不可行模式的方案。
3. **评分维度**：

| 维度 | 权重 | 说明 |
|------|------|------|
| `expected_benefit` | 40% | 消除的中间张量数 x 预估内存带宽节省 |
| `feasibility` | 30% | 是否存在已知 Triton-Ascend 实现模式 |
| `risk_analysis` | 20% | 数值稳定性风险（低->高分） |
| `complexity` | 10% | 实现复杂度（低->高分，偏好简单方案） |

4. **自动选择**：
   - 过滤 `feasibility` 为 "low" 的方案
   - 过滤 `expected_benefit` < `min_expected_benefit` 的方案
   - 按总分降序排列
   - **只选得分最高的 1 个**。所有路线共用同一融合方案，区别仅在于基因组合不同
   - 若 `enable_math_transforms == true` 且该方案存在数学等价变换，记录变换变种以备后续轮次尝试

### 5.3 方案输出格式

`{work_dir}/fusion_plan.json`：

```json
{
  "fusion_id": "F1",
  "fused_atoms": ["A2", "A3", "A4"],
  "eliminated_intermediates": ["mean", "X_centered"],
  "expected_benefit": 0.35,
  "feasibility": "high",
  "risk_analysis": "low",
  "complexity": "medium",
  "total_score": 0.85,
  "description": "融合 reduce + broadcast + elementwise 为单一 kernel",
  "math_transform": null,
  "math_transform_variant": null
}
```

**融合方案不是基因，不参与遗传进化。所有路线的唯一差异是基因组合。**

---

## 五-二、Phase 2.5 (C0)：基因编码空间设计（仅 Round 1）

### 5-2.1 目标

从外部知识源加载知识编码构建基因池。所有路线共用同一融合方案，差异仅在基因组合。

**基因 = 知识编码（优化技术）**。每个基因是一个可追踪到代码动作的优化项（tiling 策略、内存访问模式、计算原语选择、向量化方式等）。

**路线 = 融合方案 x 基因组合**。所有路线的融合方案相同。

### 5-2.2 加载知识编码 -> 构建基因池

按优先级从知识源加载：

| 优先级 | 知识源 | 示例基因 |
|--------|--------|---------|
| 1 (最高) | `develop_guide` | "使用 tl.dot 利用 TensorCore"、"沿 M 维分块 block_size=128" |
| 2 | `expert_experiences` | "对 reduction 轴使用 multiple-pipeline" |

对每个加载的知识项，构造一个基因：

```json
{
  "gene_id": "G001",
  "ku_id": "K1",
  "category": "tiling",
  "description": "沿 M 维度分块，块大小 128",
  "source": "develop_guide",
  "source_priority": 1,
  "code_target": "grid / block size 参数",
  "params": {"block_size_m": 128, "block_size_n": 64},
  "conflicts_with": [],
  "status": "active",
  "fitness": null
}
```

**基因兼容性检查**：标记互斥基因对，同一路线中不允许互斥基因共存。

### 5-2.3 路线生成

所有路线的融合方案相同（来自 B2），差异仅在基因组合：

**路线生成策略**：
1. **多样性原则**：生成多种不同的基因组合
2. **总路线数控制**：总数 <= `max_routes`（默认 3）。超出时优先选取高优先级知识源（develop_guide）基因组成的路线
3. **全覆盖原则**：保证不同类别（tiling/memory/compute/pipeline）的基因都有机会被验证

**示例**（max_routes=3，融合方案 F1）：

```
基因池: [G001(tiling_128), G002(tl.dot), G003(tiling_256), G004(pipeline)]

F1 x [G001, G002]              -> route-1  (基础: tiling+memory)
F1 x [G003, G002, G004]        -> route-2  (变种: alt_tiling+memory+pipeline)
F1 x [G001, G004]              -> route-3  (变种: tiling+pipeline)
```

**初始基因池**（fitness 均为 null）：

`{work_dir}/gene_pool.json`：

```json
{
  "generation": 1,
  "fusion_plan": {"fusion_id": "F1", "description": "融合 reduce + broadcast + eltwise"},
  "genes": [
    {"gene_id": "G001", "category": "tiling",    "source": "develop_guide",      "params": {"block_size_m": 128}, "fitness": null},
    {"gene_id": "G002", "category": "memory",     "source": "develop_guide",      "params": {}, "fitness": null},
    {"gene_id": "G003", "category": "tiling",     "source": "expert_experiences", "params": {"block_size_m": 256}, "fitness": null},
    {"gene_id": "G004", "category": "pipeline",   "source": "expert_experiences", "params": {"num_stages": 3}, "fitness": null}
  ],
  "route_assignments": [
    {"route_id": "route-1", "gene_ids": ["G001", "G002"]},
    {"route_id": "route-2", "gene_ids": ["G003", "G002", "G004"]},
    {"route_id": "route-3", "gene_ids": ["G001", "G004"]}
  ]
}
```

### 5-2.4 产物

- `{work_dir}/gene_pool.json` — 初始基因池（fitness 待 D0 后赋值）
- `{work_dir}/fusion_plan.json` — 唯一融合方案

---

## 六、Phase 3 (D0)：并行多路线派发

### 6.1 为每条路线生成 task_manifest.json

```json
{
  "op_name": "layer_norm",
  "round_index": 1,
  "work_dir": "triton_ascend_output/layer_norm/opt-round-1/route-1",
  "baseline_dir": "triton_ascend_output/layer_norm/opt-round-1",
  "algorithm": "fusion-search-route",
  "algorithm_reference": ".claude/references/algorithms/fusion-search/route.md",
  "route_id": "route-1",
  "round_strategy": "exploration",
  "analysis_policy": "pattern_entry",
  "hypothesis": "F1 x G001(tiling_128)+G002(tl.dot)",
  "direction": "fusion: F1 x tiling+memory",
  "evidence_sources": ["atom_decomposition", "fusion_plan", "gene_pool"],
  "mode": "A",
  "config": {"target_speedup": 5, "max_rounds": 50},
  "fusion_plan": {
    "fusion_id": "F1",
    "fused_atoms": ["A2", "A3", "A4"],
    "eliminated_intermediates": ["mean", "X_centered"],
    "description": "融合 reduce + broadcast + elementwise 为单一 kernel"
  },
  "optimization_genes": [
    {"gene_id": "G001", "category": "tiling", "params": {"block_size_m": 128}},
    {"gene_id": "G002", "category": "memory", "params": {}}
  ],
  "genetic_metadata": {
    "generation": 1,
    "parent_routes": [],
    "gene_ids": ["G001", "G002"],
    "route_type": "initial_exploration"
  }
}
```

- `fusion_plan`：唯一融合方案（所有 route 相同）
- `optimization_genes`：基因列表（各 route 不同）
- `genetic_metadata.gene_ids`：只包含基因 ID

### 6.2 调用 dispatch_parallel.py

```bash
python3 skills/triton-agent-loop/scripts/dispatch_parallel.py \
    --manifests route-1/.task_manifest.json route-2/.task_manifest.json route-3/.task_manifest.json \
    --timeout {route_timeout}
```

### 6.3 约束

- 所有 route 同时启动（ThreadPoolExecutor 并行）
- 每个 route 子 CLI 完全隔离，不共享上下文
- 某 route 超时/失败不影响其他 route

---

## 七、Phase 4：结果收集 + 基因适应度赋值

### 7.1-7.3

同上（读取 parallel_result.json，筛除失败 route，选 speedup 最高者，复制最优产物到 work_dir）。

### 7.4 基因适应度赋值

用实测性能为基因赋初始 fitness。**融合方案不计入基因池**。

```
fitness(gene) = max(route.speedup for route in routes_containing_gene)
```

辅助指标：

| 指标 | 计算方式 |
|------|---------|
| `fitness` | `max(route.speedup)` |
| `stability` | `1 - std(route.speedups) / mean(route.speedups)` |

更新 `gene_pool.json`：

```json
{
  "generation": 1,
  "fusion_plan": {"fusion_id": "F1"},
  "genes": [
    {"gene_id": "G001", "category": "tiling",   "fitness": 1.85, "stability": 0.90},
    {"gene_id": "G002", "category": "memory",    "fitness": 1.85, "stability": 0.85},
    {"gene_id": "G003", "category": "tiling",    "fitness": 1.23, "stability": 0.70},
    {"gene_id": "G004", "category": "pipeline",  "fitness": 1.78, "stability": 0.80}
  ],
  "route_results": [
    {"route_id": "route-1", "gene_ids": ["G001","G002"],       "speedup": 1.85, "status": "success"},
    {"route_id": "route-2", "gene_ids": ["G003","G002","G004"], "speedup": 1.78, "status": "success"},
    {"route_id": "route-3", "gene_ids": ["G001","G004"],       "speedup": 1.65, "status": "success"}
  ],
  "best_route": "route-1",
  "best_speedup": 1.85
}
```

### 7.5 输出对比摘要

```text
## Fusion Route Comparison (Generation 1)

All routes share fusion plan: F1 (reduce+broadcast+eltwise)

| Route | Genes | Internal Rounds | task_duration_us | speedup | Converged | Status |
|-------|-------|----------------|-----------------|---------|-----------|--------|
| route-1 | G001(tiling_128), G002(tl.dot) | 3 | 1234 | 1.85 | yes | success |
| route-2 | G003(tiling_256), G002(tl.dot), G004(pipeline) | 3 | 1320 | 1.78 | yes | success |
| route-3 | G001(tiling_128), G004(pipeline) | 2 | 1420 | 1.65 | yes | success |

Best: route-1, speedup=1.85
Gene pool: 4 genes, fitness assigned from profiling
```

---

## 八、Round 2+：遗传/进化循环 (Step G0-G7)

> Round 2+ **不重跑 B1/B2**。融合方案不变，从基因池出发进化。

### Step G0：加载

1. 从 `baseline_dir` 读取 `gene_pool.json`（含 fitness）
2. 从 `baseline_dir` 读取 `fusion_plan.json`（融合方案不变）
3. 读取上轮各 route 的 profiling 数据（`aiv_scalar_ratio`, `aiv_mte2_ratio`, `cube_utilization`）

### Step G1：基因选择

基于适应度选择父代基因组。**所有路线共用同一融合方案**。

1. **精英保留**：上轮最优 route 的基因组合直接存活到 1 条新路线
2. **适应度比例选择**：以 `fitness / Σfitness` 为概率抽取基因
3. **约束**：基因数 2-5 个，互斥基因不共存

### Step G2：交叉 (crossover_rate=0.7)

1. **单点交叉**：两条父代基因列表在随机位置切分后交换尾部
2. **均匀交叉**：每个基因以 0.5 概率来自父代 A 或 B
3. **参数交叉**：同一基因参数不同时交叉参数值

### Step G3：变异 (mutation_rate=0.15)

1. **参数变异**：`block_size 128->256` 等（+-50%）
2. **基因替换/删除/新增**：低概率操作
3. `develop_guide` 来源基因变异率减半

### Step G4：Profiling 驱动基因新增

从 msprof 瓶颈信号发现新基因：

| 信号 | 阈值 | 新基因类别 |
|------|------|-----------|
| `aiv_scalar_ratio > 0.3` | 标量化 | `vectorization` |
| `aiv_mte2_ratio > 0.5` | 内存瓶颈 | `memory` |
| `cube_utilization < 30%` | 计算空闲 | `compute` |
| 连续 2 轮无改善 | 收敛停滞 | `structural` |

新基因标记 `status: candidate, fitness: null`，分配探索路线验证。

### Step G5：生成进化路线 + 并行派发

```
route-1 (elite):      F1 x [上轮最优基因不变]
route-2 (crossover):  F1 x [交叉变异产物]
route-3 (explore):    F1 x [上轮优质基因 + 新候选基因*]
                                                    * = candidate
```

所有路线的 `fusion_plan` 相同。调用 `dispatch_parallel.py`。

### Step G6：收集结果 + 更新适应度

- `new_fitness = 0.3 x old_fitness + 0.7 x route_speedup`（平滑更新）
- 候选基因：有提升 -> active，无提升 -> discarded
- 更新 `gene_pool.json`，选最优

```text
## Genetic Evolution Summary (Generation 2)

All routes share fusion plan: F1

| Route | Type | Genes | Speedup | vs Parent |
|-------|------|-------|---------|-----------|
| route-1 | elite | G001,G002 | 1.92 | +3.8% |
| route-2 | crossover | G001(256),G002,G003 | 1.81 | -- |
| route-3 | exploration | G001,G002,G007* | 1.88 | new gene |

Gene pool: 5 genes (+1 new: G007 vectorization from aiv_scalar=0.35)
Best: route-1, speedup=1.92
```

### Step G7：报告 + 会话导出 + round_result.json

同 Phase 5-7。

---

## 九、Phase 5：输出报告

### 9.1 最终代码

从最优 route 复制：`{work_dir}/{op_name}_generated.py`

### 9.2 写入 report.md

包含：基本信息、B1/B2 摘要（Round1）、基因池状态（Round2+）、路线对比表、最终结果。

### 9.3 写入 `round_journal.md`

子 agent 必须在 `{work_dir}/round_journal.md` 中记录本轮详细过程：

- **Round 1**：B1 原子分解思路、B2 融合分析决策、为何选择最终融合方案、各 route 设计差异。
- **Round > 1**：上轮基因池摘要、本轮基因选择/交叉/变异/新增逻辑、各 route 进化方向、关键修复与瓶颈。
- 每个 route 的简要 iter/opt_iter 摘要（尝试了哪些参数/融合组合、修复了什么问题、最终取舍）。

建议格式：

```markdown
# Round {N} Journal

## Design / Evolution Rationale
- Round type: initial_discovery | genetic_evolution
- Key decision: ...

## Route Iteration Summary
- route-1: autotune over F1; iter_count=3; key fix: pipeline stage scheduling
- route-2: F2 with math transform; iter_count=2; abandoned due to precision error
- route-3: F1 + double buffer; opt_iter_count=2; best speedup 1.85x

## Final Outcome
- best_route: route-1
- best_speedup: 1.85x
- bottlenecks_observed: ["..."]
```

### 9.4 写入 summary.json

```json
{
  "round_index": 1,
  "algorithm": "fusion-search",
  "best_speedup": 1.85,
  "best_task_duration_us": 1234.5,
  "success": true,
  "fusion_search": {
    "phase": "initial_discovery",
    "generation": 1,
    "selected_fusion_id": "F1",
    "gene_count": 4,
    "selected_routes": 3,
    "best_route": "route-1"
  },
  "gene_pool": {
    "generation": 1,
    "total_genes": 4,
    "best_fitness": 1.85
  }
}
```

---

## 十、Phase 6：会话导出

```bash
MY_JSONL=$(grep -l "{work_dir}" /root/.claude/projects/*/*.jsonl 2>/dev/null | head -1)
if [ -n "$MY_JSONL" ]; then
  cp "$MY_JSONL" {work_dir}/session.jsonl
  python3 ./utils/render_session.py {work_dir}/session.jsonl {work_dir}/session.md
fi
```

---

## 十一、Phase 7：Round Contract 校验 + round_result.json

### 11.1 执行 submit_round.py

```bash
python3 skills/triton-agent-loop/scripts/submit_round.py \
    --round-dir {work_dir} --current-round {round_index} \
    --final-round {max_rounds} --op-dir triton_ascend_output/{op_name}-{algorithm}-{run_tag}
```

### 11.2 写入 round_result.json

```json
{
  "round_index": 1,
  "status": "success",
  "algorithm": "fusion-search",
  "best_speedup": 1.85,
  "target_reached": false,
  "improvement_made": true,
  "fusion_search": {
    "phase": "initial_discovery",
    "generation": 1,
    "selected_fusion_id": "F1",
    "route_count": 3,
    "best_route": "route-1",
    "gene_pool_size": 4
  },
  "design_summary": "Fuse layernorm scale/shift with elementwise add to eliminate intermediate HBM trips.",
  "iter_count": 3,
  "opt_iter_count": 2,
  "key_fixes": ["fixed precision loss in fused rsqrt by using float32 accumulator"],
  "bottlenecks_observed": ["atomic fusion F3 causes register pressure on small shapes"],
  "artifacts": {
    "final_code": "triton_ascend_output/{op_name}-{algorithm}-{run_tag}/opt-round-1/{op_name}_generated.py",
    "gene_pool_json": "triton_ascend_output/{op_name}-{algorithm}-{run_tag}/opt-round-1/gene_pool.json"
  }
}
```

同 naive/round.md §11.2，子 agent 应尽可能写入 `design_summary`、`iter_count`、`opt_iter_count`、`key_fixes`、`bottlenecks_observed`，供 Orchestrator 构建 enriched `lineage_summary`。

### 11.3 退出

写入 `round_result.json` 后立即退出。

---

## 十二、错误处理

| 阶段 | 错误 | 处理 |
|------|------|------|
| Phase 0 | 基线冻结失败 | 同 naive/round.md 错误处理 |
| Phase 1 | 无法完成原子分解 | 降级为全融合，继续执行 |
| Phase 2 | 无可行的融合方案 | 以全融合作为唯一方案，继续执行 |
| Phase 3/Step G5 | 全部 route 失败 | 当前 round 标记 failed |
| Phase 3/Step G5 | 部分 route 失败 | 从成功 route 中选最优 |
| Step G0 | 基因池文件缺失 | 降级为重跑 B1/B2 |
| Step G1 | 所有基因 fitness 为 null | 等权重随机选择 |
| Step G4 | profiling 数据不可用 | 跳过基因新增 |

---

## 十三、约束

| 约束 | 说明 |
|------|------|
| 融合方案只有一个 | B2 选出最高分融合方案，所有路线共用 |
| 基因 = 知识编码 | 只有优化技术是基因，融合方案不是 |
| 全自动 | 禁止等待用户确认 |
| B1/B2 仅 Round 1 | 后续 round 不重跑 |
| Round 2+ 进化循环 | 选择->交叉->变异->profiling新增->派发->更新适应度 |
| 精英保留 | 每轮至少保留上轮最优基因，保证性能不下滑 |
| 算法自包含 | 子 agent 内部完成 B1/B2/基因进化，Orchestrator 只接收 round_result.json |

---

## 十四、沟通风格

- 专业、技术、简洁。每完成一个 Phase/Step 输出一行状态。
- Round 1：B1->原子分解摘要；B2->选出的融合方案；C0->基因池规模和路线分配；D0->并行结果。
- Round 2+：G1->选中基因；G2/G3->交叉变异摘要；G4->新基因；G6->进化对比和基因池状态。
- 不向用户提问、不暂停等待。
- Phase 7 完成后输出退出摘要。
