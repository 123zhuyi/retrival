---
description: triton-optimizer-multi-agent 的 Fusion-Search Route Worker 规范。每个 route 子 CLI 在独立上下文中执行完整的 Triton-Ascend kernel 生成与优化 pipeline：内部 3 轮迭代 (E2-R1→R2→R3)、知识编码固定不变、profiling 驱动参数微调、知识价值评估与蒸馏、收敛检测。需传入 fusion_plan (融合方案) + optimization_genes (知识编码基因)，由 fusion-search worker 通过 dispatch_parallel.py 派发。
---

> **本文件是 Fusion-Search Route Worker 的权威执行规范。**
> - 由 fusion-search worker 在 D0 阶段通过 `dispatch_parallel.py` 启动。
> - 每个 route 子 CLI 上下文**仅包含**：本文件、`task_manifest.json`（含 `fusion_plan` + `optimization_genes`）、当前 `work_dir`（`route-{N}/`）、标准 skill/reference。
> - Route 子 CLI 禁止：读取其他 route 的产物、读取 `round_index.json`、做停止判定、更新 template。
> - **本文件定义了完整的 triton-ascend-kernel 内部流程**，包括内部轮次迭代、知识编码演化、profiling 驱动收敛。原 triton-ascend-kernel SKILL.md 的外部文档引用已内联。

---

## 一、Route 子 Agent 边界

你是 **Fusion-Search Route Worker**。你的任务是在给定的融合方案（`fusion_plan`）和基因组合（`optimization_genes`）下，完成从 kernel 设计到高性能交付的全流程。

**执行模型**：内部 3 轮迭代（E2-R1 → E2-R2 → E2-R3），每轮编译通过 → 精度 PASS → msprof 数据采集。3 轮中 knowledge encoding 保持不变，只基于 profiling 做模板继承和参数微调。

**禁止行为**：
- 读取其他 route 的产物或 `round_index.json`
- 做跨轮停止/继续判定
- 在 knowledge encoding 不变的前提下大幅重写 kernel 结构

---

## 二、Route Pipeline 总览

```
Phase A: 输入验证 + 环境确认 + 基线冻结
Phase B: 计算原子分解（范围为 fusion_plan.fused_atoms）
Phase C: 算法设计与知识编码锁定（编码来自 optimization_genes 基因列表）
Phase E2-R1: 编译 → 精度 PASS → msprof 采集
Phase E2-R2: 以 R1 kernel 为模板，profiling 驱动参数微调
Phase E2-R3: 以 R2 kernel 为模板，profiling 驱动参数微调
Phase G1-G3: 知识价值评估 + 低价值删除 + KB 更新
Phase G4: 自动蒸馏（speedup > 10%）
Phase H0: 收敛检测
Phase: 报告输出 + 会话导出 + 产物校验
```

全程自主，禁止向用户提问。

---

## 三、Phase A：输入验证 + 基线冻结

### A.1 读取 manifest

读取 `task_manifest.json`，提取：
- `op_name`, `route_id`, `work_dir`
- `fusion_plan`: `fusion_id`, `fused_atoms`, `eliminated_intermediates`, `description`, `math_transform`
- `optimization_genes`: `[{gene_id, ku_id, category, description, params}, ...]`
- `baseline_dir`, `input_files`, `config`
- `arch`（未指定时 `npu-smi info` 检测，失败用 `ascend910b1`）

### A.2 工作目录准备

1. 确认 `work_dir` 存在。
2. 从 `baseline_dir` 复制 `{op_name}_generated.py` → `{work_dir}/{op_name}_generated.py`。
3. 基线冻结：调用 `freeze_baseline.py`。
4. 写入初始 `summary.json`，包含 `route_id` 和 `fusion_id`。

---

## 四、Phase B：计算原子分解

范围为 `fusion_plan.fused_atoms` 指定的原子组。

1. 读取 `{work_dir}/{op_name}_generated.py` 的 kernel 实现。
2. 如果这是第一轮且 baseline 为原始代码，分析 `Model.forward()` 的计算图并拆解为原子序列。
3. 如果 baseline 已经是融合后的 kernel，分析当前 kernel 内部的计算步骤拆分。
4. 产出原子序列表格到 `{work_dir}/atom_decomposition.md`：

```text
| atom_id | op_type    | formula              | inputs       | outputs      | numerical_risk |
|---------|------------|----------------------|--------------|--------------|----------------|
| ...     | ...        | ...                  | ...          | ...          | ...            |
```

---

## 五、Phase C：算法设计与知识编码锁定

### C.1 知识编码（Knowledge Encoding）

本 route 的知识编码**直接来自 manifest 中的 `optimization_genes`**。这些是 fusion-search worker 分配的基因（知识编码），route 内部只需锁定执行，不自行增删：

```json
{
  "route_id": "route-1",
  "encoding": [
    {
      "ku_id": "K1",
      "gene_id": "G001",
      "category": "tiling",
      "description": "沿 M 维度分块，块大小 128",
      "source": "develop_guide",
      "code_target": "grid / block size 参数",
      "params": {"block_size_m": 128},
      "status": "active"
    },
    {
      "ku_id": "K2",
      "gene_id": "G002",
      "category": "memory",
      "description": "使用 tl.dot 替代逐元素乘加，利用 TensorCore",
      "source": "develop_guide",
      "code_target": "核心计算循环",
      "params": {},
      "status": "active"
    }
  ]
}
```

- `optimization_genes` 中的每个条目 → 一个知识编码单元（ku）
- `gene_id` 追溯回基因池中的基因

**关键约束**：知识编码在 E2-R1 到 E2-R3 的 3 轮中**必须保持不变**。3 轮中只允许：
- 基于 profiling 反馈调整 block size / grid size / num_warps / num_stages 等参数
- 模板继承（下一轮基础代码来自上一轮）
- 禁止新增、删除或修改知识编码单元

### C.2 算法设计 (sketch)

调用 `triton-op-designer` skill，传入：
- `op_name`, `task_desc`, `arch`
- `fusion_plan`（融合目标：要融合哪些原子）
- `encoding`（锁定的知识编码，来自 `optimization_genes`）
- `gpu_kernel_ref`（如有）

产出 `{work_dir}/sketch.txt`。

### C.3 Layer 1 合规检查

与 naive/round.md Phase 2 Step 3 一致：读取 Layer 1 约束，逐条核对 sketch 兼容性。冲突 → 反馈 designer 重新设计，最多 2 次。

---

## 六、Phase E2：内部 3 轮迭代

### 6.0 通用规则

**有效轮次定义**：
- 有效 Round = 编译通过 + 精度 PASS + msprof `Task Duration(us)` 已采集
- 无效：仅编译失败后修复（A 类错误处理，不算完成一轮）
- 无效：仅 host 计时，无 msprof 数据

**性能指标**：
- **主指标**：`msprof --output=./prof_out --task-time=on --ai-core=on python3 bench.py` 输出的 `op_summary_*.csv` 中的 `Task Duration(us)`
- Host 端计时仅用于 smoke/debug，禁止作为正式性能数字
- `aicore_time(us)` 仅作辅助诊断
- 可读取 `aiv_scalar_ratio`、`aiv_mte2_ratio`、`cube_utilization(%)` 做瓶颈分析

**精度标准**：
- 使用现有 `verify.py` 验证（`passed_cases == total_cases > 0`）
- 优先使用 `benchmark.py` 的 L1 闸门作为兜底

### 6.1 E2-R1：首轮生成

```
1. 代码生成: triton-op-coding skill
   输入: op_name, task_desc, arch, sketch, fusion_plan, encoding

2. AST 预检查: validate_triton_impl.py

3. 功能验证: verify.py
   - 通过 → 继续 4
   - 失败 → Conductor 分析 (A/B/C 类) → 重试 (最多 5 次)

4. 性能测试: benchmark.py + msprof
   msprof --output={work_dir}/prof_out_r1 --task-time=on --ai-core=on \
     python3 .claude/skills/triton-op-verifier/scripts/benchmark.py \
       --op_name {op_name} --verify_dir {verify_dir} \
       --warmup 5 --repeats 50

5. 记录: task_duration_us (从 op_summary CSV), speedup_vs_baseline,
   aiv_scalar_ratio, aiv_mte2_ratio, cube_utilization
```

产物：`{work_dir}/output/round_1/kernel.py`, `prof_out_r1/`, `round_1_result.json`

### 6.2 E2-R2：模板继承 + 参数微调

以 R1 kernel 为模板，基于 profiling 数据调整参数。**不改变知识编码。**

1. 分析 R1 瓶颈：`aiv_scalar_ratio`, `aiv_mte2_ratio`, `cube_utilization`
2. 调整参数（保持 encoding 不变）：
   - block size / grid size
   - num_warps / num_stages
   - pipeline 阶段数
   - 输入/输出 tile 形状
3. 代码修改 → AST 预检查 → 精度验证 → msprof 采集
4. 记录 R2 结果

### 6.3 E2-R3：精细收敛

以 R2 kernel 为模板，同上流程。目标是收敛到该知识编码下的局部最优。

### 6.4 内部收敛判定

3 轮完成后（或提前连续 2 轮无提升），选择 `task_duration_us` 最小的 kernel 作为本 route 的最终产物。

---

## 七、Phase G：知识管理

### G1：知识价值评估

对每个 encoding 中的知识单元，计算其对性能的贡献：

| 维度 | 计算方法 |
|------|----------|
| 直接贡献 | 该知识单元首次引入时的 speedup 变化 |
| 稳定性 | 3 轮中该知识单元相关瓶颈的变化幅度 |
| 可迁移性 | 是否在其他算子类别中有已知应用 |

### G2：低价值单元删除

标记并删除满足以下条件的知识单元：
- 连续 2 轮相关瓶颈无改善
- 引入后 speedup 无提升（< 1%）
- 导致数值不稳定

删除时记录删除依据到 `{work_dir}/knowledge_evolution.md`。

### G3：知识库更新记录

输出 `{work_dir}/knowledge_pool_update.json`：

```json
{
  "route_id": "route-1",
  "round": 1,
  "encoding_initial": ["K1", "K2", "K3"],
  "deleted_units": [
    {"ku_id": "K3", "reason": "no improvement after 2 rounds"}
  ],
  "active_units": ["K1", "K2"],
  "new_insights": [
    "沿 M 维度 block_size=128 在 round 2 达到瓶颈，可能需要更大的 block"
  ],
  "kb_changes_summary": "Removed K3, no new units added"
}
```

### G4：自动蒸馏

当本 route 的 `best_speedup > 1.1`（即提升 > 10%）时：

1. 将 route 的知识编码写入 `{work_dir}/knowledge_pool_entry.json`
2. 追加 `{work_dir}/knowledge_pool_history.json`，记录本次编码的快照
3. 提炼该 route 中**有效知识单元**的组合模式

**自动蒸馏规则**：
- 仅蒸馏 speedup > 10% 的路线的知识编码
- 蒸馏内容：知识单元列表 + 最佳参数组合 + 瓶颈分析结果
- 不允许蒸馏未经验证的推测性优化项

---

## 八、Phase H0：收敛检测

在 3 轮完成后，执行收敛检测：

1. 读取 R1, R2, R3 的 `task_duration_us`（msprof 数据）
2. 计算最近 2 轮的变化率：`|task_duration[n] - task_duration[n-1]| / task_duration[n-1]`
3. 判定：连续 2 轮变化 ≤ 3% AND 已完成 3 轮 → 收敛
4. 输出收敛状态到 `summary.json` 的 `convergence` 字段：

```json
{
  "convergence": {
    "converged": true,
    "rounds_completed": 3,
    "last_2_rounds_change_pct": [1.2, 0.8],
    "best_round": 3,
    "best_task_duration_us": 1234.5
  }
}
```

**若未收敛**（变化 > 3%）：
- 记录原因（瓶颈未解决、需要新的知识编码方向）
- 以当前最优 kernel 作为 route 最终产物（不阻塞上层流程）
- 在 `knowledge_evolution.md` 中记录 "route 内未收敛，建议跨代引入新知识编码"

---

## 九、报告输出

### 9.1 最终代码

复制最佳 kernel 到 `{work_dir}/{op_name}_generated.py`。

### 9.2 report.md

包含：
- 基本信息：arch、route_id、fusion_plan、encoding(optimization_genes)
- **原子分解摘要**
- **知识编码设计**：每个知识单元的描述、来源、代码目标
- **内部迭代记录**：
  - E2-R1: 编译状态、精度、`task_duration_us`、瓶颈分析
  - E2-R2: 以 R1 为模板、参数调整依据、结果
  - E2-R3: 以 R2 为模板、参数调整依据、结果
- **知识-代码映射表**：每个激活知识单元对应的代码片段与验证状态
- **精英统计**：各轮 `task_duration_us`、speedup、利用率变化
- **知识价值评估**：各单元评分、删除单元及原因
- **知识库更新**：新增/删除记录、变化统计
- **收敛检测**：是否收敛、最佳轮次
- 代码路径

### 9.3 summary.json

```json
{
  "success": true,
  "route_id": "route-1",
  "fusion_id": "F1",
  "gen_iterations": 2,
  "internal_rounds": 3,
  "best_speedup": 1.85,
  "best_task_duration_us": 1234.5,
  "convergence": {
    "converged": true,
    "rounds_completed": 3,
    "last_2_rounds_change_pct": [1.2, 0.8],
    "best_round": 3
  },
  "knowledge": {
    "active_units": ["K1", "K2"],
    "deleted_units": ["K3"],
    "distilled": true,
    "speedup_exceeds_10pct": true
  },
  "perf_data": { "...同现有格式..." }
}
```

### 9.4 额外产物

- `{work_dir}/knowledge_evolution.md`：知识编码演化记录
- `{work_dir}/knowledge_pool_entry.json`：知识单元详情（蒸馏用）
- `{work_dir}/knowledge_pool_history.json`：知识池演化历史
- `{work_dir}/route_iterations.json`：3 轮迭代的完整数据（task_duration, utilization, bottleneck）

---

## 十、会话导出 + 产物校验

与 naive/round.md Phase 6/7 一致：
- 复制 `session.jsonl`
- 执行 `submit_round.py` 校验
- 写入 `round_result.json`（包含 route_id、fusion_id、knowledge 摘要）

---

## 十一、错误处理

| 阶段 | 错误 | 处理 |
|------|------|------|
| Phase A | 环境不可用 | 终止 route，返回 failed |
| Phase C | 无法锁定有效知识编码 | 最小化编码（只用 develop_guide 中的基本模板） |
| E2-R1 | 编译 5 次失败 | 标记 encoding 为不可行，返回 failed |
| E2-R1 | 精度 5 次失败 | 同编译失败处理 |
| E2-R1 | msprof 不可用 | 告警，使用 benchmark.py 数据作为替代 |
| E2-R1→R3 | 连续 2 轮无提升 | 提前终止内部迭代，以最优轮结果输出 |
| Phase H0 | 未收敛 | 以当前最优输出，记录未收敛原因 |

---

## 十二、约束

| 约束 | 说明 |
|------|------|
| 知识编码不变 | 3 轮中 knowledge encoding 必须保持不变，只允许参数微调 |
| 精度标准 | 使用 verify.py / benchmark.py，不做双标杆 L0 |
| 性能主指标 | msprof `Task Duration(us)`，禁止用 host 计时 |
| 禁止 PyTorch 退化 | forward() 中禁止 torch._ 计算 |
| 验证方式 | 必须调用 triton-op-verifier skill |
| 禁止编造数据 | 所有数值从 msprof CSV / perf_result.json 读取 |
| 不读历史 | 不读取 `round_index.json` 或其他 route 产物 |
| 算法上下文 | 本 route 是 fusion-search 的内部并行单元，对外只返回 route 结果 |

---

## 十三、沟通风格

- 每完成一个 Phase 输出一行状态
- E2 每轮完成后输出紧凑摘要：`[route-1 E2-R1] compile=OK precision=PASS task_duration=1234us speedup=1.32`
- G 阶段输出知识单元变化摘要
- H0 输出收敛结论
- 全部完成后输出一行退出摘要
