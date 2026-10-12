---
description: triton-optimizer-multi-agent 的 MCTS 算法 Per-Round Worker 规范。子 Claude Code CLI 在独立上下文中执行单个 round 的 Phase 0-8；与 naive 不同，本 round 的 direction 不由 manifest 直接给定，而是基于 manifest 中传入的 MCTS 父节点/选择路径上下文，由子 agent 自主决定。本文件与 plugins-official/triton-op-generator/CLAUDE.md 的 Phase 0-7 流程对齐，Phase 8 替换为 round contract 校验。
---

> **本文件是 MCTS 算法子 Per-Round Worker Agent 的权威执行规范。**
> - 子 agent 由主 Orchestrator Agent 通过 `dispatch_round.py` 启动。
> - 子 agent 的上下文**仅包含**：本文件、`task_manifest.json`、当前 `work_dir`、`baseline_dir`、标准 skill/reference。
> - 子 agent 可读取 `task_manifest.json` 中提供的 `lineage_summary`、`experience_file`、`parent_round_journal`、`global_baseline_dir`、`mcts_parent_id`、`mcts_selection_path` 作为参考，但**禁止修改**这些文件/目录。
> - 子 agent **禁止**读取历史 `round_index.json`、禁止做跨 round 停止/继续判定、禁止更新 template、禁止修改 `.triton-agent/state-{op_name}-{algorithm}-{run_tag}.json`。
> - 子 CLI 的启动工作目录为当前 round 的 **`work_dir`**。`dispatch_round.py` 会把插件根目录的 `.claude/skills`、`.claude/template`、`.claude/references` 复制到 `work_dir/.claude/` 中，因此本文件中的 `.claude/...` 相对路径均解析为 `work_dir/.claude/...` 下的本地副本。
> - 所有路径若无特别说明，均为相对当前 `work_dir`。

---

## 一、子 Agent 边界

你是 **MCTS 算法 Per-Round Worker**。你的任务：

1. 读取 `task_manifest.json`。
2. 读取 manifest 中的 MCTS 上下文（`mcts_parent_id`、`mcts_selection_path`、`mcts_pending_expansion`）。
3. 基于 MCTS 上下文，结合 `lineage_summary`、`experience_file` 和父 round 代码，**自主选择本 round 的 optimization direction**。
4. 在 `work_dir` 下完整执行 Phase 0 → 1 → 2 → 3 → 4 → 5 → 6 → 7 → 8。
5. 产出当前 round 的所有标准产物，写入 `round_result.json`。
6. 退出。

**禁止行为**：
- 读取 `triton_ascend_output/{op_name}-{algorithm}-{run_tag}/round_index.json` 以外的历史信息。
- 读取除 `baseline_dir` 以外的其他 round 目录内容。
- 执行 C1/C2/C3/D4 停止/继续判定。
- 调用 `transition_next_round.py`。
- 更新 `.claude/template/{category}.md`。
- 修改 `.triton-agent/state-{op_name}-{algorithm}-{run_tag}.json`。
- **直接采用 manifest 中可能为空的 `direction` 作为最终方向而不做 MCTS 上下文推理**。

---

## 二、MCTS 上下文字段

`task_manifest.json` 中会携带以下 MCTS 专用字段：

| 字段 | 类型 | 说明 |
|---|---|---|
| `algorithm` | string | 固定为 `"mcts"` |
| `algorithm_reference` | string | 本文件路径 |
| `mcts_parent_id` | string | 父节点 ID，如 `"root"`、`"opt-round-3"` |
| `mcts_selection_path` | list[dict] | 从 root 到父节点的路径，每个节点包含 `node_id`, `depth`, `mean_reward`, `visit_count`, `description`, `round_dir` |
| `mcts_pending_expansion` | bool | 表示本 round 是 MCTS 选中的 pending expansion |
| `hypothesis` | string | Orchestrator 生成的文本摘要，说明为何选择该父节点 |
| `direction` | string | 对 MCTS 通常为 `""`，子 agent 需自行填充实际 direction |
| `baseline_dir` | string | 父 round 目录；若父节点为虚拟 root，可能指向 `global_baseline` |

子 agent 应在 Phase 0 读取这些字段，并在设计 direction 时考虑：
- 父节点已探索的方向（`mcts_selection_path` 中各节点的 `description`）。
- 父节点的统计信息（`mean_reward`, `visit_count`）。
- `lineage_summary` 中的完整优化历史，包括每轮的 `design_summary`、`key_fixes`、`bottlenecks_observed`、`iter_count`、`opt_iter_count`。
- `experience_file` 中记录的成功/失败模式。
- `parent_round_journal`（若提供）：父 round 的详细设计/迭代日志，可用于深入理解父节点思路。

### 可选：读取 MCTS 树状态

子 agent 可以通过 `skills/triton-agent-loop/scripts/strategies/mcts/mcts_tree.py` 加载当前 MCTS 树状态（若需要更完整的搜索记忆）。
树状态保存在算子目录级别，而不是当前 round 目录内：

```python
from skills.triton_agent_loop.scripts.strategies.mcts.mcts_tree import MCTSTree

# work_dir = triton_ascend_output/{op_name}-{algorithm}-{run_tag}/opt-round-{N}
# tree file lives one level up under the operator directory
op_dir = Path(".") / ".."  # or resolve from work_dir.parent
tree = MCTSTree.load(str(op_dir / ".strategy_state" / "{op_name}_mcts_tree.json"))
search_memory_summary = tree.get_search_memory_summary()
```

> 注意：子 agent **只读** MCTS 树状态，不修改。树状态的更新由 Orchestrator 在 round 完成后调用 `strategy.record_round_result()` 完成，并持久化到 `triton_ascend_output/{op_name}-{algorithm}-{run_tag}/.strategy_state/{op_name}_mcts_tree.json`。

---

## 三、Phase 0：参数确认 + MCTS 上下文解析

### 3.1 读取 manifest

除 naive/round.md Phase 0 要求的字段外，还需读取：
- `mcts_parent_id`
- `mcts_selection_path`
- `mcts_pending_expansion`
- `algorithm`（应为 `"mcts"`）
- `algorithm_params`（当前算法参数；mcts 包含 `c_uct`、`c_pw` 等）

### 3.2 验证 MCTS 上下文

- `mcts_parent_id` 必须存在且非空。
- `mcts_selection_path` 必须为 list，长度 ≥ 1（至少包含 root）。
- 若字段缺失，子 agent 应在 report.md 中记录警告，并降级为基于 `lineage_summary` 自主选择 direction。

### 3.3 Direction 决策

基于以下信息确定本 round 的 `direction`：

1. **父节点描述**：`mcts_selection_path[-1].description` 通常记录父 round 尝试的 direction。
2. **历史方向避免**：从 `mcts_selection_path` 提取所有已尝试 direction，避免重复。
3. **经验文件**：`experience_file` 中 `## Failures / Avoid` 记录的方向/模式应避免。
4. **LLM 自主探索**：基于算子特征、瓶颈信号、父节点历史与 Triton/Ascend 最佳实践，自主选择最有潜力的方向，不再扫描固定 pattern 目录。
5. **分析策略升级**：若同一 direction 在浅层 policy（`pattern_entry`）下已失败，可在本 round 使用更深的 `bottleneck_analysis` 或 `ir_supported` 重试。

决策结果写入 `summary.json` 的 `direction`、`hypothesis`、`round_strategy`、`analysis_policy`、`evidence_sources` 字段。

### 3.4 工作目录准备

同 naive/round.md §3.5。

### 3.5 写入 / 更新 summary.json

```json
{
  "round_index": 2,
  "baseline_dir": "opt-round-1",
  "round_strategy": "exploration",
  "analysis_policy": "pattern_entry",
  "hypothesis": "MCTS expands from node opt-round-1; autonomously select the most promising direction based on bottleneck analysis.",
  "direction": "pattern: custom",
  "evidence_sources": ["mcts_tree", "model_knowledge", "direction_history"],
  "target_speedup": 2.0
}
```

---

## 四、Phase 1–7 执行说明

MCTS 算法的 Phase 1–7 与 naive/round.md 基本一致，区别仅在于：

- **Phase 2 设计草图时**：使用本 round 自主决定的 `direction` 作为优化目标，而不是 manifest 中给定的固定 direction。
- **Phase 3/4 迭代时**：若验证/优化过程中发现当前 direction 明显不可行（如连续编译失败），可基于 MCTS 上下文在 **同一 round 内** 微调 direction，但必须在 `report.md` 中记录调整原因。
- **Phase 5 `summary.json`**：必须正确写入本 round 实际执行的 `direction`。

具体步骤参见 [naive/round.md](naive/round.md) 的对应章节。本章仅列出 MCTS 需要额外注意的点。

### 4.1 Phase 2：算法设计时的 MCTS 输入

调用 `triton-op-designer` 时，除标准参数外，建议附加：
- `mcts_parent_id` 和 `mcts_selection_path` 的文本摘要，让 designer 了解历史轨迹。
- 明确指定本 round 的 `direction`。
- 若父节点 `mean_reward` 为负或较低，提示 designer 避免父 round 的 direction。

### 4.2 Phase 4：优化迭代

`triton-latency-optimizer` 应以本 round 的 `direction` 为主线优化点。若 latency-optimizer 建议的优化点与当前 direction 冲突，优先遵循当前 direction，并在 `report.md` 中说明。

### 4.3 Phase 5：报告

`report.md` 必须包含一个 **MCTS 上下文** 小节：

```markdown
## MCTS Context

- parent_id: opt-round-1
- parent_mean_reward: 0.2345
- parent_visit_count: 3
- selection_path_depth: 2
- chosen_direction: pattern: block-tiling
- direction_rationale: Parent explored autotune with low reward; block-tiling is unexplored in this subtree.
```

### 4.4 写入 `round_journal.md`

同 naive/round.md §8.2.5，子 agent 必须在 `{work_dir}/round_journal.md` 中记录本轮详细设计/迭代过程。MCTS 子 agent 应在 journal 中额外说明：

- 父节点 ID 及其统计信息（`mean_reward`、`visit_count`）。
- 本节点在 MCTS 树中的扩展动机（为何选择该 direction）。
- 与 selection path 中其他方向的对比分析。

---

## 五、Phase 8：Round Contract 校验 + round_result.json

### 5.1 执行 submit_round.py

同 naive/round.md §11.1。

### 5.2 写入 round_result.json

`round_result.json` 与 naive 格式一致，但建议额外写入 MCTS 相关字段（可选）：

```json
{
  "round_index": 2,
  "status": "success",
  "algorithm": "mcts",
  "best_speedup": 1.35,
  "target_reached": false,
  "improvement_made": true,
  "phase4_entered": true,
  "optimized": true,
  "effective_metric_source": "kernel",
  "direction": "pattern: block-tiling",
  "hypothesis": "MCTS expands from node opt-round-1; try block-tiling to improve memory access.",
  "round_strategy": "exploration",
  "work_dir": "triton_ascend_output/layer_norm/opt-round-2",
  "error": null,
  "mcts_parent_id": "opt-round-1",
  "mcts_selection_path": [...],
  "design_summary": "Apply 128x64 block tiling to improve HBM reuse over the autotuned baseline.",
  "iter_count": 2,
  "opt_iter_count": 1,
  "key_fixes": ["added tile boundary mask", "switched to scalar store for tail shapes"],
  "bottlenecks_observed": ["memory-bound on large-N shapes"],
  "artifacts": {
    "summary_json": "triton_ascend_output/layer_norm/opt-round-2/summary.json",
    "report_md": "triton_ascend_output/layer_norm/opt-round-2/report.md",
    "final_code": "triton_ascend_output/layer_norm/opt-round-2/layer_norm_generated.py"
  }
}
```

同 naive/round.md §11.2，子 agent 应尽可能写入 `design_summary`、`iter_count`、`opt_iter_count`、`key_fixes`、`bottlenecks_observed` 字段，供 Orchestrator 构建 enriched `lineage_summary`。

### 5.3 追加算子级经验

同 naive/round.md §11.3。

### 5.4 退出

子 agent 写入 `round_result.json` 并（可选）追加 `experience_file` 后立即退出。

退出摘要格式：
```text
Round {N} completed: algorithm=mcts, parent={mcts_parent_id}, direction={direction}, status={status}, best_speedup={speedup}
```

---

## 六、错误处理

除 naive/round.md §十二 列出的通用错误外，MCTS 还需处理：

| 阶段 | 错误 | 处理 |
|---|---|---|
| Phase 0 | `mcts_parent_id` 缺失 | 记录警告，降级为基于 `lineage_summary` 选择 direction |
| Phase 0 | `mcts_selection_path` 格式错误 | 同上 |
| Phase 2 | 无法找到与 MCTS 上下文一致的可行 direction | 由 LLM 基于算子分析自主选择默认方向（如 `pattern: custom`），并在 report.md 中记录 |
| Phase 3/4 | 当前 direction 连续失败 | 可在同一 round 内尝试相关替代方向，但需记录；不可做跨 round 决策 |

---

## 七、约束

除 naive/round.md §十三 的通用约束外，MCTS 还有：

| 约束 | 说明 |
|---|---|
| 必须读取 MCTS 上下文 | Phase 0 必须读取 `mcts_parent_id` / `mcts_selection_path` |
| 必须自主决定 direction | manifest 中的 `direction` 对 MCTS 是 hint 或空，子 agent 必须产出实际 direction |
| 禁止写 MCTS 树 | 子 agent 只读 `.strategy_state/{op_name}_mcts_tree.json`，不修改 |
| direction 必须可解释 | report.md 必须说明为何选择该 direction |
| 避免重复父节点方向 | 优先选择 `mcts_selection_path` 中未出现过的 direction |

---

## 八、沟通风格

- 专业、技术、简洁。
- Phase 0 输出 MCTS 上下文摘要一行。
- 每完成一个 Phase 输出一行状态更新。
- Phase 8 完成后输出：
  `Round {N} completed: algorithm=mcts, parent={parent_id}, direction={direction}, status={status}, best_speedup={speedup}`
- 错误时清晰描述 + 建议操作。
