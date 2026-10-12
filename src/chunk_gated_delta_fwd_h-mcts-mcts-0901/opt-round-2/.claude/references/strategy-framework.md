# Evolutionary Strategies Reference

> **位置**：`plugins-community/triton-optimizer-multi-agent/skills/triton-agent-loop/scripts/strategies/`
>
> **作用**：定义主 Orchestrator 在 Step 3c（方向切换）和 Step 4（正常下一轮）中如何选择下一个优化方向、如何为所选方向挑选 baseline，以及如何对方向尝试结果进行收益分级。
>
> **面向读者**：主 Orchestrator Agent。开发者使用说明参见项目根目录 `README.md` 的“方向选择策略”章节。

---

## 一、设计原则

1. **策略即算法插件**：每种优化算法实现为一个独立的 Python 模块，放在 `strategies/` 目录下，模块/类的 `name` 与算法名一致（如 `naive`、`mcts`、`fusion-search`）。Orchestrator 通过用户选择的算法加载对应策略。
2. **默认向后兼容**：未配置 `algorithms` 时，`config_loader.read_config()` 会自动注入三种算法的默认预设。
3. **不侵入核心数据**：策略选择是运行时行为；`round_index.json`、`summary.json`、`task_manifest.json` 的 schema 不随策略改变。
4. **复杂策略可自维护状态**：如遗传算法需要保存种群，可在 `triton_ascend_output/{op_name}-{algorithm}-{run_tag}/.strategy_state/{name}.json` 中自行维护，框架不感知其内部结构。

---

## 二、策略接口

所有策略必须继承 `skills/triton-agent-loop/scripts/strategies/base.py` 中的 `Strategy` 抽象基类，并实现以下三个方法：

### 2.1 `select_next_direction`

```python
def select_next_direction(
    self,
    round_index: list[dict[str, Any]],
    pattern_index_path: Path,
    op_category: str | None,
    current_policy: str,
    context_changed: bool = False,
    direction_exhausted: bool = False,
    context: dict[str, Any] | None = None,
) -> tuple[str, str, str, str, list[str], dict[str, Any]]:
```

参数说明：
- `round_index`：历史 round 元数据。
- `pattern_index_path`：**历史接口，当前已忽略**。方向选择改为 LLM 自主探索，策略不应再读取 pattern catalog。
- `op_category`：算子类别。
- `current_policy`：当前 `analysis_policy`。
- `context_changed`：上下文是否变化（例如方向切换后）。
- `direction_exhausted`：当所有 known directions 都已在 `pattern_entry` / `bottleneck_analysis` / `ir_supported` 三层尝试且均为 low_yield/failed 时为 `True`。策略可在此情况下建议停止或继续让 LLM 探索新方向。
- `context`：调用者传入的上下文，通常包含 `op_dir`（算子输出目录）和 `current_round`（当前轮次）。需要持久化状态的策略（如 MCTS）会用到它。

返回一个六元组：

| 位置 | 含义 | 示例 |
|---|---|---|
| 0 | `round_strategy` | `"exploration"` / `"structural_change"` |
| 1 | `analysis_policy` | `"pattern_entry"` / `"bottleneck_analysis"` / `"ir_supported"` |
| 2 | `direction` | `"pattern: custom"` |
| 3 | `hypothesis` | 本轮优化假设 |
| 4 | `evidence_sources` | `["model_knowledge", "direction_history"]` |
| 5 | `metadata` | 供调试/记录的额外字典 |

### 2.2 `score_baseline_for_direction`

```python
def score_baseline_for_direction(
    self,
    candidate: dict[str, Any],
    target_direction: str,
    target_strategy: str = "exploration",
    context: dict[str, Any] | None = None,
) -> float:
```

对历史 round（`candidate`）作为下一方向的 baseline 进行打分。返回值必须在 `[0.0, 1.0]` 之间，`0.0` 表示该候选不可选（如验证失败、idle round）。

`context` 通常包含 `op_dir`，供需要读取持久化状态的策略使用。

### 2.3 `classify_direction_yield`

```python
def classify_direction_yield(
    self,
    best_speedup: float,
    baseline_speedup: float,
    target_speedup: float,
    verify_passed: bool,
    improvement_made: bool,
    plateau_detected: bool = False,
    timeout: bool = False,
    idle: bool = False,
    context: dict[str, Any] | None = None,
) -> tuple[str, str, float]:
```

返回三元组 `(direction_yield, yield_reason, absolute_gain)`，用于写入 `round_index.json` 的因果元数据字段。

`context` 通常包含 `op_dir` 和 `round_dir`，供需要记录本轮结果到持久化状态的策略使用。

### 2.4 可选生命周期钩子

除三个抽象方法外，基类还提供两个可选钩子。需要跨 round 维护状态的策略（如 MCTS、遗传算法）可以重写它们；默认实现为空。

#### `init_for_operator(self, op_dir, config, round_index)`

Orchestrator 初始化或恢复工作区后调用一次。策略可在此处创建/加载持久化状态（例如 MCTS 树）。

#### `record_round_result(self, round_index_entry, round_dir, context)`

每轮结束后、选择下一轮方向之前调用。`round_index_entry` 是本轮在 `round_index.json` 中的条目；`round_dir` 是本轮目录（如 `opt-round-3`）。策略应在此处把结果写回持久化状态。

---

## 三、内置策略

| 名称 | 文件 | 说明 |
|---|---|---|
| `naive` | `strategies/naive/strategy.py` | 默认策略，基于历史表现选择 baseline；方向固定为 `pattern: custom`，由 LLM 自主探索。 |
| `mcts` | `strategies/mcts/strategy.py` | 蒙特卡洛树搜索策略。每个 round 对应一个树节点，用 UCT 选择扩展父节点（baseline）；具体方向由子 agent 基于 MCTS 上下文自主选择。 |
| `fusion-search` | `strategies/fusion_search/strategy.py` | 融合搜索策略。负责为 fusion-search 算法选择 baseline；融合分析、基因池进化与并行 route 派发由子 agent 内部完成。 |
| `random_stub` | `strategies/random_stub.py` | 仅用于验证框架可插拔性的非确定性 stub，返回 `pattern: custom`。**不适合生产**。 |

---

## 四、与 Orchestrator 的交互

1. Orchestrator 在 `run_orchestrator` 中调用 `read_config()`，若配置中无 `algorithms` 则注入默认预设。
2. 工作区初始化/恢复后，Orchestrator 根据用户选择的算法调用 `load_strategy_for_algorithm(config, algorithm)` 加载策略。
3. 若策略有 `init_for_operator()`，Orchestrator 调用它让状态感知型策略加载或创建持久状态。
4. 每轮子 agent 完成后，Orchestrator 先调用策略的 `record_round_result()`（如果存在），再进入下一轮方向选择。
5. 需要选择方向时，Orchestrator 调用 `strategy.select_next_round()`，传入 `context`。
6. Orchestrator 通过 CLI 把 `algorithm` 以及 `algorithms.<algorithm>.params` 传递给 `transition_next_round.py`（通过 `--strategy-name` 和 `--strategy-params`）。
7. `transition_next_round.py` 使用 `load_strategy_for_algorithm(config, algorithm)` 加载策略进行收益分级。
8. `submit_round.py` 在验证 completed round 时，读取 `summary.json` 中的 `algorithm`，并使用对应策略的 `classify_direction_yield()`。

---

## 五、配置 schema

`config.json` 中的算法参数位于 `algorithms` 映射下：

```json
{
  "algorithms": {
    "naive": {"enabled": true, "params": {}},
    "mcts": {
      "enabled": true,
      "params": {
        "c_uct": 1.414,
        "c_pw": 2.0,
        "alpha": 0.5
      }
    },
    "fusion-search": {
      "enabled": true,
      "params": {
        "max_routes": 3,
        "crossover_rate": 0.7,
        "mutation_rate": 0.15
      }
    }
  }
}
```

完整 schema 参见 `skills/triton-agent-loop/schemas/config.schema.json`。

---

## 六、MCTS 策略说明

`mcts` 是首个基于状态感知的复杂策略，实现文件为 `strategies/mcts/strategy.py`，核心树逻辑在 `strategies/mcts/` 下。

### 6.1 节点映射

- 每个成功创建并评估的 `opt-round-*` 对应 MCTS 树中的一个节点。
- 虚拟根节点 `root` 没有代码，表示“从原始任务描述开始”。
- 父节点 → 子节点的边表示“以父 round 的代码作为 baseline，尝试一个新的优化方向”。

### 6.2 选择流程

1. `select_next_round` 调用 `tree.select()`，UCT 决定从哪个**已评估**节点扩展（PENDING 节点会被忽略）。
2. 得到父节点后，子 agent 基于 MCTS 上下文（父节点已尝试方向、奖励、瓶颈信号）自主选择新的优化方向；策略本身不再提供固定方向目录。
3. 将 `(parent_id, direction)` 记录为 `pending_expansion` 并持久化。
4. 返回的 `metadata.preferred_baseline_dir` 为父节点对应的 round 目录；空字符串表示父节点是根（无 baseline）。

### 6.3 结果记录

- `record_round_result` / `classify_direction_yield` 在子 round 完成后被调用。
- 策略读取 `summary.json`，构造临时 `verify_result.json` 与 `perf_result.json`，调用 `tree.evaluate()` 回传奖励。
- 奖励规则：成功且 `best_speedup > 0` → `log(best_speedup)`；失败按编译/输出错误分别给负奖励。
- 记录完成后清除 `pending_expansion`。

### 6.4 参数

| 参数 | 说明 |
|---|---|
| `c_uct` | UCT 探索常数 |
| `c_pw` | Progressive Widening 系数 |
| `alpha` | Progressive Widening 指数 |
| `failure_threshold` | 节点失败几次后剪枝 |
| `reward_output_error` | 输出/校验失败奖励 |
| `reward_compile_error` | 编译失败奖励 |

状态文件：`triton_ascend_output/{op_name}-{algorithm}-{run_tag}/.strategy_state/{op_name}_mcts_tree.json`。
