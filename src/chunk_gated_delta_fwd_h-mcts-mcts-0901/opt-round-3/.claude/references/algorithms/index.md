# 优化算法目录

本目录列出 `triton-optimizer-multi-agent` 支持的所有优化算法。每种算法对应一份独立的 sub agent 规范文档（`round.md`），由 Orchestrator 在运行前选择并写入 `task_manifest.json` 的 `algorithm_reference` 字段。

> **算法 vs 策略**
> - **算法（algorithm）**：子 agent 执行代码生成/优化的方式，对应本目录下的 `.md` 规范。
> - **策略（strategy）**：Orchestrator 选择 direction / baseline 的算法，对应 `skills/triton-agent-loop/scripts/strategies/` 下的 Python 实现。
> 两者正交：Orchestrator 先选 algorithm，再在该 algorithm 下运行对应 strategy 生成 direction / baseline。

---

## 可用算法

| 算法名 | 规范文档 | 适用场景 | 子 agent 行为 | 是否需要策略生成 direction |
|---|---|---|---|---|
| `naive` | [naive/round.md](naive/round.md) | 通用单算子优化 | 执行 manifest 中的 direction；`pattern: custom` 时由 LLM 自主决定具体方向 | 是（策略输出 `pattern: custom`，具体方向由 LLM 决定） |
| `mcts` | [mcts/round.md](mcts/round.md) | 需要探索大量方向、利用历史收益选择父 round | 基于 MCTS 父节点上下文，自主决定本 round direction | 否（MCTS 只选父 round） |
| `fusion-search` | [fusion-search/round.md](fusion-search/round.md) | 可融合多原子算子 | Round 1 做 B1/B2 发现融合方案 + 初始基因池；Round 2+ 做一代基因进化 | 可选（通常由基因池驱动） |

---

## 算法选择流程

1. Orchestrator 启动时读取 `config.json` 的 `algorithms` 映射，得到 `enabled=true` 的算法列表。
2. 扫描 `.triton-agent/` 目录，查找 `state-{op_name}-*.json`：若命中唯一文件且其中已有 `algorithm`，直接使用（会话恢复）。
3. 若只存在旧版 `state-{op_name}.json` 和旧版工作区 `triton_ascend_output/{op_name}/`，主 agent 必须使用 `AskUserQuestion` 向用户说明情况，报告推断出的算法、当前 round、迁移计划，并提供三个选项：继续使用推断算法、指定其他算法、放弃旧工作区重新开始。用户确认后调用 `orchestrator_loop.py --migrate-legacy <algorithm>` 迁移。
4. 若 state 中无算法且没有旧工作区（新任务），向用户展示 enabled 算法列表，由用户选择一种算法。
5. 将选择结果写入：
   - `.triton-agent/state-{op_name}-{algorithm}-{run_tag}.json` 的 `algorithm` 和 `algorithm_reference` 字段
   - `task_manifest.json` 的 `algorithm` 和 `algorithm_reference` 字段

选择结果**不再**写回 `config.json`。`config.json` 的 `algorithms` 映射只定义可用算法及其参数。

---

## 算法状态文件

部分算法需要在多轮之间继承状态文件，`transition_next_round.py` 会根据策略的 `state_file_templates` 类属性自动复制：

| 算法 | 状态文件（相对于 round 目录） |
|---|---|
| `naive` | 无 |
| `mcts` | 无（MCTS 树状态由策略直接管理在 `triton_ascend_output/{op_name}-{algorithm}-{run_tag}/.strategy_state/`，不复制） |
| `fusion-search` | `gene_pool.json`, `fusion_plan.json` |

---

## 扩展新算法

新增算法的标准步骤：

1. 在 `.claude/references/algorithms/` 下新建 `{algorithm-name}/` 目录。
2. 新增 `{algorithm-name}/round.md`，定义该算法的 sub agent 单 round 执行规范。
3. 若算法需要内部并行子 agent，新增 `{algorithm-name}/{sub-name}.md` 子规范。
4. 在本文件中登记新算法。
5. 在 `config.schema.json` 的 `algorithms.patternProperties` 中允许该算法名。
6. 在 `config_loader.DEFAULT_ALGORITHMS` 中添加该算法的默认预设（默认 `enabled: false`，用户可在 `config.json` 中启用）。
7. （必须）在 `skills/triton-agent-loop/scripts/strategies/` 下新增同名 Strategy 子类（模块/目录名可用下划线，类 `name` 属性为算法名），供 Orchestrator 加载。
8. 从 `orchestrator_loop.py` 和 `transition_next_round.py` 的 argparse 中移除 `--algorithm` 的硬编码 `choices`；运行期会根据 `discover_strategies()` 自动校验。
9. 将 `task_manifest.schema.json` 和 `round_result.schema.json` 中 `algorithm` 字段的 `enum` 改为 `pattern: "^[a-z0-9-]+$"`。
10. 若算法需要跨 round 复制状态文件，在 Strategy 类中设置 `state_file_templates: list[str]`。

---

## 参考

- Orchestrator 规范：[`orchestrator-loop.md`](../orchestrator-loop.md)
- 策略框架说明：[`strategy-framework.md`](../strategy-framework.md)
- 策略脚本：`skills/triton-agent-loop/scripts/strategies/`
