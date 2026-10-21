# triton-optimizer-multi-agent

Triton Ascend 算子多智能体优化插件。主 Claude Code Agent 负责多轮调度，每轮启动独立的子 Claude Code CLI 在隔离上下文中执行 Phase 0-8，最终由主 agent 执行停止判定并输出全局最优结果。

## 与单 agent 版本的区别

| 维度 | triton-optimizer | triton-optimizer-multi-agent |
|------|------------------|------------------------------|
| 执行主体 | 单一会话内完成所有 round | 每 round 一个独立子 CLI |
| 上下文 | 单一会话累积 | 子 CLI 之间完全隔离 |
| 主 agent 职责 | 执行 Phase 0-8 + 调度 | 仅负责 round 间调度与判定 |
| 子 agent 职责 | 无 | 执行单个 round 的 Phase 0-8 |
| 跨轮判定 | 同一 agent | 仅主 agent |
| 经验沉淀 | 每轮结束后自动 | 所有 round 结束后由主 agent 统一完成 |

## 安装

```bash
bash init.sh project claude
```

## 使用

在目标算子项目中启动后，输入：

> 请优化 XX 目录的 Triton Ascend 算子，从 baseline 开始按多轮迭代优化

主 agent 会自动决定每轮方向，派发子 CLI 执行，收集结果并判定是否继续。由于需要进行多轮迭代，该项目消耗 token 量较大，需注意费用开销。

## 文件结构

```
triton-optimizer-multi-agent/
├── AGENTS.md                           # 主 Orchestrator Agent 定义
├── CLAUDE.md -> AGENTS.md
├── init.sh                             # 安装脚本
├── config.json                         # 目标加速比 / 最大轮次 / 算法选择配置
├── .claude-plugin/plugin.json          # 插件元数据
├── .claude/references/                 # agent 协议文档
│   ├── orchestrator-loop.md            # 主 agent 完整规范
│   ├── algorithms/                     # 子 agent 算法规范
│   │   ├── index.md                    # 算法清单
│   │   ├── naive/round.md              # naive 算法子 agent 规范
│   │   ├── mcts/round.md               # MCTS 算法子 agent 规范
│   │   ├── fusion-search/round.md      # fusion-search 算法子 agent 规范
│   │   └── fusion-search/route.md      # fusion-search route 子 agent 规范
│   └── strategy-framework.md           # 方向/baseline 策略接口规范
├── skills/triton-agent-loop/
│   ├── SKILL.md
│   ├── references/                     # round-contract.md
│   ├── schemas/                        # task_manifest, round_result JSON schema
│   └── scripts/                        # dispatch_round.py, orchestrator_loop.py 等
├── hooks/                              # SessionStart/End, PreToolUse guard
└── template/                           # 算子类别经验模板
```

## 算子类别经验模板（template/）

`template/` 目录包含按算子类别组织的经验参考文档（如 `attention.md`、`normalization.md`、`convolution.md`、`quantization.md` 等）。`init.sh` 会将该目录**复制**到项目（或全局配置）的 `.claude/template/` 下，作为子 agent 的优化约束与经验参考。

你可以自由：

- **添加新类别**：参考 `template/template.md` 的骨架新建 `{category}.md`；
- **删除不需要的类别**：直接从 `template/` 移除对应文件，再重新执行 `init.sh`；
- **修改现有内容**：调整约束、骨架、陷阱等，使其更贴合你的算子集合；
- **复用官方模板**：从 `cannbot-skills_5171/plugins-official/triton-op-generator/template` 复制对应类别的 `.md` 文件到本项目的 `template/` 中。

注意：`init.sh` 执行的是复制而非软链接，因此项目运行时的 `.claude/template/` 与源码 `template/` 相互独立。对源码层 `template/` 的修改需要重新运行 `init.sh` 才能同步到项目；而运行中由 Orchestrator 追加的算子经验只影响项目级副本，不会回写插件源码。

## 核心脚本

- `skills/triton-agent-loop/scripts/dispatch_round.py`：启动子 CLI 执行单轮
- `skills/triton-agent-loop/scripts/orchestrator_loop.py`：主调度循环参考实现
- `skills/triton-agent-loop/scripts/transition_next_round.py`：原子化准备下一轮
- `skills/triton-agent-loop/scripts/check_stop_decision.py`：停止门禁
- `skills/triton-agent-loop/scripts/submit_round.py`：单轮产物校验

## 配置参数

`config.json` 控制优化目标、轮次限制、NPU 资源选择、子 agent 超时以及方向选择策略。

插件根目录的 `config.json` 提供全局默认值；如果目标算子项目根目录也存在同名文件，Orchestrator 会优先读取项目级配置进行覆盖。

```json
{
  "target_speedup": 5,
  "max_rounds": 20,
  "initial_worker_timeout": 7200,
  "max_worker_timeout": 28800,
  "npu_device_id": null,
  "npu_idle": {
    "aicore": 1.0,
    "aivector": 1.0,
    "npu_util": 1.0,
    "hbm": 10.0
  },
  "algorithms": {
    "naive": {
      "enabled": true,
      "params": {}
    },
    "mcts": {
      "enabled": true,
      "params": {
        "c_uct": 1.414,
        "c_pw": 2.0,
        "alpha": 0.5,
        "failure_threshold": 3,
        "reward_output_error": -2,
        "reward_compile_error": -3
      }
    },
    "fusion-search": {
      "enabled": true,
      "params": {
        "max_routes": 3,
        "min_expected_benefit": 0.05,
        "enable_math_transforms": true,
        "route_timeout": 3600,
        "crossover_rate": 0.7,
        "mutation_rate": 0.15,
        "elite_count": 1,
        "exploration_count": 1
      }
    }
  }
}
```

| 参数 | 类型 | 默认值 | 说明 |
|---|---|---|---|
| `target_speedup` | `number` | `5` | 目标几何平均加速比。当某轮 `best_speedup` 达到或超过该值时，C1 条件满足，可触发停止。 |
| `max_rounds` | `integer` | `20` | 最大优化轮次。非 idle round 数达到该值时，C2 条件满足，强制停止。 |
| `initial_worker_timeout` | `integer` | `7200` | 单个子 CLI 的初始超时时间（秒）。 |
| `max_worker_timeout` | `integer` | `28800` | 单个子 CLI 的最大超时时间（秒）。如果某轮超时，下一轮超时时间会翻倍，但不超过该上限。 |
| `npu_device_id` | `integer` \| `null` | `null` | 指定用于 benchmark 的 NPU 设备 ID。为 `null` 时由空闲检测自动选择。 |
| `npu_idle.aicore` | `number` | `1.0` | AICore 使用率阈值（%）。设备 AICore 使用率低于该值才视为空闲。 |
| `npu_idle.aivector` | `number` | `1.0` | AIVector 使用率阈值（%）。 |
| `npu_idle.npu_util` | `number` | `1.0` | NPU 整体利用率阈值（%）。 |
| `npu_idle.hbm` | `number` | `10.0` | HBM 显存使用率阈值（%）。 |
| `algorithms.<name>.enabled` | `boolean` | `true` | 是否允许用户选择该算法。 |
| `algorithms.<name>.params` | `object` | `{}` | 该算法对应 Strategy 的构造参数。 |

当前激活的算法由用户在会话开始时选择，并保存在 `.triton-agent/state-{op_name}-{algorithm}-{run_tag}.json` 中，不再写入 `config.json`。旧字段 `algorithm`、`strategy`、`fusion_search` 已移除。

`npu_idle` 的阈值也可通过环境变量覆盖：`TRITON_NPU_IDLE_MAX_AICORE`、`TRITON_NPU_IDLE_MAX_AIVECTOR`、`TRITON_NPU_IDLE_MAX_NPU_UTIL`、`TRITON_NPU_IDLE_MAX_HBM`。设置 `TRITON_SKIP_NPU_IDLE_CHECK=1` 可完全跳过空闲检测。

## 方向选择策略

主 Orchestrator 每轮选择优化方向时，会根据当前激活的算法加载对应的 `Strategy`（策略名与算法名一致）。策略配置位于 `config.json` 的 `algorithms.<algorithm>.params` 中。

```json
{
  "target_speedup": 5,
  "max_rounds": 10,
  "algorithms": {
    "naive": {
      "enabled": true,
      "params": {}
    }
  }
}
```

### 内置策略

| 名称 | 说明 | 适用场景 |
|---|---|---|
| `naive` | 基于历史 `direction_yield`、方向兼容性、策略兼容性、recency 选择 baseline；方向固定为 `pattern: custom`，由 LLM 根据算子分析自主探索。 | 默认生产策略 |
| `mcts` | 蒙特卡洛树搜索：把每个 `opt-round-*` 视为树节点，用 UCT + Progressive Widening 决定从哪个历史 round 扩展；具体优化方向由子 agent 基于 MCTS 上下文自主选择。 | 需要结构化探索、希望利用历史奖励指导 baseline 选择的场景 |
| `fusion-search` | 融合搜索策略：为 fusion-search 算法选择 baseline；融合分析、基因池进化与 route 并行派发由子 agent 内部完成。 | 可融合多原子算子 |
| `random_stub` | 测试用策略，直接返回 `pattern: custom`，让 LLM 自主选择方向；仅用于验证框架可插拔性。 | 测试/演示 |

### mcts 策略配置

`mcts` 把每次优化尝试建模为 MCTS 树中的一个节点。父节点是本轮 baseline 对应的 round，子节点是本轮产生的新 round。UCT 选择决定下一轮从哪个历史 round 出发；具体优化方向由子 agent 基于 MCTS 上下文自主选择，不再依赖固定 pattern 目录。

```json
{
  "algorithms": {
    "mcts": {
      "enabled": true,
      "params": {
        "c_uct": 1.414,
        "c_pw": 2.0,
        "alpha": 0.5,
        "failure_threshold": 3,
        "reward_output_error": -2,
        "reward_compile_error": -3
      }
    }
  }
}
```

| 参数 | 类型 | 默认值 | 说明 |
|---|---|---|---|
| `c_uct` | `number` | `1.414` | UCT 探索常数。 |
| `c_pw` | `number` | `2.0` | Progressive Widening 系数。 |
| `alpha` | `number` | `0.5` | Progressive Widening 指数。 |
| `failure_threshold` | `integer` | `3` | 节点累计失败次数达到该值时剪枝。 |
| `reward_output_error` | `number` | `-2` | 输出/精度校验失败的奖励。 |
| `reward_compile_error` | `number` | `-3` | 编译失败或无有效结果的奖励。 |

MCTS 树状态持久化在 `triton_ascend_output/{op_name}-{algorithm}-{run_tag}/.strategy_state/{op_name}_mcts_tree.json`。

### 添加自定义策略

1. 在 `skills/triton-agent-loop/scripts/strategies/` 下新建文件，例如 `genetic.py`（或 `genetic/strategy.py`）。
2. 继承 `Strategy` 并实现三个抽象方法（`select_next_round`、`score_baseline_for_direction`、`classify_direction_yield`）。
3. 设置类属性 `name = "genetic"`。
4. 在 `config.json` 的 `algorithms` 映射中引用：

```json
{
  "algorithms": {
    "genetic": {
      "enabled": true,
      "params": {
        "population_size": 20,
        "mutation_rate": 0.1,
        "crossover_rate": 0.7
      }
    }
  }
}
```

5. （可选）若策略需要跨 round 保存状态，可读写：

```
triton_ascend_output/{op_name}-{algorithm}-{run_tag}/.strategy_state/genetic.json
```

### 未来可扩展的复杂算法示例

| 策略 | 核心思想 | 状态存储 |
|---|---|---|
| `genetic` | 维护一组 kernel 配置个体，按 speedup 作为 fitness 进行选择、交叉、变异 | `.strategy_state/genetic.json` |
| `bayesian` | 用高斯过程建模 direction/hyperparameter → speedup 的映射，选择采集函数最大点 | `.strategy_state/bayesian.json` |
| `bandit` | 把每个 direction 视为一个臂，用 UCB/Thompson Sampling 动态平衡探索与利用 | `.strategy_state/bandit.json` |

这些策略只需新增文件并在 `config.json` 的 `algorithms` 映射中登记，无需修改 Orchestrator、`transition_next_round.py` 或 schema。

## 约束

- 主 agent 不执行 Phase 0-8 细节。
- 子 agent 不做跨 round 判定、不读取历史 `round_index.json`。
- 所有 C1/C2/C3/D4 判定保留在主 agent。
- 子 agent 禁止写入 `round_index.json`、`.claude/template/*.md`、`.triton-agent/state-{op_name}-{algorithm}-{run_tag}.json`。
- **`naive` 策略不再主动建议停止**（`can_recommend_stop=false`）。停止仅由 C1（达标）或 C2（`max_rounds`）触发；若目标未达成，任务将持续运行到 `max_rounds`，请据此评估 token/NPU 成本。
