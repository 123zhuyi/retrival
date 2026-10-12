---
description: triton-optimizer-multi-agent 主 Orchestrator Agent 的完整执行规范。包含初始化、task_manifest/round_result 协议、子 CLI 派发、策略驱动的方向/baseline 选择、C1/C2 停止/继续判定、最终输出。本文件是主 agent 的权威规范。
---

> **本文件是主 Orchestrator Agent 的完整执行规范。**
> - 由 `AGENTS.md` 在每轮调度前通过 Read 引用。
> - 与各算法 sub agent 规范互补：本文件只规定主 agent 行为，Phase 0-8 的具体执行规范由 `.claude/references/algorithms/{algorithm}/round.md` 规定。
> - 所有路径均为相对插件根目录（`.claude/references/...`、`skills/triton-agent-loop/...`），与 `init.sh` 安装后的布局一致。

---

## 一、多智能体架构总览

```
┌─────────────────────────────────────────────────────────────┐
│  主 Claude Code (Orchestrator)                              │
│  - 维护 .triton-agent/state-{op_name}-{algorithm}-{run_tag}.json                            │
│  - 决定每轮 strategy / hypothesis / direction / baseline    │
│  - 生成 task_manifest.json                                  │
│  - 调用 dispatch_round.py 启动子 CLI                        │
│  - 解析 round_result.json                                   │
│  - 委托策略生成方向/baseline，执行 C1/C2 判定               │
│  - 最终输出全局最优结果                                     │
└─────────────────────────────────────────────────────────────┘
                              │
                              ▼ spawn one sub-process per round
┌─────────────────────────────────────────────────────────────┐
│  子 Claude Code CLI (Per-Round Worker)                      │
│  - 读取 task_manifest.json                                  │
│  - 在独立上下文中执行 Phase 0 → 1 → 2 → 3 → 4 → 5 → 6 → 7 → 8   │
│  - 产出当前 round 的全部产物                                │
│  - 写入 round_result.json 后退出                            │
└─────────────────────────────────────────────────────────────┘
```

**边界原则**：
- 主 agent 不执行 Phase 0-8 细节。
- 子 agent 不做跨 round 判定、不读取历史 `round_index.json`、不更新 `round_index.json`。
- 子 agent 只能访问其 `baseline_dir`（父 round）和当前 `work_dir`。
- 经验沉淀（template 更新）由主 agent 在所有 round 结束后统一完成。

---

## 二、状态与产物

### 2.1 运行态目录

```
.triton-agent/        ← 主 agent 创建、更新；子 agent 只读
  └── state-{op_name}-{algorithm}-{run_tag}.json      ← 当前 round 状态快照

triton_ascend_output/  ← 算子级工作区
  └── {op_name}-{algorithm}-{run_tag}/
        ├── round_index.json          ← 主 agent 更新
        ├── opt-note.md               ← 主 agent 追加
        ├── operator_experience.md    ← 算子级私有经验（子 agent 可读、可追加）
        ├── global_baseline/          ← 用户任务文件的只读副本，用于全局加速比计算
        │     ├── {op_name}_generated.py
        │     └── baseline_info.json
        ├── {op_name}_final.py        ← 主 agent 在停止时生成
        ├── opt-round-1/              ← Round 1 工作目录
        │     ├── summary.json        ← 子 agent 写入
        │     ├── report.md           ← 子 agent 写入
        │     ├── {op_name}_generated.py
        │     ├── round_result.json   ← Phase 8：子 agent 写入
        │     ├── session.jsonl       ← Phase 7：会话导出（子 agent 写入）
        │     ├── session.md          ← Phase 7：会话渲染（子 agent 写入，可选）
        │     ├── kkb_ingestion.md    ← Phase 6：知识库摄入摘要（子 agent 写入，可选）
        │     └── output/             ← Phase 3/4/5 产物
        ├── opt-round-2/
        └── ...

# 知识库副作用（.claude/skills/triton-knowledge-retrieval/ 目录下，跨任务共享）
# 各子 Skill 自主检索时写入缓存，子 Agent Phase 6 写入经验 YAML 和 log.md
.claude/skills/triton-knowledge-retrieval/
├── knowledge/cache/
│   ├── codegen_init__<keywords>.md           # designer/coding 检索写入
│   ├── compile_error__<keywords>.md          # coding 模式 3 检索写入
│   ├── precision_debug__<keywords>.md        # coding 模式 4 检索写入
│   ├── perf_tuning__<keywords>.md            # optimizer 检索写入
│   └── README.md
├── log.md                                    # append-only：F1-F5 / INGESTION / MAINTENANCE_REPORT
└── knowledge/raw/experiences/<category>/
    └── exp_*_<YYYYMMDD>.yaml                 # Phase 6 自动摄入
```

### 2.2 state-{op_name}-{algorithm}-{run_tag}.json 格式

```json
{
  "op_name": "layer_norm",
  "status": "active | completed | failed",
  "current_round": 2,
  "rounds": {
    "1": {"status": "completed", "round_dir": "opt-round-1"},
    "2": {"status": "active", "round_dir": "opt-round-2"}
  },
  "phase": "round_active",
  "work_dir": "triton_ascend_output/layer_norm/opt-round-2/",
  "last_phase": 0,
  "performance_history": [
    {"round": 1, "speedup_vs_torch": 0.85, "target_reached": false}
  ]
}
```

### 2.3 round_index.json 格式

与单 agent 版本保持一致，并新增方向级因果元数据：

```json
[
  {
    "round_index": 1,
    "round_dir": "opt-round-1",
    "direction": "pattern: custom",
    "hypothesis": "...",
    "evidence_sources": ["model_knowledge"],
    "analysis_policy": "pattern_entry",
    "effective_metric_source": "kernel",
    "status": "success",
    "best_speedup": 0.85,
    "round_strategy": "exploration",
    "direction_yield": "low",
    "yield_reason": "plateau_after_2_rounds",
    "absolute_gain_vs_baseline": 0.02,
    "absolute_gain_vs_global_baseline": -0.15,
    "baseline_dir": "global_baseline",
    "global_baseline_dir": "global_baseline",
    "global_baseline_speedup": 1.0,
    "consecutive_low_gain_count": 2,
    "round_type": "active",
    "design_summary": "Attempt vectorized elementwise with 64-wide tiles; under-utilizes Ascend cores on small shapes.",
    "iter_count": 2,
    "opt_iter_count": 1,
    "key_fixes": ["added mask for non-multiple-of-64 shapes"],
    "bottlenecks_observed": ["core under-utilization on small shapes"]
  }
]
```

字段说明：
- `direction_yield`: `high | medium | low | failed`，该方向尝试的质量标签。
- `yield_reason`: 产生该标签的原因。
- `absolute_gain_vs_baseline`: 本 round 相对其所用 baseline 的绝对加速比提升。
- `absolute_gain_vs_global_baseline`: 本 round 相对全局基线（`global_baseline/`）的绝对加速比提升。
- `baseline_dir`: 本 round 初始代码所复制的 baseline round 目录名（父 round），用于继续优化。
- `global_baseline_dir`: 全局基线目录名，仅用于加速比计算，**禁止修改**。
- `global_baseline_speedup`: 全局基线的加速比，固定为 `1.0`。
- `round_type`: `active | idle`；idle 表示最终代码与历史最佳 baseline 字节级相同，不得再被选为 baseline。
- `design_summary`: 一句话设计思路/算法图摘要，由子 agent 写入。
- `iter_count` / `opt_iter_count`: Phase 3 / Phase 4 迭代次数。
- `key_fixes`: 本 round 关键修复列表。
- `bottlenecks_observed`: 观察到的性能/编译瓶颈列表。

---

## 三、主流程

### 3.1 初始化

1. `session_start.py` 检测 workspace。
2. 若存在有效 `.triton-agent/state-{op_name}-{algorithm}-{run_tag}.json` 和 `triton_ascend_output/{op_name}-{algorithm}-{run_tag}/round_index.json`，则恢复会话。
3. 若不存在，从用户输入解析：
   - `op_name`
   - `arch`（默认 `ascend910b1`）
   - 输入模式（Mode A / Mode B）
   - 输入文件路径
4. 读取 `config.json` 的 `algorithms` 映射；若 `state-{op_name}-{algorithm}-{run_tag}.json` 中无已选算法，使用 `AskUserQuestion` 列出 `enabled=true` 的算法让用户选择，并将选择结果写入 `state-{op_name}-{algorithm}-{run_tag}.json`。
5. 创建 `triton_ascend_output/{op_name}-{algorithm}-{run_tag}/opt-round-1/`。
6. 创建 `triton_ascend_output/{op_name}-{algorithm}-{run_tag}/global_baseline/`，将用户提供的任务描述文件复制为 `{op_name}_generated.py`，并写入 `baseline_info.json`。该目录**只读**，用于全局加速比计算。
7. 创建 `triton_ascend_output/{op_name}-{algorithm}-{run_tag}/operator_experience.md`，供子 agent 记录本算子的私有经验。
8. 写入 `state-{op_name}-{algorithm}-{run_tag}.json`，`current_round=1`，`last_phase=0`，`algorithm=<用户选择的算法>`。
9. 确定 Round 1 的 `round_strategy`、`analysis_policy`、`hypothesis`、`direction`、`evidence_sources`；Round 1 的 `baseline_dir` 为 `global_baseline`。

### 3.2 Round 循环

每轮执行前需明确：子 agent 严格遵循 `task_manifest.json` 中 `algorithm_reference` 指向的算法规范（如 `.claude/references/algorithms/naive/round.md`、`.claude/references/algorithms/mcts/round.md`、`.claude/references/algorithms/fusion-search/round.md`）执行 Phase 0 → 8，其流程与 `plugins-official/triton-op-generator/CLAUDE.md` 的 Phase 0-7 保持一致（包含 `freeze_baseline`、Phase 2 三层检查门、Phase 4 普通迭代 / IR 多轮迭代 / simulator 采集驱动分支、Phase 6 知识库摄入与维护等）。子 agent 的 Phase 8 不再执行经验沉淀，而是改为 **Round Contract 校验**（`submit_round.py`）并写入 `round_result.json`。

每轮执行：

```
1. 生成 task_manifest.json
2. 调用 dispatch_round.py 启动子 CLI
3. 等待子 CLI 退出并读取 round_result.json
4. 更新 round_index.json 与 opt-note.md
5. 调用 strategy.record_round_result(...)（若策略需要持久状态，如 MCTS）
6. 调用 strategy.select_next_round(...) 获取 RoundDecision
7. 执行 §4 停止/继续判定
8. 若继续 → 调用 transition_next_round.py → 回到 1
9. 若停止 → 进入 §5 最终输出
```

**关键变化（v2）**：

- C3（plateau）和 D4（direction exhausted）不再由 Orchestrator 全局判定，而是下沉到各 **Strategy** 内部（如 `naive` 在 `strategy.py` 中自行检测）。
- 不再创建 `opt-round-N-switch` 目录；方向切换只作为 `round_index.json` 和 `opt-note.md` 中的元数据记录。
- 策略通过统一的 `RoundDecision` 信封返回决策；Orchestrator 只补全策略未提供的字段（如 baseline）。

---

### 3.3 子 CLI 超时处理

主 agent 调用 `dispatch_round.py` 时**必须显式传入 `--timeout`**，并维护一个当前会话级的 `current_timeout` 变量。

**默认值**（可在 `config.json` 中覆盖）：

```json
{
  "initial_worker_timeout": 7200,
  "max_worker_timeout": 28800
}
```

- `initial_worker_timeout`：Round 1 使用的子 CLI 超时时间，默认 7200s。
- `max_worker_timeout`：允许的最大超时时间，默认 28800s（8 小时）。

**超时后的处理流程**：

1. `dispatch_round.py` 在子 CLI 运行超过 `--timeout` 后将其 `kill`，并返回：

   ```json
   {"status": "timeout", "pid": 12345}
   ```

2. 主 agent 打印子 CLI PID（以及 `kill <pid>` 提示），便于用户手动清理孤儿进程。
3. 主 agent 在 `state-{op_name}-{algorithm}-{run_tag}.json` 的 `performance_history` 中追加一条超时记录：

   ```json
   {"round": 3, "speedup_vs_torch": 0.0, "target_reached": false, "timeout": true}
   ```

4. 计算下一轮超时时间：

   ```
   next_timeout = min(current_timeout * 2, max_worker_timeout)
   ```

   即：7200 → 14400 → 28800，达到上限后不再增加。
5. 调用 `transition_next_round.py` 准备下一轮，并在下一轮启动 `dispatch_round.py` 时使用 `next_timeout`。

**关键约束**：

- **严禁在子 CLI 运行中途中修改其超时时间**。已经启动的子 CLI 超时后只能被 kill，不能“续命”。
- 当前 round 因超时未产生有效 `round_result.json`，因此**不更新 `round_index.json`**。
- 子 CLI 的实时日志会写入 `work_dir/sub_agent.log` 和 `work_dir/sub_agent_readable.log`，超时后可通过这两个文件定位卡死位置。

### 3.4 子 CLI 的 NPU 空闲卡选择

`dispatch_round.py` 在启动子 CLI 前会调用 `check_npu_idle.py` 选择一张**空闲**的 NPU 卡：

1. 枚举所有计算芯片（`npu-smi info -m`，跳过 MCU）。
2. 查询每张卡的利用率（`npu-smi info -t usages -i <npu_id>`）。
3. 空闲判定阈值（优先级从高到低）：
   - 环境变量：`TRITON_NPU_IDLE_MAX_AICORE` / `_AIVECTOR` / `_NPU_UTIL` / `_HBM`
   - `config.json` 中的可选 `npu_idle` 对象：
     ```json
     {
       "npu_idle": {
         "aicore": 1.0,
         "aivector": 1.0,
         "npu_util": 1.0,
         "hbm": 10.0
       }
     }
     ```
   - 内置默认值：`aicore=1%`, `aivector=1%`, `npu_util=1%`, `hbm=10%`
4. 若用户显式指定 `TRITON_NPU_DEVICE_ID=<id>`，则只接受该卡；不空闲时直接报错。
5. 否则自动选择第一张空闲卡，并将卡号写入 `work_dir/npu_device.json`，同时向子 CLI 环境注入：
   - `ASCEND_VISIBLE_DEVICES=<id>`
   - `NPU_CALCULATE_DEVICE=<id>`
   - `ASCEND_RT_VISIBLE_DEVICES=<id>`

若服务器上没有空闲卡，`dispatch_round.py` 会返回 `status: "error"`，主 agent 不得继续下一轮。调试或纯 CPU 环境可设置 `TRITON_SKIP_NPU_IDLE_CHECK=1` 跳过检测。

---

## 四、停止 / 继续判定

子 agent 返回 `round_result.json` 后，主 agent 按以下顺序执行判定链。

### 4.1 读取与校验

- 读取 `round_result.json`。
- 校验 schema（`skills/triton-agent-loop/schemas/round_result.schema.json`）。
- 若 `status == "failed"`，记录失败原因，但仍可进入判定（视失败类型决定是否继续）。
- 读取该 round 的 `summary.json` 作为冗余校验。

### 4.2 调用策略获取 RoundDecision

```python
decision = strategy.select_next_round(
    round_index,
    pattern_index_path,   # legacy argument, ignored; directions are LLM-driven
    op_category=op_category,
    current_policy=current_policy,
    context={"op_dir": str(op_dir)},
)
```

`pattern_index_path` 是历史接口参数，策略不应再基于它选择方向。当前所有方向均为 **LLM 自主探索**：`naive` / `random_stub` 返回 `direction="pattern: custom"`，由子 agent 决定具体优化方向；`mcts` 返回 `direction=None`，子 agent 基于 MCTS 上下文自主决定。

`decision` 是 `RoundDecision` 信封，包含以下字段：

| 字段 | 类型 | 说明 |
|------|------|------|
| `direction` | `str | None` | 下一轮优化方向；`"pattern: custom"` 表示由 LLM 自主探索。MCTS 等节点选择策略返回 `None`，由子 agent 自主决定。 |
| `hypothesis` | `str | None` | 方向/节点假设。 |
| `round_strategy` | `str | None` | 如 `exploration`、`focused_tuning`、`structural_change`。 |
| `analysis_policy` | `str | None` | 如 `pattern_entry`、`bottleneck_analysis`、`ir_supported`。 |
| `evidence_sources` | `list[str]` | 证据来源。 |
| `baseline` | `BaselineProposal` | `mode="none" | "single" | "multiple"`，及 `round_dirs`。对 MCTS，`round_dirs[0]` 为选中的父 round；虚拟节点映射为空字符串，Orchestrator 将回退到 `global_baseline`。 |
| `parent_round_dir` | `str | None` | 选中的父 round 目录名（MCTS 使用）。 |
| `selected_round_dir` | `str | None` | 由 Orchestrator 在创建新 round 后回填的新 round 目录名。 |
| `switch_marker` | `SwitchMarker | None` | 方向切换审计信息，不为 `None` 时记录到 `round_index.json` 与 `opt-note.md`。 |
| `algorithm_recommends_stop` | `bool` | 策略是否建议停止。 |
| `stop_reason` | `str | None` | 建议停止的原因。 |
| `metadata` | `dict` | 策略附加数据（如 `mcts_parent_id`、`mcts_selection_path`）。 |

策略能力通过 `StrategyCapabilities` 声明：

| 能力 | 含义 |
|------|------|
| `can_propose_direction` | 策略能给出 `direction`。 |
| `can_propose_baseline` | 策略能给出 `baseline.round_dirs`。 |
| `can_propose_multiple_baselines` | 策略能给出多个 baseline 并指定合并规则。 |
| `can_select_round` | 策略能在已有 round（节点）中选择下一个要扩展的父 round；此时 `direction` 为 `None`。 |
| `can_decide_switch_internally` | 策略内部处理方向切换，不依赖 Orchestrator 的 C3/D4。 |
| `can_recommend_stop` | 策略能设置 `algorithm_recommends_stop=True`。 |
| `needs_persistent_state` | 策略需要在 `.strategy_state/` 中读写状态。 |

### 4.3 补全策略未提供的字段

- 若 `decision.direction is None`：
  - 若策略 `can_select_round=True`（如 MCTS），接受 `None`，由子 agent 自主决定优化方向。
  - 若策略 `can_propose_direction=True` 但未给出方向，回退到 `direction_selector.select_next_direction()`。
  - 否则，使用默认方向 `"pattern: custom"`。
- 若 `decision.baseline.mode == "none"` 且已选定方向，Orchestrator 使用 `strategy.score_baseline_for_direction()` 在所有成功且非 `idle` 的历史 round 中打分，选择最高分作为 baseline。
- 若策略选中的父 round 无代码（如 MCTS 虚拟节点），`baseline_dir` 回退到 `global_baseline/`。

### 4.4 C1/C2 判定

| 条件 | 检查方法 | 结果 |
|------|----------|------|
| C1 target_reached | `best_speedup >= target_speedup` | 满足 → 进入停止门禁 |
| C2 max_rounds | 非空转轮次数 >= max_rounds | 满足 → 进入停止门禁 |

### 4.5 停止门禁

无论 C1/C2 触发，还是 `algorithm_recommends_stop=True`，都必须执行：

```bash
python3 skills/triton-agent-loop/scripts/check_stop_decision.py \
    --op-dir triton_ascend_output/{op_name}-{algorithm}-{run_tag} \
    --current-round {round_index} \
    --best-speedup {best_speedup} \
    --target-speedup {target_speedup} \
    --max-rounds {max_rounds} \
    [--algorithm-recommends-stop]
```

- `can_stop: true` → 允许停止。
- `can_stop: false` → 禁止停止，必须继续；此时应使用策略返回的方向/baseline 进入下一轮。

### 4.6 方向切换元数据

策略在决策信封中返回 `switch_marker` 时，Orchestrator 在调用 `transition_next_round.py` 时通过 `--switch-marker '{...}'` 传入。脚本将切换原因写入：

- `round_index.json` 对应条目的 `switch_reason`、`previous_direction`、`new_direction`。
- `opt-note.md` 的切换记录段落。

**不再创建 `opt-round-N-switch` 目录**；切换只是下一个普通 round 的元数据。

### 4.7 Step 4：进入下一轮

判定进入下一轮后，**必须**调用 `transition_next_round.py`。`--algorithm` 为当前选中的算法；`--strategy-name` 与 `--strategy-params` 取自 `config.json` 中 `algorithms.<algorithm>` 的配置。当显式传入 `--strategy-params` 时，该参数优先于 config 中的对应配置。

```bash
python3 skills/triton-agent-loop/scripts/transition_next_round.py \
    --op-dir triton_ascend_output/{op_name}-{algorithm}-{run_tag} \
    --current-round {round_index} \
    --next-round-strategy {next_round_strategy} \
    --next-analysis-policy {next_analysis_policy} \
    --next-hypothesis "{next_hypothesis}" \
    --next-direction "{next_direction}" \
    --next-evidence-sources '{next_evidence_sources}' \
    --algorithm {algorithm} \
    --strategy-name {algorithm} \
    --strategy-params '{algorithm_params}' \
    --baseline-dir {baseline_dir} \
    [--switch-marker '{"reason": "...", "previous_direction": "...", "new_direction": "..."}'] \
    --current-success-criteria "{本轮成功标准}" \
    --current-key-learnings "{本轮关键学习}" \
    --global-baseline-dir global_baseline \
    --global-baseline-speedup 1.0
```

脚本原子化执行：创建 `opt-round-{N+1}/`、更新 `state-{op_name}-{algorithm}-{run_tag}.json`、复制 baseline、写入 `summary.json`、更新 `round_index.json`、追加 `opt-note.md`。

---

## 五、最终输出

当 C1/C2 满足且 `check_stop_decision.py` 通过时：

1. 遍历 `round_index.json`，选择 `best_speedup` 最高的 round。
2. 复制 `{op_name}/opt-round-{best_round}/{op_name}_generated.py` → `{op_name}/{op_name}_final.py`。
3. 执行经验沉淀：读取最佳 round 的 `summary.json`、`report.md` 以及算子级 `operator_experience.md`，更新项目本地的 `.claude/template/{category}.md`（`init.sh` 按项目复制模板；内容可在项目间手动同步）。
4. 输出最终总结报告，包含：
   - 最终代码来源 round
   - `best_speedup`
   - 是否达到 `target_speedup`
   - 总 round 数
   - 最终代码路径
   - 全局经验模板路径（如有更新）

---

## 六、与子 agent 的通信协议

子 agent 的全部行为由 `task_manifest.json` 中的 `algorithm_reference` 字段指向的算法规范规定（如 `.claude/references/algorithms/naive/round.md`、`.claude/references/algorithms/mcts/round.md`、`.claude/references/algorithms/fusion-search/round.md`），这些规范已与 `plugins-official/triton-op-generator/CLAUDE.md` 的 Phase 0-7 流程对齐。主 agent 只需确保 `task_manifest.json` 中的字段完整，并调用 `dispatch_round.py` 启动子 CLI；子 agent 会在独立上下文中自主完成 Phase 0-8，主 agent 不干预具体执行步骤。

### 6.1 输入：task_manifest.json

主 agent 在每轮启动子 agent 前写入：

```json
{
  "op_name": "layer_norm",
  "round_index": 2,
  "work_dir": "triton_ascend_output/layer_norm/opt-round-2",
  "baseline_dir": "triton_ascend_output/layer_norm/opt-round-1",
  "algorithm": "mcts",
  "algorithm_reference": ".claude/references/algorithms/mcts/round.md",
  "algorithm_params": {
    "c_uct": 1.414,
    "c_pw": 2.0,
    "alpha": 0.5,
    "failure_threshold": 3,
    "reward_output_error": -2,
    "reward_compile_error": -3
  },
  "round_strategy": "focused_tuning",
  "analysis_policy": "bottleneck_analysis",
  "hypothesis": "MCTS selects to expand from node opt-round-1 ...",
  "direction": "",
  "evidence_sources": ["mcts_tree"],
  "mode": "A",
  "input_files": {
    "task_desc": "...",
    "task_json": null,
    "gpu_kernel_ref": null,
    "gpu_perf_csv": null
  },
  "config": {
    "target_speedup": 5,
    "max_rounds": 10,
    "algorithms": {
      "naive": {"enabled": true, "params": {}},
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
  },
  "is_recovery": false,
  "worker_reference": ".claude/references/algorithms/mcts/round.md",
  "mcts_parent_id": "opt-round-1",
  "mcts_selection_path": [...],
  "mcts_pending_expansion": true,
  "lineage_summary": [
    {
      "round_index": 1,
      "round_dir": "opt-round-1",
      "direction": "pattern: custom",
      "best_speedup": 0.85,
      "direction_yield": "low",
      "hypothesis": "Autotune elementwise config.",
      "design_summary": "Autotune BLOCK_SIZE for elementwise; low yield due to small-shape under-utilization.",
      "iter_count": 2,
      "opt_iter_count": 1,
      "key_fixes": ["added mask for non-multiple-of-64 shapes"],
      "bottlenecks_observed": ["core under-utilization on small shapes"],
      "parent_round_dir": "global_baseline",
      "outcome": "low_yield"
    }
  ],
  "experience_file": "triton_ascend_output/layer_norm/operator_experience.md",
  "global_baseline_dir": "triton_ascend_output/layer_norm/global_baseline",
  "global_baseline_speedup": 1.0,
  "parent_round_journal": "triton_ascend_output/layer_norm/opt-round-1/round_journal.md"
}
```

> `algorithm` 与 `algorithm_reference` 必须一致，指向 `.claude/references/algorithms/{algorithm}/round.md`。
>
> `algorithm_params` 从 `config.json` 的 `algorithms.<algorithm>.params` 复制而来，供子 agent 读取当前算法的专属参数（如 MCTS 的 `c_uct`、fusion-search 的 `max_routes`）。
>
> `config` 块现在只包含全局参数和 `algorithms` 映射；旧字段 `algorithm`、`strategy`、`fusion_search` 已移除。
>
> `lineage_summary` 为从 root 到当前父 round 的优化历史链，已 enriched 设计摘要、迭代次数、关键修复、瓶颈等字段。`experience_file` 为算子级私有经验文件。`parent_round_journal` 为父 round 的详细日志路径，子 agent 可读但不可修改。`global_baseline_dir` 为只读全局基线目录，仅用于计算加速比。子 agent **不得修改** `global_baseline_dir` 中的文件。
>
> 子 agent 应在 `report.md` 之外写入 `{work_dir}/round_journal.md`，供后续 round 通过 `parent_round_journal` 读取。
>
> 对 `mcts` 算法，manifest 额外包含 `mcts_parent_id`、`mcts_selection_path`、`mcts_pending_expansion`，供子 agent 自主选择 direction。

### 6.2 输出：round_result.json

子 agent 退出前写入当前 `work_dir`：

```json
{
  "round_index": 2,
  "status": "success",
  "best_speedup": 1.35,
  "target_reached": false,
  "improvement_made": true,
  "phase4_entered": true,
  "optimized": true,
  "effective_metric_source": "kernel",
  "direction": "pattern: program-multiple-rows",
  "hypothesis": "...",
  "round_strategy": "focused_tuning",
  "work_dir": "triton_ascend_output/layer_norm/opt-round-2",
  "error": null,
  "artifacts": {
    "summary_json": "triton_ascend_output/layer_norm/opt-round-2/summary.json",
    "report_md": "triton_ascend_output/layer_norm/opt-round-2/report.md",
    "final_code": "triton_ascend_output/layer_norm/opt-round-2/layer_norm_generated.py"
  }
}
```

---

## 七、约束

| 约束 | 说明 |
|------|------|
| **主 agent 不执行 Phase 0-8** | 所有具体执行工作交给子 agent。 |
| **子 agent 不做跨轮判定** | 子 agent 不读取 `round_index.json`，不执行停止/继续判定。 |
| **C1/C2 归 Orchestrator** | Orchestrator 检查 C1/C2 并运行 `check_stop_decision.py` 门禁。 |
| **C3/D4 归 Strategy** | 收益递减和方向耗尽由各策略内部处理，通过 `RoundDecision` 返回；Orchestrator 不再全局执行 C3/D4。 |
| **策略驱动方向/baseline** | 方向、baseline、切换、停止建议必须由用户选择的算法对应的 Strategy 生成，禁止模型臆造。 |
| **经验沉淀归主 agent** | template 更新由主 agent 在停止后统一完成；经验分为两层：算子级 `operator_experience.md`（私有）和项目本地 `.claude/template/{category}.md`（`init.sh` 按项目复制，可在项目间手动同步）。 |
| **知识库摄入归子 agent** | 每轮 Phase 6 由子 agent 完成 `triton-knowledge-retrieval` 的 Source B/C/D 摄入；`.claude/skills/triton-knowledge-retrieval/` 是唯一允许跨任务共享副作用的目录。 |
| **全局基线只读** | `global_baseline/` 中的用户任务文件副本仅用于加速比计算，任何 agent 不得修改。 |
| **父 round baseline 可继续优化** | `baseline_dir` 是子 agent 的初始代码来源，允许修改；禁止将 `global_baseline_dir` 作为修改对象。 |
| **产物格式不变** | `summary.json`、`round_index.json`、`opt-note.md`、`report.md` 与单 agent 版本保持一致。 |
| **禁止修改原 triton-optimizer** | 本插件自包含于 `triton-optimizer-multi-agent/`。 |
| **串行调度** | 禁止同时启动多个子 CLI。fusion-search 模式下的内部并行派发发生在子 Agent 内部（`dispatch_parallel.py`），不受此约束。 |
| **停止门禁** | 输出最终报告前必须执行 `check_stop_decision.py`；`can_stop: false` 时禁止停止。 |
| **轮间过渡强制脚本** | 进入下一轮必须调用 `transition_next_round.py`，不得手动复制 baseline 或写状态文件。 |
| **上下文管理** | 主 agent 只保留结构化状态，不加载 iter/opt_iter 产物。 |
| **方向切换只写元数据** | 不再创建 `opt-round-N-switch` 目录；切换原因写入 `round_index.json` 与 `opt-note.md`。 |
| **源 benchmark 只读** | 用户数据目录（`npu_benchmark/`、`ascendc-kernelgen-data*/` 等）受 PreToolUse guard 保护，禁止 Write/Edit 与 Bash 删除/修改/覆盖，但允许 Read/Grep/Glob。 |

---

## 八、算法选择

### 8.1 `algorithms` 映射

`config.json` 使用 `algorithms` 映射定义每个可用算法是否启用及其参数：

```json
{
  "algorithms": {
    "naive": {"enabled": true, "params": {}},
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

| algorithm | algorithm_reference | 策略类 | 说明 |
|---|---|---|---|
| `naive` | `.claude/references/algorithms/naive/round.md` | `naive` | 默认模式，单 round 串行迭代优化 (Phase 0-8)。Direction 由 Orchestrator 通过 naive strategy 提前选定，子 agent 直接执行。 |
| `mcts` | `.claude/references/algorithms/mcts/round.md` | `mcts` | MCTS 树搜索模式。Orchestrator 通过 MCTS 选择父 round，子 agent 基于 MCTS 上下文自主决定本 round direction。 |
| `fusion-search` | `.claude/references/algorithms/fusion-search/round.md` | `fusion-search` | 融合搜索+遗传进化模式。Round 1: B1 原子分解 → B2 自动融合分析 → C0 基因编码空间设计 → D0 并行多路线搜索 → 构建初始基因池。Round 2+: 基因选择/交叉/变异 → profiling 驱动基因新增 → 并行派发进化路线 → 更新基因适应度。 |

### 8.2 manifest 中的 `algorithm_reference`

主 agent 在 `build_manifest()` 中根据当前激活的 `algorithm` 自动设置 `algorithm_reference` 和 `worker_reference`，并把 `config.json` 中 `algorithms.<algorithm>.params` 复制到 `algorithm_params`：

```python
algorithm_reference = f".claude/references/algorithms/{algorithm}/round.md"
manifest["algorithm"] = algorithm
manifest["algorithm_reference"] = algorithm_reference
manifest["algorithm_params"] = config["algorithms"][algorithm]["params"]
manifest["worker_reference"] = algorithm_reference  # 兼容旧代码
```

`dispatch_round.py` 优先从 manifest 读取 `algorithm_reference`，回退到 `worker_reference`；两者均缺失时回退到 `.claude/references/algorithms/naive/round.md`。

### 8.3 算法选择流程

1. Orchestrator 启动时读取 `config.json` 的 `algorithms` 映射，得到 enabled 的算法列表。
2. **扫描 `.triton-agent/` 目录**：查找 `state-{op_name}-*.json`。若命中唯一文件且其中已有 `algorithm`，直接使用（会话恢复）。
3. **旧工作区检测**：若只存在旧版 `state-{op_name}.json` 和旧版工作区 `triton_ascend_output/{op_name}/`，主 agent 必须使用 `AskUserQuestion` 向用户说明情况，报告推断出的算法、当前 round、迁移计划，并提供三个选项：
   - 继续使用推断算法；
   - 指定其他算法；
   - 放弃旧工作区，重新开始。
   用户确认后，主 agent 调用 `orchestrator_loop.py --migrate-legacy <algorithm>` 迁移旧工作区。
4. 若 state 中无算法且没有旧工作区（新任务），向用户展示 enabled 算法列表并由用户选择。
5. 选择结果写入：
   - `.triton-agent/state-{op_name}-{algorithm}-{run_tag}.json` 的 `algorithm` 和 `algorithm_reference` 字段
   - `task_manifest.json` 的 `algorithm` 和 `algorithm_reference` 字段

不再把选择结果写回 `config.json`。

### 8.4 派发行为

无论 `algorithm` 取何值，主 agent **始终只派发 1 个子 CLI**（通过 `dispatch_round.py`）。

- `naive` / `mcts`：子 CLI 在独立上下文中执行 Phase 0-8。
- `fusion-search`：子 CLI 在独立上下文中执行。**Round 1**：B1 原子分解 + B2 融合分析（一次性发现）+ C0 基因编码空间设计，然后**内部**调用 `dispatch_parallel.py` 并行启动多个 route 子 CLI（每个 route 走 `.claude/references/algorithms/fusion-search/route.md`，包含内部 3 轮迭代、知识编码进化、profiling 收敛和 G1-G4 知识管理），收集结果后选最优并构建初始基因池。**Round 2+**：从基因池出发做选择/交叉/变异 + profiling 驱动基因新增，生成进化路线后并行派发，更新基因适应度。

主 agent 不关心子 Agent 内部如何派发，只等待 `round_result.json` 返回。

### 8.5 算法状态继承

`transition_next_round.py` 在准备下一轮时，会根据策略的 `state_file_templates` 类属性从 `baseline_dir` 复制算法状态文件到 `next_round_dir`：

| algorithm | 状态文件 |
|---|---|
| `naive` | 无 |
| `mcts` | 无（MCTS 树状态由策略直接管理在 `triton_ascend_output/{op_name}-{algorithm}-{run_tag}/.strategy_state/`，不复制） |
| `fusion-search` | `gene_pool.json`, `fusion_plan.json` |

### 8.6 fusion-search 算法参数

`fusion-search` 的参数位于 `config.json` 的 `algorithms["fusion-search"].params` 下：

```json
{
  "algorithms": {
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
| `max_routes` | `integer` | `3` | 每轮并行派发的最大路线数 |
| `min_expected_benefit` | `number` | `0.05` | 融合方案的最低预期收益阈值（Round 1 B2 用） |
| `enable_math_transforms` | `boolean` | `true` | 是否对融合方案启用数学等价变换变种（Round 1 B2 用） |
| `route_timeout` | `integer` | `3600` | 每个 route 子 CLI 的超时时间（秒） |
| `crossover_rate` | `number` | `0.7` | 遗传交叉率（Round 2+ 用） |
| `mutation_rate` | `number` | `0.15` | 遗传变异率（Round 2+ 用） |
| `elite_count` | `integer` | `1` | 每轮保留的精英路线数（Round 2+ 用） |
| `exploration_count` | `integer` | `1` | 每轮用于验证新基因的探索路线数（Round 2+ 用） |

---

## 九、沟通风格

- 每轮子 agent 完成后仅输出一行 JSON 摘要。
- 停止时显式输出 C1/C2 判定、`algorithm_recommends_stop` 状态和 `check_stop_decision.py` 结果。
- 最终报告仅在所有 round 终止后输出一次。
- 错误时清晰描述 + 建议操作。
