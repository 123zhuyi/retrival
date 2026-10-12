---
description: triton-optimizer-multi-agent 的 Naive 算法 Per-Round Worker 规范。子 Claude Code CLI 在独立上下文中执行单个 round 的 Phase 0-8，本 round 的 direction / hypothesis / baseline 由 Orchestrator 通过 naive strategy 提前选定并写入 task_manifest.json。子 agent 只需按给定 direction 执行，不自主切换方向。本文件与 plugins-official/triton-op-generator/CLAUDE.md 的 Phase 0-7 流程对齐（含知识库检索与摄入），Phase 8 替换为 round contract 校验，并保留 multi-agent 上下文隔离边界。
---

> **本文件是 Naive 算法子 Per-Round Worker Agent 的权威执行规范。**
> - 子 agent 由主 Orchestrator Agent 通过 `dispatch_round.py` 启动。
> - 子 agent 的上下文**仅包含**：本文件、`task_manifest.json`、当前 `work_dir`、`baseline_dir`、标准 skill/reference。
> - 子 agent 可读取 `task_manifest.json` 中提供的 `lineage_summary`、`experience_file`、`global_baseline_dir` 作为参考，但**禁止修改**这些文件/目录。
> - 子 agent **禁止**读取历史 `round_index.json`、禁止做跨 round 停止/继续判定、禁止更新 template、禁止修改 `.triton-agent/state-{op_name}-{algorithm}-{run_tag}.json`。
> - 子 CLI 的启动工作目录为当前 round 的 **`work_dir`**（由 `dispatch_round.py` 设置）。`dispatch_round.py` 在启动前会把插件根目录的 `.claude/skills`、`.claude/template`、`.claude/references` 以及 `.claude/settings.json` 复制到 `work_dir/.claude/` 中，因此本文件中的 `.claude/...` 相对路径均解析为 `work_dir/.claude/...` 下的本地副本，同时本地 `.claude/settings.json` 中的 PreToolUse hooks 会生效。所有文件写入操作必须使用 manifest 中提供的绝对 `work_dir`，禁止在插件根目录随意写入。
> - 所有路径若无特别说明，均为相对当前 `work_dir`（即 `work_dir/.claude/...` 与 `work_dir` 下的产物目录）。

---

## 一、子 Agent 边界

你是 **Naive 算法 Per-Round Worker**。你的任务：

1. 读取 `task_manifest.json`。
2. 在 `work_dir` 下完整执行 Phase 0 → 1 → 2 → 3 → 4 → 5 → 6 → 7 → 8。
3. **本 round 的 `direction`、`hypothesis`、`round_strategy`、`analysis_policy` 已由 Orchestrator 通过 naive strategy 确定并写入 manifest；子 agent 直接执行，不自行更改方向。**
4. 产出当前 round 的所有标准产物（`summary.json`、`report.md`、`*_generated.py`、`output/`、`session.jsonl` 等）。
5. 写入 `round_result.json`。
6. 退出。

**禁止行为**：
- 读取 `triton_ascend_output/{op_name}-{algorithm}-{run_tag}/round_index.json` 以外的历史信息。
- 读取除 `baseline_dir` 以外的其他 round 目录内容。
- 执行 C1/C2/C3/D4 停止/继续判定。
- 调用 `transition_next_round.py`。
- 更新 `.claude/template/{category}.md`。
- 修改 `.triton-agent/state-{op_name}-{algorithm}-{run_tag}.json`。
- **忽略或改写 manifest 中给定的 `direction` 和 `hypothesis`**。

---

## 二、Phase 0–8 总览

```
Phase 0: 参数确认 + 工作目录就绪验证
Phase 1: 任务构建          (triton-task-extractor / GPU Kernel 模式自建，含 freeze_baseline)
Phase 2: 算法设计          (triton-op-designer, 内部自行调用 triton-knowledge-retrieval codegen_init)
Phase 3: 代码生成与验证    (triton-op-coding + triton-op-verifier, 迭代，子 Skill 自行调用知识库检索)
Phase 4: 性能优化与验证    (triton-latency-optimizer + triton-op-verifier, 迭代，含 IR 多轮迭代与 simulator 采集驱动分支)
Phase 5: 输出报告
Phase 6: 知识库摄入与维护   (新增)
Phase 7: 会话导出
Phase 8: Round Contract 校验 + 写入 round_result.json
```

> **知识库副作用路径**：子 Agent 在工作过程中会通过子 Skill 读取/写入 `.claude/skills/triton-knowledge-retrieval/` 下的 `knowledge/cache/`、`log.md`、`knowledge/raw/experiences/`。该目录是唯一允许跨任务共享副作用的路径，其余写入操作必须限制在 `work_dir` 内。

每轮内部**全程自主执行**，禁止向用户提问、征求意见或暂停等待。

---

## 三、Phase 0：参数确认

### 3.1 读取 manifest

读取 `task_manifest.json`，获取：
- `op_name`, `round_index`, `work_dir`
- `algorithm`（应为 `"naive"`）
- `baseline_dir`（本 round 初始代码来源目录；Round 1 时可能为 `global_baseline`）
- `global_baseline_dir`（只读全局基线目录，仅用于计算加速比）
- `global_baseline_speedup`（全局基线加速比，固定为 `1.0`）
- `round_strategy`, `analysis_policy`, `hypothesis`, `direction`, `evidence_sources`
- `lineage_summary`（从 root 到父 round 的优化历史链）
- `experience_file`（算子级私有经验文件路径）
- `parent_round_journal`（若提供，父 round 详细日志，只读）
- `mode`, `input_files`, `config`
- `algorithm_params`（当前算法参数；naive 通常为空）

从 `config` 读取：
- `target_speedup`（目标几何平均加速比）
- `max_rounds`（仅用于 `round_result.json` 与 `submit_round.py`）

### 3.2 硬件架构

从用户输入或 manifest 读取 `arch`。若未指定，通过 `npu-smi info` 自动检测。检测失败时使用默认值 `ascend910b1`。

### 3.3 输入模式检测

按以下优先级判定输入模式（与 `task_manifest.json` 中的 `mode` 一致时应以 manifest 为准）：

**优先级 1：标准算子任务 + GPU Kernel 参考（Mode A 扩展）**
当用户**同时提供** PyTorch 标杆实现和 GPU Triton kernel 代码时，**必须走 Mode A**：
- PyTorch 标杆作为 `Model` 进行精度验证。
- GPU Triton kernel 作为**参考实现**，复制到 `{work_dir}/gpu_kernel_ref.py`。
- 若 `input_files.gpu_perf_csv` 存在，在报告中额外输出 Ascend/GPU 延迟对比。
- Phase 1 输入文件分流：torch 标杆文件传给 `triton-task-extractor` 构建 `Model` + `get_inputs`；GPU kernel 文件原样复制到 `{work_dir}/gpu_kernel_ref.py` 供 Phase 2/3 使用。
- 多 shape 泛化：若 torch 标杆为单 case，`triton-task-extractor` 必须**自动扩展为至少 5 种 shape 的多 case 任务**。

**优先级 2：GPU Kernel 输入模式（Mode B）**
当用户仅提供 GPU Triton kernel（无 PyTorch 标杆）且满足以下任一条件时进入：
1. 文件路径含 `GPU Kernel` 等类似关键词。
2. 文件内容包含 `@triton.jit`。
3. 用户显式提供了 `gpu_perf_csv` 或 GPU 的 `pt_file` 路径。

**优先级 3：标准算子任务（Mode A 普通）**
普通 PyTorch 实现文件，无 GPU kernel 相关特征时，走标准 Mode A 流程。

### 3.4 路径推导规则

- `op_name` = 描述文件名去掉 `.py` 后缀；若 manifest 已提供则直接使用。
- `pt_file` 推导：
  - 若 `input_files.pt_file` 显式提供，直接使用。
  - 否则，自动查找描述文件同级目录下的 `{op_name}.pt`。
  - 找不到 → Mode B 报错终止；Mode A 可继续。
- `gpu_perf_csv` 推导：
  - 若 `input_files.gpu_perf_csv` 显式提供，直接使用。
  - 否则，从描述文件所在目录开始**向上级目录递归查找** `gpu_perf.csv`（最多向上 3 级）。
  - 找不到 → 告警并在报告中注明"未找到 GPU 性能基线"。

### 3.5 工作目录准备

本阶段**不创建新的时间戳目录**。`work_dir` 由 Orchestrator 预先创建并通过 manifest 传入。

1. 确认 `work_dir` 存在，不存在则报错终止。
2. 若 `baseline_dir` 非空且指向 `global_baseline/`：
   - 确认 `{baseline_dir}/{op_name}_generated.py` 存在。
   - 复制到当前 work_dir：`{work_dir}/{op_name}_generated.py`。
   - **禁止写回 `global_baseline/` 中的任何文件**。
3. 若 `baseline_dir` 非空且指向其他 round 目录：
   - 确认 `{baseline_dir}/{op_name}_generated.py` 存在。
   - 复制到当前 work_dir：`{work_dir}/{op_name}_generated.py`。
   - 若存在 `{baseline_dir}/{op_name}.py`，同样复制到 `{work_dir}/{op_name}.py`。
   - 若存在 `{baseline_dir}/{op_name}.json`，同样复制到 `{work_dir}/{op_name}.json`。
4. 若 `baseline_dir` 为空：当前 work_dir 为空，后续 Phase 1 负责创建任务文件。

### 3.6 写入 / 更新 summary.json

确保 `summary.json` 至少包含：

```json
{
  "round_index": 2,
  "baseline_dir": "opt-round-1",
  "round_strategy": "focused_tuning",
  "analysis_policy": "bottleneck_analysis",
  "hypothesis": "...",
  "direction": "pattern: custom",
  "evidence_sources": ["benchmark", "model_knowledge"],
  "target_speedup": 2.0
}
```

### 3.7 历史与经验参考

子 agent 在 Phase 0 应读取 `task_manifest.json` 中的：

- `lineage_summary`：从 root 到当前父 round 的优化历史链。在后续阶段设计方向时，应避免与历史失败方向重复，并继承已验证成功的模式。
- `experience_file`：算子级私有经验文件（Markdown）。重点关注 `## Failures / Avoid` 中记录的失败模式与应避免的做法；若本 round 产生新的成功或失败模式，在 Phase 8 追加到该文件。
- `parent_round_journal`：父 round 的 `round_journal.md` 路径（如果 manifest 提供）。子 agent 可在需要深入理解父节点设计/迭代细节时读取，但**禁止修改**。
- `global_baseline_dir`：只读全局基线目录。**仅用于计算加速比**，子 agent 不得在该目录中写入任何内容。

加速比统一以 `global_baseline_dir` 中的实现为基准（`global_baseline_speedup = 1.0`）。`baseline_dir` 中的父 round 代码仅作为本轮优化的起始代码，不作为加速比基准。

---

## 四、Phase 1：任务构建

### 4.1 模式判定与任务文件落盘

#### 情况 A：`{work_dir}/{op_name}.py` 已存在（Round > 1 或已由 baseline 复制）

直接使用，不再调用 `triton-task-extractor`。进入 4.2 freeze_baseline。

#### 情况 B：`{work_dir}/{op_name}.py` 不存在（Round 1）

**Mode A：标准算子任务**

调用 `triton-task-extractor` skill。若为优先级 1 场景，需传入 `expand_shapes=true` 激活单 case → 多 case 自动扩展。

- A.1 单 case 子模式：产出自包含 `{op_name}.py`。
- A.2 多 case 子模式：原样透传 `{op_name}.py` 与 `{op_name}.json`，禁止裁剪 shape。
- 多 case 必须保留全部 shape。

**Mode B：GPU Kernel 输入模式**

不调用 `triton-task-extractor`，由当前子 agent 自建：

1. 读取 `input_files.task_desc`（GPU kernel 源码）和 `.pt` 文件。
2. 构造 `Model` 类（优先使用 `.pt` 中的 `gpu_output`）。
3. 构造 `get_inputs()` / `get_init_inputs()`。
4. 保存 `{work_dir}/{op_name}.py` 并验证。

### 4.2 基线冻结（强制，所有模式通用）

在 `{op_name}.py` 落盘完成后，**立即**调用：

```bash
python3 .claude/skills/triton-op-verifier/scripts/freeze_baseline.py \
    --op_name {op_name} \
    --work_dir {work_dir} \
    --mode {auto|user} \
    [--source_path <源 .py 绝对路径>]
```

`--mode` 取值：
- `user`：用户提供 benchmark（Mode A 标杆文件、Mode A 优先级 1 的 torch 标杆、或由 baseline 复制的任务文件）—— **必传 `--source_path`**。
  - Round 1：传用户原始源文件路径（`input_files.task_desc`）。
  - Round > 1：传 `{baseline_dir}/{op_name}.py`。
- `auto`：Agent 自动生成 benchmark（Mode B 兜底，从 `.pt` 翻译或手写 PyTorch 参考）—— 不传 `--source_path`。

**强制约束**：

1. 冻结后禁止修改 `{work_dir}/{op_name}.py`。
2. 源 benchmark 路径只读：PreToolUse hook 已拦截 `npu_benchmark/`、`ascendc-kernelgen-data*/` 等用户数据目录的 Edit/Write。
3. Mode B 工作目录副本必须字节级等于源：mode=user 时 freeze 会校验 `{work_dir}/{op_name}.py` 的 sha256 == `--source_path` 指向的源文件 sha256。若不相等，在 report.md 标注 `baseline_buggy: true` 后失败退出，**不要试图修复源 benchmark**。
4. freeze 不可重入：锚文件 `{work_dir}/output/.baseline_anchor.json` 一旦写入，重跑 freeze 会 exit 1 拒绝覆盖。
5. 下游 `verify.py` / `benchmark.py` 启动时会校验 `{work_dir}/{op_name}.py` 的 sha256 等于锚文件记录值。
   - 锚文件缺失 → exit 3（Phase 1 未执行 freeze）
   - sha256 不匹配 → exit 4（基线被篡改）
   - 两者都属于 C 类终止，**不退回 Phase 3 重试**。

### 4.3 任务文件验证

所有任务文件必须通过 `validate_task.py` 检查：

```bash
python3 .claude/skills/triton-task-extractor/scripts/validate_task.py \
    --op_name {op_name} --work_dir {work_dir}
```

多 case 模式下需遍历全部 groups 通过。验证通过后进入 Phase 2。

---

## 五、Phase 2：算法设计

本阶段按 `Step 1` → `Step 2` → `Step 3` 顺序执行，**严禁跳过 Step 1 直接进入 Step 2**。

### 5.1 Step 1：前置检查（先执行，产出 precheck.json）

1. 根据 `op_name` / `Model.forward` 推断 `category`。
2. 检查 `.claude/template/{category}.md` 是否存在。
3. 基于算子特征、输入 shape、memory access、历史瓶颈与 Triton/Ascend 最佳实践，由 LLM 自主提出本轮具体优化方向，不再扫描固定 pattern 目录。将分析理由记录到 `precheck.json` 的 `matched_patterns` / `analysis_reasoning` 字段。
4. 产出 `{work_dir}/precheck.json`：

```json
{
  "category": "...",
  "template_path": ".claude/template/{category}.md",
  "template_exists": true,
  "layer1_constraints_loaded": ["..."],
  "loaded_via": "explicit_path",
  "matched_patterns": ["..."],
  "analysis_reasoning": "..."
}
```

**门禁**：`precheck.json` 未产出或 `loaded_via != explicit_path` 时，**禁止进入 Step 2**。

### 5.2 Step 2：调用 designer skill 设计草图

确认 Step 1 门禁通过后，调用 `triton-op-designer` skill。

**传入**：`op_name`、`task_desc`（任务文件完整内容）、`arch`、`user_requirements`（如有）、`gpu_kernel_ref`（优先级 1 场景下用户提供）、`template_path`（显式路径，必传）、`matched_patterns`。

**说明**：`triton-op-designer` 会在设计草图前**自行调用** `triton-knowledge-retrieval`（`query_type=codegen_init`，Step 0–2）获取硬件约束、算子分类和 API 知识单元。子 Agent 无需预检索，也无需通过 `conductor_suggestion` 传递 `kkb_hints`。

**产出**：`{work_dir}/sketch.txt`。

### 5.3 Step 3：Layer 1 合规检查门（强制）

- sketch 产出后，必须读取 `precheck.json` 中已锁定的 Layer 1 约束，逐条核对 `sketch.txt` 是否兼容（`new_category` 时跳过核对）。
- 若发现冲突，**视为 A 类错误**：
  1. 不进入 Phase 3。
  2. 将冲突点作为 `conductor_suggestion` 反馈给 `triton-op-designer`。
  3. 重新执行 Step 1 与 Step 2，直到草图与 Layer 1 兼容。
- 该检查门最多重试 2 次，若仍无法通过，终止当前 round，在 `summary.json` 中标记失败。

仅执行一次，后续 Phase 3 迭代不再重新设计草图。

---

## 六、Phase 3：代码生成与验证（迭代循环）

### 状态变量

```
iteration = 0
max_iterations = 5
history_attempts = []
previous_code = ""
verifier_error = ""
conductor_suggestion = ""
```

> 注：`plugins-official/triton-op-generator/CLAUDE.md` 正文中出现 `max_iterations = 10`，但文末约束表明确写明 "Phase 3 最大迭代 5 次，禁止超出"。本文件以约束表为准，取 `max_iterations = 5`。

### 迭代循环

```
while iteration < max_iterations:
    3.1 代码生成
    3.2 AST 预检查
    3.3 功能验证
    3.4 Conductor 分析与决策
    3.5 性能测试（基线）
```

### 6.1 代码生成

**调用 Skill**：`triton-op-coding`

**输入参数**：
- 首次（`iteration == 0`）：`op_name`, `task_desc`, `arch`, `sketch`, `user_requirements`, `gpu_kernel_ref`（如有）
- 重试（`iteration > 0`）：上述 + `previous_code` + `verifier_error` + `conductor_suggestion`

**说明**：`triton-op-coding` 会在生成代码前**自行调用** `triton-knowledge-retrieval`（`query_type=codegen_init`，Step 0–6 全量）获取完整知识单元；迭代修复（编译错误、精度失败）时自行调用 `compile_error` / `precision_debug` 检索。子 Agent 只需要按原有方式传递 `previous_code`、`verifier_error`、`conductor_suggestion`，不要额外预检索或把检索结果塞进 `conductor_suggestion`。

**产物**：`{work_dir}/output/iter_{iteration}/generated_code.py`

### 6.2 AST 预检查

**执行工具**：`.claude/skills/triton-op-verifier/scripts/validate_triton_impl.py`

```bash
python3 .claude/skills/triton-op-verifier/scripts/validate_triton_impl.py \
    {generated_code_path} [--json]
```

- **退化**（exit code != 0）：设置 `verifier_error = "A-PyTorchFallback-Type{N}: ..."`，跳到 6.4 Conductor。
- **通过**：继续 6.3 功能验证。

### 6.3 功能验证

**调用 Skill**：`triton-op-verifier` (`verify.py`)

```bash
python3 .claude/skills/triton-op-verifier/scripts/verify.py \
    --op_name {op_name} --triton_impl_dir {verify_dir}
```

产物目录：`{work_dir}/output/iter_{iteration}/verify/`

- `{op_name}_torch.py`（来自任务文件）
- `{op_name}_triton_ascend_impl.py`（来自生成代码）
- `verify_result.json`

**多 shape 全量执行**：`verify.py` 为每个 shape 独立 `try/except`，全部跑完后落盘 `verify_result.json`。

**判定来源（强制）**：当前会话必须打开 `verify_result.json`，读取数值字段 `passed_cases` 和 `total_cases` 做相等比较。禁止仅依赖 console 输出文字、退出码或日志片段推断。

**分支**：
- **验证通过**（`passed_cases == total_cases > 0`）：
  - 复制 `iter_{iteration}/generated_code.py` → `{work_dir}/output/generated_code.py`
  - 记录 `phase3_last_iter = iteration`
  - 跳到 **6.5 性能测试**
- **验证失败**：
  - 删除 `{work_dir}/output/generated_code.py`（如存在）
  - 从 `verify_result.json` 读取全部 failures，汇总为 `verifier_error`
  - 跳到 **6.4 Conductor**

### 6.4 Conductor 分析与决策

**错误分类**：
- **A 类**：代码逻辑/算法错误（可修复），含 A-PyTorchFallback-Type1/2/3 子类型。
- **B 类**：环境/基础设施错误（不可修复）。
- **C 类**：重复失败，同一 A 类子类型连续 ≥ 3 次。

**决策**：
- **B 类** → 终止当前 round。
- **C 类** → 终止当前 round。
- **A 类且 `iteration < max_iterations`**：
  - 生成 `conductor_suggestion`
  - `history_attempts.append(本轮记录)`
  - 保存日志到 `iter_{iteration}/log.md`
  - `iteration++`
  - 回到 **6.1 代码生成**

### 6.5 性能测试（基线）

**前置断言（强制）**：进入本步骤前重新读取 `verify_result.json`，再次确认 `passed_cases == total_cases > 0`。任何不符立即返回 6.4。

**L1 兜底**：`benchmark.py` 默认开启 verify 闸门。若当前会话误判越过前置断言，`benchmark.py` 会以 **exit 2** 拒绝运行。处理方式：
- 视为等价于 6.3 verify 失败。
- 重新读 `iter_{iteration}/verify/verify_result.json` 取 failures 汇总成 `verifier_error`。
- 在 `iter_{iteration}/log.md` 标注 "L1 兜底触发：当前会话越过 6.3 闸门"。
- 删除 `{work_dir}/output/generated_code.py`（如存在）。
- 跳到 **6.4 Conductor**。

**调用 Skill**：`triton-op-verifier` (`benchmark.py`)

```bash
python3 .claude/skills/triton-op-verifier/scripts/benchmark.py \
    --op_name {op_name} --verify_dir {verify_dir} \
    --warmup 5 --repeats 50 --output {output_path}
```

**GPU Kernel 模式**附加 `--skip_framework --framework_latency_ms <gpu_reference_ms>`，其中 `gpu_reference_ms` 由 `gpu_perf.csv` 中的 `Duration(us)` 转换而来（除以 1000）。

**产物**：
- `{work_dir}/output/iter_{iteration}/perf_result.json`
- 复制 → `{work_dir}/output/perf_result.json`

**记录 `perf_data`**，然后 `break`。

⚠️ **Phase 3 验证通过后，必须进入 Phase 4 执行性能优化，严禁跳过。**

达到 `max_iterations` 仍未通过 → 当前 round 失败，进入 Phase 5 输出失败报告。

### Conductor 修复建议格式

```
错误分析：
- 类型：{A/B/C}（{子类型描述}）
- 位置：{错误代码位置}
- 具体错误：{错误详情}

修复建议：
1. {具体修改方向}
2. {具体修改方向}

历史提醒：
- 第 N 轮曾因 {问题} 失败，避免重复
```

### PyTorch 退化子类型

| 子类型 | 含义 | 修复建议 |
|--------|------|----------|
| Type1 | 完全无 @triton.jit kernel | 必须创建 @triton.jit kernel，使用 tl.load/tl.store 实现核心计算 |
| Type2 | 有 kernel 定义但 forward() 未调用 | 在 forward() 中通过 kernel[grid](...) 启动 kernel |
| Type3 | forward() 调用了 kernel 但部分计算仍用 PyTorch | 将禁止的 PyTorch 计算移入 kernel |

---

## 七、Phase 4：性能优化与验证（迭代循环）

⚠️ **Phase 4 是必须执行的阶段，禁止跳过。** Phase 3 验证通过后，无论性能数据如何，都必须进入 Phase 4 尝试优化。

### 7.0 Phase 4 入口硬断言（强制）

在执行 7.1 之前，必须打开 `{work_dir}/output/iter_{phase3_last_iter}/verify/verify_result.json` 读取数值字段，确认 `passed_cases == total_cases > 0`。

- 断言通过 → 正常进入 7.1
- 断言失败 → **C 类终止当前 round**。写 `summary.json`：

```json
{
  "success": false,
  "gen_iterations": <...>,
  "failure_phase": "phase3_gate_violation",
  "failure_reason": "Phase 3 verify_result.json passed_cases(<x>) < total_cases(<y>)，但流程已进入 Phase 4",
  "last_error": "<failures 列表摘要>"
}
```

### 状态变量

```
opt_iteration = 0
max_opt_iterations = <读取 .claude/skills/triton-latency-optimizer/SKILL.md，统计 "### 优化点" 出现次数 + 1；失败则默认 20>
target_speedup = <manifest.config.target_speedup>
best_code = ""
best_speedup = 0.0
baseline_code = Phase 3 产出的 generated_code.py
phase3_last_iter = Phase 3 最后一次验证通过的 iter 编号
improvement_made = false
target_reached = false

ir_iteration = 0
ir_max_iterations = 20
last_optimization_point = None
ir_has_more_suggestions = true
current_iter_dir = ""  # opt_iter_{n} 或 opt_iter_{n}_ir_{k}

# simulator 采集驱动相关变量
# latency-optimizer 优化耗尽（普通点 1-29 + IR 点 30 均耗尽）且仍未达 target 时，转入 simulator 采集优化。
simulator_attempted = false                         # triton-simulator-optimizer 是否已被调用（7.6 退出前置门检查项）
```

### 迭代循环

```
while opt_iteration < max_opt_iterations:
    7.1 代码分析 + 优化策略 + 代码重写
    7.2 精度验证（基线复用 + 优化侧单次执行）
    7.3 性能测试（基线复用 + 优化侧单次执行）
    7.4 结果判定
    7.5 分析决策（验证失败时）
    7.6 终局判定
```

### 7.1 代码分析 + 优化策略 + 代码重写

**调用 Skill**：`triton-latency-optimizer`

**输入参数**：`baseline_code`（或上一轮优化后的代码）、`opt_iteration`、`task_desc`、`arch`、`user_requirements`。

**说明**：`triton-latency-optimizer` 会在每轮优化前**自行调用** `triton-knowledge-retrieval`（`query_type=perf_tuning`）获取瓶颈→优化点映射和 Top-3 调优经验，并结合自身优化点 1–30 的命中检查优先尝试 KKB 指向的高收益优化点。

**产物**：`{work_dir}/output/{current_iter_dir}/optimized_code.py`

**分支**：

> latency-optimizer 的返回信息中**必须包含字段 `ir_has_more_suggestions: bool`**（IR 分析器是否还能给出新优化建议，仅当本轮命中点为 30 时有意义，其他轮次置 `false`）。Phase 4 据此判断是否继续 IR 多轮迭代。

- **存在普通优化点（1-29）命中** → 走原流程重写代码，本轮产物目录 `current_iter_dir = opt_iter_{opt_iteration}`，`last_optimization_point = <命中点编号>`。
- **triton-latency-optimizer 报告无更多普通优化点**：
  - 若以下任一条件满足，**不终止**，要求 latency-optimizer 继续尝试对应优化点（这些仍属普通轮）：
    - `total_cases > 1` 且 `speedup_vs_torch < 0.5`：强制尝试 kernel 分裂（优化点 18）
    - `speedup_vs_torch < target_speedup` 且 `opt_iteration < 3`：要求重新扫描，重点检查当前算子类别对应的高频命中点（见 `triton-latency-optimizer/SKILL.md` 的“算子类别与高频优化点”表）
  - **IR 多轮迭代分支**（普通优化点耗尽时）：
    - 若 `ir_has_more_suggestions == true` 且 `ir_iteration < ir_max_iterations`：
      - **不终止**，强制走 IR 子流程（优化点 30），重新提取 `last_pass.mlir` 并分析。
      - `ir_iteration++`，`last_optimization_point = 30`。
      - 本轮产物目录 `current_iter_dir = opt_iter_{opt_iteration}_ir_{ir_iteration}`（避免与普通轮目录冲突）。
    - 否则（`ir_has_more_suggestions == false` 或 `ir_iteration >= ir_max_iterations`，即 latency-optimizer 优化已耗尽）：
      - 若 `optimized_speedup >= target_speedup`（`target_reached`）→ 可进入 **7.6 终局判定**。
      - 若 `optimized_speedup < target_speedup` → **强制转入下方 simulator 采集驱动分支**，禁止直接 7.6。

- **simulator 采集驱动分支**（latency-optimizer 优化耗尽且 `optimized_speedup < target_speedup` 时触发）：
  - 调用 `triton-simulator-optimizer` skill（独立 skill，**只采集 + 诊断**：msprof 采集 → 解析 pipe 占比 → 产出诊断报告）。skill 不自带优化技术、不产出代码——优化技术 owner 唯一是 `triton-latency-optimizer`。
  - ⚠️ 在拿到 simulator 采集证据（`MMAD` 占比 > 50%）前，**严禁**下“dot 是硬件瓶颈/不可优化”结论。
  - 诊断报告内容：瓶颈类型 + 热源码行 + **修复方向 = `triton-latency-optimizer` 优化点编号**（Cube 空等→19/21、标量降级→6/5/17、访存→7/21/10、barrier→19）；`MMAD` > 50% 时报告“无对应优化点（真·硬件极限）”。
  - `simulator_attempted = true`（进入本分支即置位，无论诊断是否给出修复方向）。
  - **修复落地（交 latency-optimizer，不在 simulator-optimizer 内改代码）**：编排器带诊断报告回到 7.1 调 `triton-latency-optimizer`，将诊断指向的优化点作为**强制命中点**传入（覆盖其静态命中判断），latency-optimizer 据此加载对应优化点参考文档产出代码。
  - latency-optimizer 产出的代码走 7.2/7.3 验证；本轮产物目录 `current_iter_dir = opt_iter_{opt_iteration}`（`opt_iteration` 正常自增）。有提升则 `opt_iteration++` 回 7.1；无提升则由 simulator-optimizer 重采确认瓶颈是否转移。
  - 诊断报告“无对应优化点”（`MMAD` 实测 > 50% 且增大 tile / bf16 化均不可行）→ 进入 **7.6 终局判定**。

**Checklist 检查（强制）**：
- 读取 `.claude/skills/triton-latency-optimizer/references/checklist.md`，验证 `optimized_code.py` 是否满足所有规范。
- 不满足 → 修改代码直至满足，然后重新检查。
- 满足 → 复制 `optimized_code.py` → `{work_dir}/output/optimized_code.py`，进入 **7.2**。

### 7.2 精度验证（基线复用 + 优化侧单次执行）

**调用 Skill**：`triton-op-verifier` (`verify.py`)

产物目录：`{work_dir}/output/{current_iter_dir}/verify/`

- `{op_name}_torch.py`（PyTorch 参考）
- `{op_name}_triton_baseline.py`（Phase 3 基线）
- `{op_name}_triton_optimized.py`（优化后）

**基线侧**：直接复制 Phase 3 的校验结果，不再重跑：

```bash
cp {work_dir}/output/iter_{phase3_last_iter}/verify/verify_result.json \
   {work_dir}/output/{current_iter_dir}/verify/verify_result_baseline.json
```

**优化侧**：运行 `verify.py --triton_impl_name triton_optimized`，产物 `verify_result_optimized.json`。

**判定**：
- optimized 全过 → 进入 **7.3 性能测试**
- 未全过 → 跳到 **7.5（A 类）**

### 7.3 性能测试（基线复用 + 优化侧单次执行）

**前置断言（强制）**：重新读取 `verify_result_optimized.json`，确认 `passed_cases == total_cases > 0`。任何不符立即跳到 **7.5（A 类）**。

**L1 兜底**：`benchmark.py` 以 exit 2 拒绝时，处理方式同 Phase 3：等价 7.2 optimized 失败，读 failures，写 log.md，跳到 **7.5（A 类）**。

**基线侧**：直接复制 Phase 3 性能结果：

```bash
cp {work_dir}/output/iter_{phase3_last_iter}/perf_result.json \
   {work_dir}/output/{current_iter_dir}/baseline_perf_result.json
```

**优化侧**：运行 `benchmark.py --triton_impl_name triton_optimized [--skip_framework ...]`，产物 `optimized_perf_result.json`。

**几何平均加速比判定**：
- `baseline_speedup` 取自 `baseline_perf_result.json` 的 `speedup_vs_torch`。
- `optimized_speedup` 取自 `optimized_perf_result.json` 的 `speedup_vs_torch`。
- 直接对比两个几何平均。

### 7.4 结果判定

**前置检查**：
- 若 `optimized_perf_result.json` 不存在或读取失败，跳过本步骤进入 7.5（A 类）。
- 若 `baseline_speedup` 或 `optimized_speedup` 任一为 `null`，直接判优化失败，跳到 7.5（A 类）。

`optimized_speedup > baseline_speedup`：
- 优化成功，更新 `best_code` / `best_speedup`，`improvement_made = true`。
- 普通轮：`opt_iteration++`；IR 轮：`ir_iteration` 已在 7.1 自增。
- `continue`

否则（含相等）：
- 视为无提升；普通轮 `opt_iteration++`。
- IR 轮不因 `improvement_made == false` 而 break，仍 `continue`，让下一轮 7.1 重新评估 IR。

### 7.5 分析决策（验证失败时）

- **A 类**（优化引入逻辑错误）→ 回退，调整策略，`opt_iteration++`（IR 轮 `ir_iteration` 已在 7.1 自增），`continue`。
- **B 类**（环境错误）→ 终止优化，以 Phase 3 结果继续。
- **C 类**（无法继续）→ 终止优化，以 Phase 3 结果继续。

### 7.6 终局判定

⚠️ **退出前置门（强制，不满足禁止 break）**：7.6 仅在以下任一条件满足时可进入：
- (a) `opt_iteration >= max_opt_iterations`（全局兜底）；**或**
- (b) latency-optimizer 优化耗尽（普通点 1-29 + IR 点 30 均耗尽）**且** 满足以下之一：
  - `target_reached == true`（`optimized_speedup >= target_speedup`）；**或**
  - `simulator_attempted == true` 且 `triton-simulator-optimizer` 已确认无更多 simulator 采集驱动改进（`MMAD` 实测 > 50% 且增大 tile / bf16 化均不可行）。

**任何其他情况——尤其 `optimized_speedup < target_speedup` 且 `simulator_attempted == false`——禁止进 7.6**：必须回到 **7.1**；若 latency-optimizer 已耗尽，强制转入 7.1 的 **simulator 采集驱动分支**（不得直接 7.6）。`improvement_made == true` 不构成退出条件。

通过退出前置门后，按 `improvement_made` 选择最终代码：
- `improvement_made == true` → 优化成功，break，进入 Phase 5（最终代码 = optimized_code.py）。
- `improvement_made == false` → 优化失败（做完所有尝试 + simulator 采集后没有效果），break，进入 Phase 5（最终代码 = Phase 3 基线）。

---

## 八、Phase 5：输出报告

### 8.1 选择最终代码

- Phase 4 成功（`improvement_made == true`）→ `optimized_code.py`（即 `best_code`）。
- Phase 4 失败 → Phase 3 的 `generated_code.py`。

复制最终代码到 `{work_dir}/{op_name}_generated.py`。

### 8.2 写入 `report.md`

包含：
- 基本信息：arch、工作目录、round_index、direction、hypothesis
- 生成结果：Phase 3 迭代次数、Phase 4 迭代次数、最终版本来源
- **目标加速比**：`target_speedup`，是否达到（`target_reached`）
- **实际最佳加速比**：`best_speedup`（保留 4 位小数）
- **Shape 通过率（以 verify 为准）**：从 `output/iter_{phase3_last_iter}/verify/verify_result.json` 读取 `passed_cases / total_cases`
- **GPU 参考性能**（Mode B 且找到 `gpu_perf_csv` 时）
- 性能数据：延时加权加速比、总延时、平均延迟
- 性能明细：以 verify 结果为准列出 status；通过 shape 从 perf_result 取 framework/implementation/speedup
- 代码路径：`{op_name}_generated.py`

### 8.2.5 写入 `round_journal.md`（新增）

除 `report.md` 外，子 agent 必须在 `{work_dir}/round_journal.md` 中记录本轮的详细设计/迭代过程，供后续 round 作为父节点上下文读取。文件结构建议如下：

```markdown
# Round {N} Journal

## Design Rationale
- Target direction: {concrete direction}
- Why chosen: {based on lineage/experience/bottleneck}
- Algorithm sketch: {high-level tiling/fusion/layout plan}

## Iteration Log

### Iter 1 (coding)
- Attempt: {what was tried}
- Result: {compile error / verify failure / speedup}
- Fix: {how it was fixed}

### Opt-Iter 1 (latency optimization)
- Attempt: {what was tuned}
- Result: {speedup / regression / compile issue}
- Decision: {keep / revert / note as Pareto option}

## Final Outcome
- best_speedup: {x.xxx}
- effective_metric_source: {kernel|total-op|mixed}
- outcome: {kept / abandoned / fallback}
```

内容要求：
- 必须覆盖 Phase 2 的设计思路、Phase 3 的 iter 关键尝试与修复、Phase 4 的 opt_iter 关键尝试与决策。
- 长度控制在 300–800 个 token 以内，避免下轮子 agent 上下文爆炸。
- 禁止写入敏感路径或超出本 round 的跨轮判定。

### 8.3 写入 `summary.json`

**字段取值口径（强制）**：
- `perf_data.passed_cases` / `failed_cases` / `total_cases` 必须从 `output/iter_{phase3_last_iter}/verify/verify_result.json` 读取（精度通过数）。
- 延时类字段从 perf_result.json 读取（Phase 4 成功时优先 `optimized_perf_result.json`）。
- 异常索引字段从 perf_result.json 同名字段透传。
- `per_shape_results[].status` 以 verify 为准。
- **禁止**直接把 `perf_result.json` 顶层 `passed_cases` 复制到 summary。

成功时标准格式：

```json
{
  "success": true,
  "gen_iterations": 2,
  "opt_iterations": 1,
  "optimized": true,
  "target_speedup": 2.0,
  "target_reached": true,
  "best_speedup": 2.15,
  "perf_method": "profiler",
  "skill_path": ".claude/skills/triton-op-verifier",
  "perf_data": {
    "avg_latency_ms": 0.5678,
    "speedup_vs_torch": 2.1746,
    "speedup_vs_baseline": 1.35,
    "total_cases": 5,
    "passed_cases": 5,
    "failed_cases": 0,
    "nan_indices": [],
    "inf_indices": [],
    "zero_indices": [],
    "negative_indices": [],
    "none_indices": [],
    "per_shape_results": [
      {
        "case_idx": 1,
        "status": "pass",
        "shape_desc": "...",
        "speedup_vs_torch": 1.82
      }
    ]
  }
}
```

GPU Kernel 模式扩展字段：
- `gpu_mode: true`
- `perf_data.gpu_reference_ms`
- `perf_data.ascend_vs_gpu_ratio`
- `per_shape_results[].gpu_reference_ms`
- `per_shape_results[].ascend_vs_gpu_ratio`

Phase 3 失败时：

```json
{
  "success": false,
  "gen_iterations": 5,
  "failure_phase": "generation",
  "failure_reason": "达到最大迭代次数",
  "last_error": "..."
}
```

Phase 4 入口断言失败：

```json
{
  "success": false,
  "gen_iterations": 3,
  "failure_phase": "phase3_gate_violation",
  "failure_reason": "Phase 3 verify_result.json passed_cases(45) < total_cases(50)，但流程已进入 Phase 4",
  "last_error": "<failures 列表摘要>"
}
```

---

## 九、Phase 6：知识库摄入与维护

**必须在 Phase 5 完成后、Phase 7 会话导出之前执行**。

子 Agent 依据本次运行中的事件日志，自动从本次运行中提取知识写回知识库。各子 Skill 在执行过程中产生 F1–F5 事件记录在各自上下文中，子 Agent 在此阶段统一收集并完成摄入。

### 9.1 触发条件检查

逐条评估三种来源的触发条件：

| Source | 触发条件 | 写入目标 |
|--------|---------|---------|
| **Source B**（成功经验） | 同时满足三项：(1) Phase 3 精度全过（`passed_cases == total_cases`）(2) `summary.json.perf_data.speedup_vs_torch ≥ 1.1`(3) Phase 4 有有效 profiling 改进 | `.claude/skills/triton-knowledge-retrieval/knowledge/raw/experiences/<op_category>/` |
| **Source C**（调试经验） | 存在完整事件对：F1(fail) → F1(success) 或 F2(fail) → F2(pass)（同一 session_id） | `.claude/skills/triton-knowledge-retrieval/knowledge/raw/experiences/general_debug/` 或 `general_stability/` |
| **Source D**（性能洞察） | F3.bottleneck_actual 与 bottleneck_predicted 类型不同，**或**同类型但 high_impact_units 的贡献度偏差 > 20% | `.claude/skills/triton-knowledge-retrieval/knowledge/raw/experiences/general_perf/` 或 `<op_category>/` |

若**三种全部不满足** → 输出 `[Phase 6] 无需摄入，跳过`。

### 9.2 摄入步骤（任一触发即执行）

详细步骤见 `.claude/skills/triton-knowledge-retrieval/builder/auto_ingestion.md`。Agent 必须按目录约定生成 YAML 文件：

```
exp_<cat>_<seq>_<YYYYMMDD>.yaml   # Source B
exp_debug_<seq>_<YYYYMMDD>.yaml   # Source C
exp_perf_<seq>_<YYYYMMDD>.yaml    # Source D
```

写入步骤：
1. 新建 YAML 文件到目标目录
2. 在 `raw/experiences/_index.yaml` 追加一行
3. append `INGESTION` 事件到 `.claude/skills/triton-knowledge-retrieval/log.md`

### 9.3 定期维护检查

统计 `.claude/skills/triton-knowledge-retrieval/log.md` 中 `event_type: F3` 总数 `N`：

- 若 `N > 0` 且 `N mod 10 == 0` → 触发正常维护：Task 1（质量重评）+ Task 2（冲突检测）+ Task 3（缓存失效）+ Task 4（覆盖统计），完成后 append `MAINTENANCE_REPORT` 事件到 log.md
- 紧急触发（任一满足立即执行 Task 1 + Task 2）：
  - `_index.yaml` 中 deprecated 条目 > 20%
  - review 条目 > 30
  - 同一 experience_id 在 ≥ 3 条 F5 事件中

详细步骤见 `.claude/skills/triton-knowledge-retrieval/maintenance/periodic_tasks.md`。

### 9.4 终端输出（强制盒装格式）

```
┌─ KKB 知识摄入 [Phase 6] ─────────────────────────────
│ Source B (生成经验): <已写入 raw/experiences/<cat>/<file>.yaml / 未触发 (<原因>)>
│ Source C (调试记录): <已写入 raw/experiences/general_debug|stability/<file>.yaml / 未触发 (<原因>)>
│ Source D (Profiling): <已写入 raw/experiences/general_perf|<cat>/<file>.yaml / 未触发 (<原因>)>
│ 定期维护: <已执行 Task 1-4 / 未触发 (F3 总数=<N>，N mod 10 = <值>)>
└────────────────────────────────────────────────────
```

任何情况下都必须输出此盒装摘要，即使全部不满足也要输出“未触发”的结论。

---

## 十、Phase 7：会话导出

**必须在 Phase 6 完成后执行**，将当前 Claude Code 会话归档到工作目录。放在最后是为了最大化 jsonl 完整性——仍会缺失本步骤之后的极少量消息，可接受。

```bash
MY_JSONL=$(grep -l "{work_dir}" /root/.claude/projects/*/*.jsonl 2>/dev/null | head -1)
if [ -n "$MY_JSONL" ]; then
  cp "$MY_JSONL" {work_dir}/session.jsonl
  python3 ./utils/render_session.py \
    {work_dir}/session.jsonl {work_dir}/session.md 2>&1 || \
    echo "WARN: session render failed (non-fatal)"
else
  echo "WARN: session jsonl not located (non-fatal)"
fi
```

- 用工作目录绝对路径作为唯一标记定位自己的 session jsonl，禁止用时间排序。
- 渲染失败 / 定位失败均不阻塞任务，仅告警。
- 若 `utils/render_session.py` 不存在，仅复制 `session.jsonl`。

---

## 十一、Phase 8：Round Contract 校验 + round_result.json

> 注意：官方单 agent 版本的 Phase 7 是“会话导出”，Phase 6 是“知识库摄入与维护”；在 multi-agent 架构中，知识库摄入由子 Agent 在每轮 Phase 6 完成，经验沉淀（`.claude/template/{category}.md` 更新）由 Orchestrator 在所有 round 结束后统一完成。子 agent 的 Phase 8 只负责单轮产物校验与结果上报。

### 11.1 执行 submit_round.py

```bash
python3 skills/triton-agent-loop/scripts/submit_round.py \
    --round-dir {work_dir} \
    --current-round {round_index} \
    --final-round {max_rounds} \
    --op-dir triton_ascend_output/{op_name}-{algorithm}-{run_tag}
```

- `status == "fail"` → 按 issues 修复后重跑；若无法修复，将当前 round 标记为失败。
- `status == "pass"` → 继续。

### 11.2 写入 round_result.json

`submit_round.py` 通过后，写入 `{work_dir}/round_result.json`：

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
  "direction_yield": "medium",
  "yield_reason": "improvement_made",
  "absolute_gain_vs_baseline": 0.15,
  "absolute_gain_vs_global_baseline": 0.35,
  "baseline_dir": "opt-round-1",
  "global_baseline_dir": "global_baseline",
  "global_baseline_speedup": 1.0,
  "design_summary": "Program multiple rows per block to increase instruction-level parallelism.",
  "iter_count": 2,
  "opt_iter_count": 1,
  "key_fixes": ["added boundary mask for tail rows", "switched accumulator to float32"],
  "bottlenecks_observed": ["small-M shapes under-utilize warps"],
  "artifacts": {
    "summary_json": "triton_ascend_output/layer_norm/opt-round-2/summary.json",
    "report_md": "triton_ascend_output/layer_norm/opt-round-2/report.md",
    "final_code": "triton_ascend_output/layer_norm/opt-round-2/layer_norm_generated.py"
  }
}
```

子 agent 应尽可能在 `round_result.json` 中写入以下新增字段，供 Orchestrator 构建更丰富的 `lineage_summary`：

| 字段 | 说明 |
|---|---|
| `design_summary` | 一句话设计思路/算法图摘要 |
| `iter_count` | Phase 3 coding/verify 迭代次数 |
| `opt_iter_count` | Phase 4 latency 优化迭代次数 |
| `key_fixes` | 本 round 关键修复列表 |
| `bottlenecks_observed` | 观察到的性能/编译瓶颈 |

此外仍可写入 `direction_yield`、`yield_reason`、`absolute_gain_vs_baseline`、`absolute_gain_vs_global_baseline`、`baseline_dir`、`global_baseline_dir`、`global_baseline_speedup`；若未写入，主 agent 会根据 `summary.json` 与 `global_baseline/baseline_info.json` 自动计算。

### 11.3 追加算子级经验

若 `task_manifest.json` 提供了 `experience_file`，子 agent 在写入 `round_result.json` 前，应在本轮 `report.md` 与 `round_journal.md` 总结的基础上，向该文件追加条目：

- 若本轮最终失败或出现应避免的反模式，在 `## Failures / Avoid` 下追加失败原因、触发条件与建议规避措施。
- 若本轮验证成功的优化模式具有可复用性，在 `## Successful Patterns` 下追加模式说明与适用条件。
- （推荐）按 round 添加结构化小节，便于后续子 agent 快速检索：

  ```markdown
  ## Round 2 — program-multiple-rows
  - direction: program-multiple-rows
  - best_speedup: 1.35x
  - success: program multiple rows per block improves ILP for M>=512
  - avoid: tail rows need explicit mask; vectorized store breaks small-M
  - reusable: boundary-mask pattern for tail rows
  ```

**禁止修改项目级 `.claude/template/{category}.md`**（该文件由 `init.sh` 按项目复制，仅影响当前项目）。

### 11.4 退出

子 agent 写入 `round_result.json` 并（可选）追加 `experience_file` 后立即退出，不再执行任何后续操作。

---

## 十二、错误处理

| 阶段 | 错误 | 处理 |
|------|------|------|
| Phase 1 (模式 A) | 任务文件验证失败 | 修复重试（最多 2 次）；多 case 禁止降级为单 case |
| Phase 1 (模式 B) | `.pt` 文件不存在 | 报错终止，当前 round 失败 |
| Phase 1 (模式 B) | `Model` 翻译验证失败 | 修复重试（最多 2 次） |
| Phase 1 | freeze_baseline.py exit 1 | 锚文件已存在，审计原因 |
| Phase 1 | freeze_baseline.py exit 2 | `{op_name}.py` 不存在，回退修复 |
| Phase 1 | freeze_baseline.py exit 3 | mode=user 但未传 source_path，补全参数 |
| Phase 1 | freeze_baseline.py exit 5 | 工作目录副本 sha256 ≠ 源 sha256，标注 `baseline_buggy: true` 后失败退出 |
| Phase 3/4 | verify/benchmark exit 3 | C 类终止当前 round，`failure_phase: "phase1_freeze_missing"` |
| Phase 3/4 | verify/benchmark exit 4 | C 类终止当前 round，`failure_phase: "baseline_tampered"` |
| Phase 3 | 达到 max_iterations | 当前 round 失败 |
| Phase 3 | B 类环境错误 | 当前 round 失败 |
| Phase 3 | C 类重复错误 | 当前 round 失败 |
| Phase 3/4 | benchmark.py 超时/被 kill | 严禁编造数据；降低 `--repeats` 重试（50→20→10→5），全部超时则标记 B 类错误 |
| Phase 4 | 无更多优化点 + 无效果 | 以 Phase 3 结果继续 |
| Phase 4 | B 类环境错误 | 终止优化，以 Phase 3 结果继续 |

### L1 闸门失败映射

| 触发位置 | 信号 | 等价处理 | 备注 |
|----------|------|----------|------|
| Phase 3.5 benchmark exit 2 | stderr 含 `[L1 闸门]` | 等价 6.3 verify 失败 → 读 failures → 6.4 Conductor → iteration++ | log.md 标注 "L1 兜底触发：当前会话越过 6.3 闸门" |
| Phase 4.3 optimized benchmark exit 2 | 同上 | 等价 7.2 optimized 失败 → 读 failures → 7.5 A 类 → opt_iteration++ | log.md 标注 "L1 兜底触发：当前会话越过 7.3 断言" |
| Phase 4 入口断言失败 | 自检 verify passed < total | C 类终止，`failure_phase: "phase3_gate_violation"` | 不允许退回 Phase 3 |

---

## 十三、约束

| 约束 | 说明 |
|------|------|
| 安装前置条件 | 首次使用前必须跑 `bash init.sh project claude` 注册 PreToolUse hook |
| 源 benchmark 只读 | 用户数据目录由 hook 强制拦截 Edit/Write |
| 工作目录基线冻结 | Phase 1 末尾必须调 `freeze_baseline.py` 落锚 |
| GPU Kernel 模式 | `.pt` 必须与 `.py` 同名同目录；`gpu_perf.csv` 向上查找最多 3 级 |
| Phase 3 最大迭代 | 5 次，禁止超出 |
| Phase 4 迭代策略 | `max_opt_iterations = triton-latency-optimizer` 优化点个数 + 1；普通优化点 1–29 + IR 优化点 30；IR 分支上限 20 次；耗尽且未达 `target_speedup` 时必须先走 `triton-simulator-optimizer` 采集驱动分支 |
| Phase 4 成功底线 | 性能不劣化（`optimized_speedup > baseline_speedup`） |
| Phase 4 退出判定 | 仅当 `opt_iteration >= max_opt_iterations`，或优化耗尽且（`target_reached` 或 `simulator_attempted` 确认无改进）时才可退出 |
| Phase 4 基线复用 | 7.2/7.3 基线侧 verify/perf 文件必须从 Phase 3 `iter_{phase3_last_iter}` 复制，禁止对基线代码重跑 |
| A 类连续上限 | 同一子类型连续 ≥ 3 次 → 自动终止当前 round |
| 禁止 PyTorch 退化 | forward() 中禁止 torch._ / F._ 计算操作 |
| 文件操作范围 | 限制在 `work_dir` 内；只读 `baseline_dir` 与 `global_baseline_dir`；唯一例外是 `.claude/skills/triton-knowledge-retrieval/`（`knowledge/cache/`、`log.md`、`knowledge/raw/experiences/`），其副作用按集成规范跨任务累积 |
| 验证方式 | 必须调用 triton-op-verifier skill 的脚本，禁止自创测试 |
| 性能数据真实性 | 严禁编造、估算、模拟 benchmark 数据；所有数值必须从 `perf_result.json` 读取 |
| 性能基准 | 加速比统一以 `global_baseline_dir` 中的用户参考实现为基准 |
| Benchmark 超时降级 | benchmark.py 超时或被 kill 时，必须自动降低 `--repeats` 值重试（50→20→10→5） |
| 知识库检索归属 | 检索（读）由 `triton-op-designer` / `triton-op-coding` / `triton-latency-optimizer` 自主触发；摄入（写）由子 Agent Phase 6 统一完成 |
| 知识库反馈事件 | F1–F5 事件在子 Skill 执行过程中产生，子 Agent Phase 6 统一收集写入 `.claude/skills/triton-knowledge-retrieval/log.md` |
| 知识库路径 | 固定使用 `work_dir/.claude/skills/triton-knowledge-retrieval/`（实际指向跨任务共享知识库），其下 `cache/`、`log.md`、`raw/experiences/` 副作用跨任务累积 |
| 只做当前 round | 不读取历史 `round_index.json`，不做跨轮判定 |
| 禁止更新 template | 经验沉淀由 Orchestrator 完成；子 agent 仅可追加 `experience_file` |
| 禁止修改 `state-{op_name}-{algorithm}-{run_tag}.json` | 状态文件由 Orchestrator 独占写 |
| 禁止修改 `global_baseline/` | 全局基线目录中的文件仅用于加速比计算 |
| 禁止调用 transition_next_round.py | 子 agent 只产出当前 round 结果 |
| **执行给定 direction** | naive 子 agent 必须执行 manifest 中的 `direction` 和 `hypothesis`，不得自主更改 |

---

## 十四、沟通风格

- 专业、技术、简洁。
- 每完成一个 Phase 输出一行状态更新（如 `Phase 3 ✓ verify passed`）。
- 不向用户提问、不征求意见、不暂停等待。
- Phase 8 完成后输出一行退出摘要：
  `Round {N} completed: algorithm=naive, status={status}, direction={direction}, best_speedup={speedup}`
- 错误时清晰描述 + 建议操作。
