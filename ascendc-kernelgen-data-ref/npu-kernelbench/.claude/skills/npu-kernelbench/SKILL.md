---
name: npu-kernelbench
description: >
  Run the npu-kernelbench framework to evaluate custom Ascend NPU kernels for correctness and performance.
  Wrap user source code into solution.json, launch single or batch evaluation, configure anti-cheat switches,
  generate HTML/JSONL reports, and interpret traces. Use this skill when the user wants to benchmark a kernel,
  run a dataset level, evaluate an operator, pack a solution, or tune anti-hack settings.
---

# NPU KernelBench 评测与封装 Skill

面向 Ascend NPU 的 kernel 评测框架，支持对自定义算子（PyTorch/torch_npu、Ascend C/CANN、Triton、TileLang、Catlass）进行**精度校验**和**性能评测**，并与参考实现对比。

本 Skill 覆盖三件事：

1. **把用户源码封装成 `solution.json`**（各种语言）并放到同名算子的文件路径下。
2. **启动评测**：单个算子、批量数据集、报告生成。
3. **使用防作弊开关**：启用/禁用 anti-hack 检查。

## 触发场景

- 用户给了源码，需要包成 `solution.json`。
- 用户想跑单个算子评测或批量数据集。
- 用户需要生成 HTML/JSONL 报告。
- 用户需要配置 anti-hack 开关。
- 用户需要解读 `traces.jsonl` 结果。

## 前置条件

- Python >= 3.10
- 已安装 `torch`（CPU 版）与 `torch_npu`，且版本一致
- 已 source CANN 环境（`msprof` 等工具可用）
- 项目内使用 `uv` 管理依赖：

```bash
cd ascendc-kernelgen-data/npu-kernelbench   # 以实际路径为准
uv sync
```

## 把源码封装成 solution.json

### 通用约定（所有语言通用）

| 项目 | 约定 |
|------|------|
| Python 入口文件名 | **必须**是 `main.py` |
| 入口函数名 | **必须**是 `def run(...)` |
| `entry_point` | **必须**是 `main.py::run` |
| 多文件解 | `kernel.py` + `main.py`，`main.py` 里用 `import kernel` 或 `from kernel import ...`，禁止相对导入 `from . import ...` |
| `sources` 里禁止 | `CMakeLists.txt`、`Makefile`、`setup.py`、`pyproject.toml`、`__init__.py`、任何构建脚本 |
| 源文件路径 | 相对路径，禁止包含 `..` |

### 按语言封装

| 语言 | `spec.languages` | 常见源文件 | 特殊字段 |
|------|------------------|-----------|----------|
| PyTorch / 纯 Python | `["python"]` 或 `["pytorch"]` | `main.py` | — |
| Triton | `["triton"]` | `main.py` 或 `kernel.py` + `main.py` | 建议填 `declared_kernel` 辅助 anti-hack |
| TileLang | `["tilelang"]` | `main.py` | — |
| AscendC (bisheng) | `["ascendc"]` | `op_kernel/*.cpp` + `op_host/*.cpp` + `main.py` | `build_method: "bisheng"`、`bisheng_cflags`、`ld_flags` |
| AscendC (cmake) | `["ascendc"]` | 同上 | `build_method: "cmake"` |
| Catlass | `["catlass"]` | C++ 模板 + `main.py` | `build_method: "bisheng"` |

### destination_passing_style

- `true`（默认）：框架会预分配输出张量，通过 `run(..., out)` 传给你的 kernel，`run` 把结果写入 `out`，**不需要 return**。
- `false`：`run(...)` 直接返回新的输出张量。

### 推荐封装流程

1. 确认入口文件名为 `main.py`，入口函数为 `def run(...)`。
2. 确认语言、目标硬件（默认 `ascend910b`）、build method（AscendC）。
3. 运行 skill 脚本：

```bash
uv run python .claude/skills/npu-kernelbench/scripts/pack_solution.py \
    --name my_matmul_triton \
    --definition matmul_triton \
    --source main.py \
    --source kernel.py \
    --language triton \
    --hardware ascend910b \
    --output my_solutions/level2/MatmulTriton/solution.json
```

4. 验证：

```bash
uv run python .claude/skills/npu-kernelbench/scripts/validate_solution.py \
    my_solutions/level2/MatmulTriton/solution.json
```

完整 schema 与各种语言示例见 `references/solution-format.md`。

## 启动评测

### 单个算子评测（`eval`）

```bash
uv run npu-kernelbench eval tests/samples/gelu \
    --solution tests/samples/gelu/solution.json \
    -o /tmp/gelu_traces.jsonl \
    --html /tmp/gelu_report.html
```

等价显式路径写法：

```bash
uv run npu-kernelbench eval \
    --definition tests/samples/gelu/definition.json \
    --workload tests/samples/gelu/workload.jsonl \
    --solution tests/samples/gelu/solution.json
```

### 批量数据集评测（`run-dataset`）

```bash
# 跑 ascendc_migration 的 level1
uv run npu-kernelbench run-dataset data/ascendc_migration \
    --scenario ascendc_migration \
    --level level1 \
    --solution-dir ./solutions \
    -o /tmp/ascendc_l1.jsonl \
    --html /tmp/ascendc_l1.html

# 跑多个 level，多 NPU
uv run npu-kernelbench run-dataset data/kernel_generator \
    --scenario kernel_generator \
    --level level1 --level level2 \
    --npu-ids 0,1,2,3 \
    --solution-dir ./solutions \
    -o /tmp/kg_l1l2.jsonl \
    --html /tmp/kg_l1l2.html
```

### 已有 traces 生成 HTML

```bash
uv run npu-kernelbench report /tmp/gelu_traces.jsonl --html /tmp/gelu_report.html
```

### 快捷脚本

使用本 skill 提供的 `quick_eval.sh`：

```bash
.claude/skills/npu-kernelbench/scripts/quick_eval.sh \
    tests/samples/gelu tests/samples/gelu/solution.json /tmp/gelu.jsonl
```

完整 CLI 选项表、resume、profiling、runner 选择见 `references/cli-reference.md`。

## 防作弊开关

默认开启两项：

- `ast_scan` — 静态 AST 扫描，禁止危险导入/调用。
- `kernel_signature` — 运行时检查声明的 kernel 是否真正被调用。

### 启用/禁用方式

1. **CLI 逐项开关**（覆盖配置）：

```bash
uv run npu-kernelbench eval tests/samples/matmul_triton \
    --solution tests/samples/matmul_triton/solution.json \
    --enable-anti-hack ast_scan \
    --enable-anti-hack kernel_signature
```

2. **`config.json`** 默认配置：

```json
{
  "enable_anti_hack": ["ast_scan", "kernel_signature"]
}
```

3. **全部关闭**：

```json
{
  "enable_anti_hack": []
}
```

4. **跳过性能 + anti-hack**：

```bash
uv run npu-kernelbench eval tests/samples/gelu \
    --solution tests/samples/gelu/solution.json \
    --mode correctness_only
```

### 可选检查

| 检查名 | 作用 |
|--------|------|
| `ast_scan` | 静态 AST 白名单/黑名单：禁止 `ctypes`、`vllm`、`cupy`、危险 `torch.*`、`exec/eval`、动态加载、别名绕过等。 |
| `kernel_signature` | 检查声明的 kernel（Python 装饰器函数 / AscendC `__global__` 函数）是否出现在 profiler 的 `kernel_details.csv`。 |
| `output_caching` | 5 次不同随机种子运行，输出完全相同则判定缓存作弊。 |
| `monkey_patch` | 重新计算 `definition.reference` 哈希并与 `reference_hash` 比较，检测 reference 被篡改。 |
| `clock_anomaly` | 检查迭代耗时变异系数（CV），异常高则判定时钟作弊。 |

完整说明见 `references/anti-hack.md`。

## 结果解读

每道 workload 产生一条 `Trace`，以 JSONL 写入 `-o` 文件：

```json
{
  "definition": "OperatorName",
  "solution": "solution_name",
  "workload": { "uuid": "...", "axes": {...}, "inputs": {...}, "tolerance": {...} },
  "evaluation": {
    "status": "PASSED",
    "environment": { "hardware": "Ascend_910B" },
    "timestamp": "...",
    "log": "",
    "correctness": { "max_absolute_error": 1e-05, "max_relative_error": 1e-05 },
    "performance": { "latency_ms": 123.4, "reference_latency_ms": 456.7, "speedup_factor": 3.70 }
  }
}
```

关键状态：

| 状态 | 含义 |
|------|------|
| `PASSED` | 通过 |
| `REWARD_HACK` | 触发防作弊检查 |
| `COMPILE_ERROR` | 编译失败（AscendC / Catlass） |
| `RUNTIME_ERROR` | 运行时异常 |
| `INCORRECT_NUMERICAL` | 数值精度不达标 |
| `INCORRECT_SHAPE` / `INCORRECT_DTYPE` | 输出形状/类型错误 |
| `TIMEOUT` | 超时 |
| `CRASH` | 进程崩溃 |

## 故障排查

| 现象 | 可能原因 | 解决方法 |
|------|----------|----------|
| `msprof not found` | 未 source CANN | `source /usr/local/Ascend/cann-*/set_env.sh` |
| `torch_npu not available` | torch 与 torch_npu 版本不匹配 | 检查并重新安装 |
| `compile_error` | AscendC 编译失败 | 查看 `--work-dir` 下的编译日志 |
| `timeout` | 算子太慢或死锁 | 增大 `--timeout`，或换 `--runner isolated` |
| 精度失败 | 数值误差或实现错误 | 查看 `max_absolute_error`，对比参考实现 |
| `REWARD_HACK` | 触发 anti-hack | 查看 `evaluation.log` 中的具体检查 |
| `solution.json` 找不到 | 路径错误 | 检查 `--solution` 或 `--solution-dir` |
| `entry_point` 校验失败 | 入口文件/函数命名错误 | 确认 `main.py` 和 `def run(...)` |

## 相关资源

| 文件 | 内容 |
|------|------|
| `references/solution-format.md` | `solution.json` 完整 schema 与按语言示例 |
| `references/anti-hack.md` | 防作弊检查详细说明 |
| `references/cli-reference.md` | CLI 选项表、resume、多 NPU、profiling、config.json schema |
| `scripts/pack_solution.py` | 把源码打包成 `solution.json` |
| `scripts/validate_solution.py` | 校验 `solution.json` 格式与约定 |
| `scripts/quick_eval.sh` | 单算子快捷评测脚本 |

