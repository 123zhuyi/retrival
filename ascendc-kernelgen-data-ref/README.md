<div align="center">


# 🧪 NPUKernelBench

<p align="center">
  <strong>Ascend NPU 算子评测与基准测试框架</strong>
</p>


<p align="center">
  <a href="#-快速开始">快速开始</a> •
  <a href="#-准备待测算子">准备算子</a> •
  <a href="#-运行评测">运行评测</a> •
  <a href="#-评测方法">评测方法</a> •
  <a href="#-数据格式">数据格式</a> •
  <a href="#-数据集分级">数据集</a> •
  <a href="#-场景">场景</a> •
  <a href="#-架构">架构</a> •
  <a href="#-测试">测试</a>
</p>


<p align="center">
  <img src="https://img.shields.io/badge/Python-3.10%2B-blue" alt="Python 3.10+">
  <img src="https://img.shields.io/badge/Hardware-Ascend%20NPU-orange" alt="Ascend NPU">
  <img src="https://img.shields.io/badge/License-Apache%202.0-green" alt="Apache 2.0">
</p>


</div>

---

## 📖 简介

**NPUKernelBench** 是一款面向华为昇腾（Ascend）NPU 的纯软件算子评测框架，支持对多种 DSL/原生算子实现进行端到端的**精度验证**与**性能测试**。

- 支持 `PyTorch`、`Triton-Ascend`、`TileLang`、`Ascend C/CANN` 多种算子实现语言
- 基于参考实现自动对比输出，判定正确性；正确性对照昇腾生态精度统一标准
- 基于 Torch NPU Profiler 测量端到端延迟，提供 NPU 理论硬件性能基准
- 支持**三种评测场景**：
  - `kernel_generator`：Agent 算子生成场景
  - `triton_migration`：GPU Triton 算子迁移到 Ascend NPU
  - `ascendc_migration`：AscendC kernel 从旧代际芯片（910B / 910_93）迁移到新代际芯片（950）场景
- 支持 `--scenario` 场景选择、`--perf-reference-source` 切换加速比基准
- 支持单算子评测、批量数据集评测、断点续跑、HTML 报告
- 算子级自定义输入生成器，保证复杂算子输入合法且可复现
- 内置反作弊检查（AST 扫描、kernel 签名校验、输出缓存、monkey-patch、时钟异常），可按场景灵活开关

---

## ⚙️ 安装

```bash
git clone <仓库地址>
cd npu-kernelbench
uv sync
```

### 环境要求

| 依赖      | 说明                                                |
| --------- | --------------------------------------------------- |
| Python    | >= 3.10                                             |
| 硬件      | Ascend NPU（ascend910b / ascend950） |
| torch     | 2.9.0               |
| torch-npu | 2.9.0                                      |

```bash
# 示例
pip install torch==2.9.0
pip install torch-npu==2.9.0
```

---

## 🚀 快速开始

### 1. 单算子评测

算子目录内需包含 `definition.json`、`workload.jsonl` 和 `solution.json`：

```bash
# 目录模式（推荐）
uv run npu-kernelbench eval tests/samples/matmul_triton \
    --solution tests/samples/matmul_triton/solution.json

# 显式指定路径&显式指定防作弊检测
uv run npu-kernelbench eval \
    --definition tests/samples/gelu/definition.json \
    --workload tests/samples/gelu/workload.jsonl \
    --solution path/to/solution.json
    --runner isolated \
    --enable-anti-hack ast_scan \
    --enable-anti-hack kernel_signature \
    --enable-anti-hack output_caching \
    --enable-anti-hack monkey_patch \
    --enable-anti-hack clock_anomaly

```

### 2. 批量数据集评测

```bash
# 评测 kernel_generator level1 全部算子
uv run npu-kernelbench run-dataset data/kernel_generator --level level1

# 指定场景：triton_migration / ascendc_migration
uv run npu-kernelbench run-dataset data --scenario triton_migration --level level1
uv run npu-kernelbench run-dataset data --scenario ascendc_migration --level level1

# 多 level + 限制每算子负载数
uv run npu-kernelbench run-dataset data/kernel_generator \
    --level level1 --level level2 --limit 10

# 生成聚合 HTML 报告
uv run npu-kernelbench run-dataset data/kernel_generator \
    --level level1 --html report.html
```

---

## 📝 准备待测算子

你只需要准备 `solution.json`。

`definition.json`（算子接口与参考实现）和 `workload.jsonl`（评测输入规格）由数据集提供；如果你想评测自定义算子，也可以按后文「数据格式」自行编写。

待测算子的源码唯一需要出现的地方就是 `solution.json` 的 `sources[].content` 中。

### solution.json 最小模板

```json
{
  "name": "my_operator",
  "definition": "operator_name_in_definition",
  "author": "developer",
  "spec": {
    "languages": ["triton"],
    "target_hardware": ["ascend910b"],
    "entry_point": "main.py::run",
    "destination_passing_style": false
  },
  "sources": [
    {
      "path": "main.py",
      "content": "import torch\n\ndef run(a, b):\n    return torch.add(a, b)\n"
    }
  ]
}
```

### 关键字段说明

| 字段                             | 必填 | 说明                                                        |
| -------------------------------- | ---- | ----------------------------------------------------------- |
| `definition`                     | 是   | 必须与目标 `definition.json` 中的 `name` 一致               |
| `spec.languages`                 | 是   | `python` / `triton` / `tilelang` / `ascendc` / `catlass`    |
| `spec.target_hardware`           | 是   | `ascend910b` / `ascend910_93` / `ascend950` / `LOCAL`       |
| `spec.entry_point`               | 是   | 推荐统一写成 `"main.py::run"`                               |
| `spec.destination_passing_style` | 否   | `true` 时框架会把输出张量作为最后一个参数传入并期望原地写入 |
| `spec.build_method`              | 否   | AscendC 专用：`bisheng`（默认）或 `cmake`                   |

### 入口函数 `run` 的约定

推荐入口：文件 `main.py` 中的函数 `run`，即 `entry_point = "main.py::run"`。

- 参数顺序：必须与 `definition.json` 中 `inputs` 定义的顺序完全一致。
- 非 DPS：`def run(a, b, ...): return output`，返回输出张量或张量元组。
- DPS：`def run(a, b, ..., output=None): ...; return output`，框架会把分配好的输出张量传进来。

### 常见实现模板

#### 1. PyTorch / 原生实现

`main.py`：

```python
import torch

def run(a, b):
    return torch.add(a, b)
```

`languages` 填 `["python"]`。

#### 2. Triton-Ascend

`main.py`：

```python
import torch
import triton
import triton.language as tl

@triton.jit
def add_kernel(x_ptr, y_ptr, out_ptr, n, BLOCK: tl.constexpr):
    pid = tl.program_id(0)
    offs = pid * BLOCK + tl.arange(0, BLOCK)
    mask = offs < n
    x = tl.load(x_ptr + offs, mask=mask)
    y = tl.load(y_ptr + offs, mask=mask)
    tl.store(out_ptr + offs, x + y, mask=mask)

def run(a, b):
    out = torch.empty_like(a)
    n = a.numel()
    BLOCK = 256
    grid = (triton.cdiv(n, BLOCK),)
    add_kernel[grid](a, b, out, n, BLOCK=BLOCK)
    return out
```

`languages` 填 `["triton"]`，`entry_point` 仍为 `"main.py::run"`。

#### 3. Ascend C

Ascend C 通常需要核函数文件 + host 入口文件两份源码。

`main.py`：

```python
import torch
import torch_npu

def run(a, b):
    return torch.ops.cpp_extension_acs.my_add(a, b)
```

`add_custom.asc`（核函数模板，需按实际算子实现）：

```cpp
#include <torch/extension.h>
#include "torch_npu/csrc/core/npu/NPUStream.h"
#include "kernel_operator.h"

// 你的 AscendC Kernel 类、__global__ __aicore__ 核函数、TORCH_LIBRARY 注册
// 完整示例请参考 tests/samples/vector_add_ascendc/
```

`languages` 填 `["ascendc"]`，多文件打包时两个源文件都要放进 `sources`。

TileLang、Catlass 等其它语言的写法可参考 `tests/samples/` 下的示例。

### 打包生成 solution.json

框架提供 `scripts/pack_solution.py`，自动读取源码文件并生成 `solution.json`。

单文件算子：

```bash
python scripts/pack_solution.py \
  --name my_add \
  --definition vector_add \
  --source main.py \
  --language python \
  --hardware ascend910b \
  --entry-point main.py::run \
  --output my_solutions/level1/vector_add/solution.json
```

多文件 Ascend C 算子：

```bash
python scripts/pack_solution.py \
  --name my_add \
  --definition vector_add \
  --source main.py \
  --source add_custom.asc \
  --language ascendc \
  --hardware ascend910b \
  --entry-point main.py::run \
  --output my_solutions/level1/vector_add/solution.json
```

未来会提供 skill 自动把源码目录转换为 `solution.json`；当前请先使用打包脚本。

### 文件放置位置

#### 单算子评测

推荐把 `solution.json` 和 `definition.json` / `workload.jsonl` 放在同一目录（即 problem_dir 模式）：

```text
my_operator/
├── definition.json      # 数据集提供
├── workload.jsonl       # 数据集提供
└── solution.json        # 你准备的待测算子
```

```bash
uv run npu-kernelbench eval my_operator \
    --solution my_operator/solution.json
```

如果 `definition.json` / `workload.jsonl` 来自数据集，只想替换 `solution.json`，也可以用显式路径：

```bash
uv run npu-kernelbench eval \
    --definition data/kernel_generator/level1/Matmul/definition.json \
    --workload data/kernel_generator/level1/Matmul/workload.jsonl \
    --solution my_solutions/level1/Matmul/solution.json
```

#### 批量数据集评测

把 `solution.json` 按照与数据集相同的目录层级放置，通过 `--solution-dir` 指定：

```text
my_solutions/
├── level1/
│   └── Matmul/
│       └── solution.json
└── level2/
    └── MatmulTriton/
        └── solution.json
```

```bash
uv run npu-kernelbench run-dataset data \
    --scenario kernel_generator --level level1 --level level2 \
    --solution-dir my_solutions
```

省略 `--solution-dir` 时，框架会直接在数据集目录下查找 `data/<scenario>/<level>/<Operator>/solution.json`。

---

## ▶️ 运行评测

### 单算子 eval

```bash
# 目录模式
uv run npu-kernelbench eval my_operator --solution my_operator/solution.json

# 显式路径
uv run npu-kernelbench eval \
    --definition my_operator/definition.json \
    --workload my_operator/workload.jsonl \
    --solution my_operator/solution.json
```

### 批量 run-dataset

```bash
# 评测 kernel_generator level1 全部算子
uv run npu-kernelbench run-dataset data/kernel_generator --level level1

# 指定场景
uv run npu-kernelbench run-dataset data --scenario triton_migration --level level1
uv run npu-kernelbench run-dataset data --scenario ascendc_migration --level level1

# 多 level
uv run npu-kernelbench run-dataset data/kernel_generator \
    --level level1 --level level2

# 限制每算子负载数
uv run npu-kernelbench run-dataset data/kernel_generator \
    --level level1 --limit 10

# 指定 solution 目录
uv run npu-kernelbench run-dataset data/kernel_generator \
    --level level1 --solution-dir my_solutions

# 生成 HTML 报告
uv run npu-kernelbench run-dataset data/kernel_generator \
    --level level1 --html report.html
```

### 性能基准与加速比

框架支持两种参考延迟来源，可通过 CLI 切换：

| 来源               | 说明                                     | 适用场景                                 |
| ------------------ | ---------------------------------------- | ---------------------------------------- |
| `measured`（默认） | 运行时测量 reference 实现的延迟          | `kernel_generator`                       |
| `baseline`         | 读取 `workload.perf_baseline.latency_ms` | `triton_migration` / `ascendc_migration` |

```bash
# 使用实测 reference latency 计算 speedup（默认）
uv run npu-kernelbench eval my_operator --solution solution.json

# 使用 workload 中预记录的 GPU/Ascend 基线计算 speedup
uv run npu-kernelbench eval my_operator --solution solution.json \
    --perf-reference-source baseline

# 场景级批量评测，使用 baseline 计算加速比
uv run npu-kernelbench run-dataset data --scenario triton_migration \
    --level level1 --perf-reference-source baseline
```

当 `--perf-reference-source measured` 且实测 reference 缺失时，会自动 fallback 到 `perf_baseline`。

### 常用选项

```bash
# 隔离运行（每轮独立进程，避免状态污染）
uv run npu-kernelbench eval my_operator --solution solution.json --runner isolated

# 仅做正确性验证，跳过性能测试
uv run npu-kernelbench eval my_operator --solution solution.json --mode correctness_only

# 多 NPU
uv run npu-kernelbench eval my_operator --solution solution.json --npu-ids 0,1,2,3

# 保存结果、日志、HTML 报告
uv run npu-kernelbench eval my_operator \
    --solution solution.json \
    -o my_traces.jsonl \
    --log-file eval.log \
    --html report.html \
    --verbose

# 断点续跑
uv run npu-kernelbench run-dataset data/kernel_generator \
    --level level1 -o runs.jsonl --resume
```

### eval 完整参数

| 参数                             | 说明                          | 默认值                              |
| -------------------------------- | ----------------------------- | ----------------------------------- |
| `--definition PATH`              | 算子定义文件路径              | —                                   |
| `--workload PATH`                | 负载文件路径                  | —                                   |
| `--solution PATH`                | 算子实现文件路径（必填）      | —                                   |
| `--config PATH`                  | 评测配置 JSON                 | —                                   |
| `--compile-timeout N`            | 编译超时（秒）                | 120                                 |
| `--timeout N`                    | 评测子进程超时（秒）          | 600                                 |
| `--limit N`                      | 最大评测负载数                | 不限                                |
| `--runner MODE`                  | `persistent` / `isolated`     | `persistent`                        |
| `--npu-ids IDS`                  | NPU 设备 ID（逗号分隔）       | 0                                   |
| `--profile-dir PATH`             | 保存性能分析轨迹              | —                                   |
| `--work-dir PATH`                | 工作目录                      | `/tmp/npu-kernelbench`              |
| `--log-file PATH`                | 日志文件                      | —                                   |
| `--json`                         | 输出 JSON 到 stdout           | —                                   |
| `--verbose, -v`                  | 调试日志                      | —                                   |
| `-o, --output PATH`              | 结果输出路径（JSONL）         | `/tmp/npu-kernelbench/traces.jsonl` |
| `--html PATH`                    | 生成 HTML 报告                | —                                   |
| `--resume`                       | 断点续跑                      | —                                   |
| `--mode MODE`                    | `correctness_only` 或默认全量 | —                                   |
| `--perf-reference-source SOURCE` | `measured` 或 `baseline`      | `measured`                          |

### run-dataset 完整参数

| 参数                             | 说明                                                         | 默认值                              |
| -------------------------------- | ------------------------------------------------------------ | ----------------------------------- |
| `--scenario SCENARIO`            | 场景：`kernel_generator` / `triton_migration` / `ascendc_migration` | `kernel_generator`                  |
| `-l, --level TEXT`               | 评测 level（可多次指定）                                     | `level1`                            |
| `--limit N`                      | 每算子最大负载数                                             | 不限                                |
| `--solution-dir PATH`            | 算子实现文件目录                                             | —                                   |
| `--timeout N`                    | 超时（秒）                                                   | 600                                 |
| `--runner MODE`                  | 运行模式                                                     | `persistent`                        |
| `--npu-ids IDS`                  | NPU 设备 ID                                                  | 0                                   |
| `--profile-dir PATH`             | 性能分析轨迹目录                                             | —                                   |
| `--work-dir PATH`                | 工作目录                                                     | `/tmp/npu-kernelbench`              |
| `--log-file PATH`                | 日志文件                                                     | —                                   |
| `--verbose, -v`                  | 详细输出                                                     | —                                   |
| `-o, --output PATH`              | 结果输出（JSONL）                                            | `/tmp/npu-kernelbench/traces.jsonl` |
| `--html PATH`                    | 聚合 HTML 报告                                               | —                                   |
| `--resume`                       | 断点续跑                                                     | —                                   |
| `--perf-reference-source SOURCE` | `measured` 或 `baseline`                                     | `measured`                          |

---

## 📐 评测方法与结果标准

本章节说明 NPUKernelBench 如何判定一个算子实现“正确”以及“多快”，并解释最终生成的 `Trace` 结果。

### 评测流程

对每一个 `(definition, solution, workload)` 三元组，框架按以下顺序执行：

1. **输入生成**  
   根据 `workload.jsonl` 中的 `inputs` 规格生成张量（`generator.py`）。支持随机张量、标量、自定义生成器、 safetensors 文件、张量列表等。所有随机输入使用固定 `seed`（默认 200）以保证可复现。

2. **参考实现运行（Baseline）**  
   使用与 solution 完全相同的输入运行 `definition.reference` 中的 `run(...)` 函数，得到 golden 输出。结果封装为 `BaselineHandle(inputs, outputs)`。

3. **Solution 运行**  
   通过 `entry_point` 调用 solution 的入口函数，传入同样的输入，得到待测输出。

4. **形状与类型检查**  
   比较 solution 输出与 reference 输出的：
   - 输出数量
   - 每个输出张量的 `shape`
   - 每个输出张量的 `dtype`  
   任一不匹配即返回 `INCORRECT_SHAPE` 或 `INCORRECT_DTYPE`。

5. **精度校验（Correctness）**  
   调用 `src/npu_kernelbench/evaluator/correctness.py` 中的 `compute_error_stats`。详见下节「正确性判定标准」。

6. **性能测试（Performance）**  
   仅当正确性通过且未启用 `correctness_only` 模式时执行。调用 `src/npu_kernelbench/evaluator/performance.py` 中的 `profile_kernel`，测量 solution 与 reference 的端到端延迟。详见下节「性能测量方法」。

7. **反作弊检查（Anti-hack）**  
   根据 `enable_anti_hack` 配置运行 AST 扫描、kernel 签名检查、输出缓存检测、monkey-patch 检测、时钟异常检测等。任一检查触发会将状态改写为 `REWARD_HACK`。

8. **生成 Trace**  
   将结果写入 JSONL，每条记录是一个 `Trace` 对象，包含 `definition`、`solution`、`workload`、`evaluation`（状态、正确性、性能、日志、环境）。

---

### 正确性判定标准（精度标准）

NPUKernelBench 遵循昇腾生态开源精度统一标准，采用**混合容差 + 双条件通过**机制。

#### 浮点输出

对每个输出元素，定义误差：

```text
|actual - golden| <= atol + rtol * |golden|
```

一条 workload **通过**当且仅当同时满足：

1. **匹配比例** `matched_ratio = count(matched) / total >= required_matched_ratio`
2. **最大绝对误差** `max_abs_error <= max_abs_error_limit`

默认 `required_matched_ratio = 0.9`。若 `workload.tolerance` 中显式指定了 `required_matched_ratio`，则以 workload 为准。

#### 默认 dtype 阈值

| dtype | rtol | atol | max_abs_error_limit |
|-------|------|------|---------------------|
| `float32` | 2⁻¹⁰ | 2⁻¹⁶ | 1e-2 |
| `float16` | 2⁻⁹  | 2⁻⁹  | 1e-1 |
| `bfloat16` | 2⁻⁶  | 2⁻⁶  | 1.0 |
| `hifloat32` | 2⁻⁹  | 2⁻¹⁰ | 1e-1 |
| `float8_e4m3` | 2⁻²  | 2⁻⁴  | 1.0 |
| `float8_e5m2` | 2⁻¹  | 2⁻³  | 1e-1 |

未知 dtype 默认回退到 `float32` 阈值。

#### Workload 级覆盖

在 `workload.jsonl` 的 `tolerance` 字段中可以覆盖默认值：

```json
{"max_atol": 1e-3, "max_rtol": 1e-3, "max_error_cap": 1.0, "required_matched_ratio": 0.95, "allow_negative_inf": true}
```

| 字段 | 作用 |
|------|------|
| `max_atol` | 覆盖默认 atol |
| `max_rtol` | 覆盖默认 rtol |
| `max_error_cap` | 覆盖默认 max_abs_error_limit |
| `required_matched_ratio` | 覆盖默认匹配比例 |
| `allow_negative_inf` | 允许 reference 与 solution 同时出现 `-inf` 而不判错 |

#### 非浮点输出

- **bool 输出**：严格使用 `torch.equal`，必须完全一致。
- **整数输出**：
  - 若输入为浮点、输出为整数（典型量化场景）：允许逐元素差值 `|diff| <= 1`。
  - 若输入为整数、输出为整数：默认严格相等；对 `int8`/`uint8` 输出，可通过 `tolerance.max_atol >= 1` 允许 off-by-one。
- **混合 int/float**：按量化 fp→int 处理，允许 `|diff| <= 1`。

#### NaN / Inf 与全零检查

- **NaN**：输出中 NaN 出现的位置必须与 reference 完全一致。
- **Inf**：Inf 出现的位置与符号必须与 reference 完全一致（受 `allow_negative_inf` 影响）。
- **全零作弊**：当 reference 输出非零且 norm > 0 时，若 solution 输出全 0，直接判定精度失败。

---

### 性能测量方法

NPUKernelBench 以 **torch_npu profiler** 为主要计时源，并提供多层 fallback，确保在各种环境下都能获得稳定延迟。

#### 主计时源：torch_npu profiler

```python
with torch_npu.profiler.profile(
    activities=[CPU, NPU],
    schedule=torch_npu.profiler.schedule(
        wait=0, warmup=warmup_runs, active=iterations, repeat=1
    ),
    on_trace_ready=torch_npu.profiler.tensorboard_trace_handler(trace_dir),
    ...
) as prof:
    for _ in range(warmup_runs + iterations):
        kernel_fn(*inputs)
        torch.npu.synchronize(device)
        prof.step()
```

- **warmup**：默认 5 次（可配置 `warmup_runs`）。
- **active**：默认 10 次（可配置 `iterations`）。
- 均值延迟取自 `step_trace_time.csv` 的 `Computing` 列，除以 `iterations` 得到单次 latency（ms）。

#### Fallback 计时

若 profiler 无法解析或数据缺失，依次降级：

1. **kernel_details.csv**：将 `Duration(us)` 总和除以 `iterations`。
2. **NPU Event 墙钟计时**：使用 `torch.npu.Event(enable_timing=True)` 记录 `kernel_fn` 执行时间，丢弃第一次冷启动。

#### 加速比计算

- **实测模式**（`--perf-reference-source measured`，默认）：reference 与 solution 都通过上述方法测量。
- **基线模式**（`--perf-reference-source baseline`）：reference 延迟使用 `workload.perf_baseline.latency_ms`，solution 仍实测。
- 若实测 reference 失败或缺失，自动 fallback 到 `perf_baseline`。

加速比公式：

```text
speedup = reference_latency_ms / solution_latency_ms
```

#### Profiling 原始数据

```bash
uv run npu-kernelbench eval my_operator \
    --solution solution.json \
    --profile-dir /tmp/my_operator_prof
```

`--profile-dir` 会保留 profiler 原始 CSV，可用 `msprof` 或可视化工具进一步分析 kernel 级耗时。

---

### 结果状态与 Trace 解读

每条 workload 生成一条 `Trace`（JSONL 中的一行）。核心结构：

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
    "correctness": {
      "max_absolute_error": 1e-05,
      "max_relative_error": 1e-05,
      "extra": { "matched_ratio": 0.99, "atol": 1e-09, "rtol": 0.0009765625, "max_abs_error_limit": 0.1 }
    },
    "performance": {
      "latency_ms": 123.4,
      "reference_latency_ms": 456.7,
      "speedup_factor": 3.70
    }
  }
}
```

#### 状态码

| 状态 | 含义 |
|------|------|
| `PASSED` | 正确性与性能（如启用）均通过 |
| `INCORRECT_NUMERICAL` | 形状/类型通过，但数值精度不达标 |
| `INCORRECT_SHAPE` | 输出 shape 与 reference 不一致 |
| `INCORRECT_DTYPE` | 输出 dtype 与 reference 不一致 |
| `RUNTIME_ERROR` | solution 运行时抛出异常 |
| `COMPILE_ERROR` | AscendC / Catlass 编译失败 |
| `TIMEOUT` | 单 workload 执行超过 `--timeout` |
| `CRASH` | 子进程崩溃 |
| `REWARD_HACK` | 触发反作弊检查 |
| `INVALID_REFERENCE` | reference 实现本身无法运行 |

#### Correctness 字段

| 字段 | 说明 |
|------|------|
| `max_absolute_error` | 最大绝对误差 |
| `max_relative_error` | 最大相对误差 |
| `has_nan` | 输出存在非预期 NaN |
| `has_inf` | 输出存在非预期 Inf |
| `extra` | 包含 `matched_ratio`、`matched_count`、`atol`、`rtol`、`max_abs_error_limit` 等诊断信息 |

#### Performance 字段

| 字段 | 说明 |
|------|------|
| `latency_ms` | solution 平均延迟（ms） |
| `reference_latency_ms` | reference 平均延迟或 `perf_baseline`（ms） |
| `speedup_factor` | 加速比 |

> 当状态不是 `PASSED` 时，`performance` 字段通常为 `null`；当状态不是 `PASSED`/`INCORRECT_NUMERICAL` 时，`correctness` 字段通常为 `null`。

---

## 🎯 场景

NPUKernelBench 支持三类评测场景，目录结构统一放在 `data/` 下：

```text
data/
├── kernel_generator/       # 通用算子生成
│   ├── level1/
│   │   ├── Matmul/
│   │   │   ├── definition.json
│   │   │   ├── workload.jsonl
│   │   │   └── solution.json
│   │   └── ...
│   └── level2/
├── triton_migration/       # GPU Triton → NPU Triton 迁移
│   ├── level1/
│   │   ├── Index/
│   │   │   ├── definition.json   # 含 kernel_reference（原始 GPU Triton）
│   │   │   ├── workload.jsonl    # 含 perf_baseline（GPU 延迟基线）
│   │   │   └── solution.json     # NPU Triton 实现
│   │   └── ...
│   └── level2/
└── ascendc_migration/      # AscendC 跨代际迁移
    ├── level1/
    │   ├── AdaptiveAvgPool3d/
    │   │   ├── definition.json   # 含 kernel_reference（源文件列表）
    │   │   ├── workload.jsonl    # 含 perf_baseline（源硬件延迟基线）
    │   │   └── solution.json     # 目标平台 AscendC/Python 封装
    │   └── ...
    └── level2/
```

### kernel_generator

目标：面向通用算子生成任务，评测任意 NPU 实现（PyTorch / Triton / AscendC 等）。

- 正确性参考：`definition.reference`（等价 torch 实现）。
- 性能参考：默认运行时测量 reference 延迟，计算 `speedup = reference_latency / solution_latency`。

### triton_migration

目标：GPU Triton kernel 迁移到 Ascend NPU。

- 正确性参考：`definition.reference`（等价 torch 实现）。
- 迁移源：`definition.kernel_reference`（原始 GPU Triton 源码，不参与校验，仅作参考）。
- 性能基线：`workload.perf_baseline`（预记录 GPU 延迟，如 NVIDIA_A100）。
- 加速比：`speedup = perf_baseline.latency_ms / solution_latency_ms`。

### ascendc_migration

目标：AscendC 算子跨 CANN/SOC 版本迁移。

- 正确性参考：`definition.reference`（等价 torch 实现）。
- 迁移源：`definition.kernel_reference`（源平台文件列表/源码，不参与校验）。
- 性能基线：`workload.perf_baseline`（预记录源硬件延迟，如 Ascend910B2C）。
- 加速比：`speedup = perf_baseline.latency_ms / solution_latency_ms`。

---

## 📊 数据集分级

Agent 算子生成场景数据集按难度分为四个级别：

| 级别    | 目录      |     算子数量 | 描述                                                |
| ------- | --------- | -----------: | --------------------------------------------------- |
| Level 1 | `level1/` |           33 | 基础 Vector 算子（如 GELU、Add、Sort 等）           |
| Level 2 | `level2/` |           31 | 复杂融合算子（如 RotaryMul、MoeFinalizeRouting 等） |
| Level 3 | `level3/` | 10（新增中） | Cube 算子（如 BatchMatmul、ConvStandard 等）        |
| Level 4 | `level4/` | 12（新增中） | CV 融合算子（如 LSTM、FFN、FusionAttention 等）     |

---

## 📄 数据格式

### definition.json

```json
{
  "name": "matmul_triton",
  "op_type": "matmul",
  "axes": {
    "M": {"type": "var", "description": "A 和 C 的行数"},
    "K": {"type": "var", "description": "收缩维度"},
    "N": {"type": "var", "description": "B 和 C 的列数"}
  },
  "inputs": {
    "a": {"shape": ["M", "K"], "dtype": "float32"},
    "b": {"shape": ["K", "N"], "dtype": "float32"}
  },
  "outputs": {
    "c": {"shape": ["M", "N"], "dtype": "float32"}
  },
  "reference": "import torch\ndef run(a, b):\n    return torch.mm(a, b)\n",
  "kernel_reference": {
    "path": "matmul_gpu_triton.py",
    "content": "import triton\nimport triton.language as tl\n..."
  },
  "description": "矩阵乘法"
}
```

轴类型：

- `"const"` — 固定值，例如 `{"type": "const", "value": 2048}`
- `"var"` — 运行时由 workload 确定
- `"expr"` — 由其他轴计算，例如 `{"type": "expr", "expression": "M * 2"}`

新增字段：

| 字段                    | 说明                                                         |
| ----------------------- | ------------------------------------------------------------ |
| `kernel_reference`      | 迁移场景下的原始 kernel 源码（GPU Triton 代码或 AscendC 文件列表），不参与正确性校验 |
| `kernel_reference_repo` | 可选，原始 kernel 所在仓库地址                               |

### workload.jsonl

```jsonl
{"axes": {"M": 64, "K": 128, "N": 64}, "inputs": {"a": {"type": "random", "dtype": "float32"}, "b": {"type": "random", "dtype": "float32"}}, "uuid": "mm-001"}
{"axes": {"M": 128, "K": 256, "N": 128}, "inputs": {"a": {"type": "random", "dtype": "float16"}, "b": {"type": "random", "dtype": "float16"}}, "uuid": "mm-002", "perf_baseline": {"hardware": "NVIDIA_A100", "latency_ms": 0.012, "notes": "CUDA 12.2"}}
```

新增字段：

| 字段            | 说明                                                         |
| --------------- | ------------------------------------------------------------ |
| `perf_baseline` | 预记录性能基线：`hardware`（源硬件）、`latency_ms`（延迟）、`notes`（测量备注） |

### solution.json

```json
{
  "name": "my_matmul",
  "definition": "matmul_triton",
  "author": "developer",
  "spec": {
    "languages": ["triton"],
    "target_hardware": ["ascend910b"],
    "entry_point": "kernel.py::matmul",
    "destination_passing_style": true
  },
  "sources": [
    {
      "path": "kernel.py",
      "content": "import torch\nimport triton\n...\ndef matmul(a, b):\n    ...\n"
    }
  ]
}
```

---

## 🏗️ 架构

```text
cli/main.py                  Click CLI（eval / run-dataset）
    │
    ▼
runner/                      执行编排
├── runner.py                Runner 抽象基类
├── isolated_runner.py       每轮独立进程（完全隔离）
├── persistent_runner.py     长驻 worker 进程（默认）
└── eval.py                  独立复现脚本（staging 到 work_dir）
    │
    ├── builder/             Solution → 可执行算子
    │   ├── python_builder.py    PyTorch / Triton / TileLang（importlib）
    │   └── ascendc_builder.py   Ascend C（bisheng / CMake 编译）
    │
    ├── evaluator/           评测流水线
    │   ├── generator.py     输入张量生成
    │   ├── correctness.py   数值正确性校验
    │   ├── performance.py   CANN Profiler 性能测试
    │   ├── anti_hack.py     反作弊检测
    │   ├── evaluator.py     Evaluator ABC + BaselineHandle
    │   ├── default.py       通用算子评测器
    │   └── registry.py      EvaluatorRegistry
    │
    ├── data/                Pydantic v2 数据模型
    │   ├── definition.py    算子定义（符号轴、张量规格、参考实现、kernel_reference）
    │   ├── workload.py      负载（InputSpec、容忍度、perf_baseline）
    │   ├── solution.py      算子实现（源码、构建配置）
    │   ├── trace.py         评测轨迹与结果
    │   ├── shapes.py        轴类型（const / var / expr）
    │   └── dtypes.py        数据类型枚举与转换
    │
    └── report/              输出格式化
        ├── formatter.py     Rich 表格输出
        ├── score.py         分数汇总
        ├── json_reporter.py JSON 导出
        └── html_reporter.py HTML 报告
```

### 执行流程

```text
CLI → Runner.pre_build（准备源码、评测文件、编译）
    → Runner.run_workloads_batch（逐个负载执行）
      → worker 进程：
         → Builder 加载 Solution
         → Evaluator.build_baseline: gen_inputs → 运行参考实现
         → Evaluator.evaluate: 运行算子 → 正确性 → 性能 → Trace
    → Report 格式化结果（Rich 表格 + 分数汇总）
    → Traces 保存为 JSONL
```

### 独立复现

评测完成后，`work_dir` 包含独立复现所需的全部文件：

```text
work_dir/
├── definition.json
├── workload.jsonl
├── solution.json
├── eval.py              # cd work_dir && python eval.py 即可复现
├── sources/
├── kernel.so            # AscendC 场景
└── profile/
```

---

### 反作弊检查

NPUKernelBench 在评测流水线中内置多层反作弊检查，防止投机/作弊行为影响结果可信度。

| 层级 | 名称 | 检查时机 | 目标作弊行为 |
|------|------|----------|--------------|
| L0 | `ast_scan` | 源码加载前 | 危险导入、禁用 API、动态加载、非法 torch_npu 调用等 |
| L1 | `kernel_signature` | 性能 profiling 后 | 源码声明的 kernel 与实际执行的 kernel 不一致 |
| L2 | `output_caching` | 性能 profiling 后 | 不同输入下返回相同输出（缓存/硬编码） |
| L2 | `monkey_patch` | 性能 profiling 后 | 参考实现被运行时篡改 |
| L2 | `clock_anomaly` | 性能 profiling 后 | 计时数据异常波动 |

通过 `config.json` 或 CLI 控制启用哪些检查：

```bash
# CLI 示例：启用全部检查
uv run npu-kernelbench eval my_operator \
  --solution solution.json \
  --enable-anti-hack ast_scan \
  --enable-anti-hack kernel_signature \
  --enable-anti-hack output_caching \
  --enable-anti-hack monkey_patch \
  --enable-anti-hack clock_anomaly
```

`config.json` 示例：

```json
{
  "enable_anti_hack": ["ast_scan", "kernel_signature"]
}
```

- 默认：`["ast_scan", "kernel_signature"]`
- 严格评测（比赛/打榜）：`["ast_scan", "kernel_signature", "output_caching", "monkey_patch", "clock_anomaly"]`
- 关闭全部：`[]`

> 注意：`monkey_patch` 依赖 `definition.reference_hash` 已正确填充；为空时自动跳过。`kernel_signature` 在源码未声明 kernel 时自动跳过。

当任意反作弊检查触发时，`Trace.evaluation.status` 会从 `PASSED` 变为 `REWARD_HACK`，并保留具体的作弊描述在 `log` 中。

---

## 🔧 配置

`config.json` 示例：

```json
{
  "warmup_runs": 5,
  "iterations": 10,
  "seed": 200,
  "timeout": 600,
  "compile_timeout": 120,
  "benchmark_reference": true,
  "npu_id": 0,
  "npu_ids": [0],
  "runner": "persistent",
  "work_dir": "/tmp/npu-kernelbench",
  "profile_dir": "",
  "health_check_interval": 10,
  "max_solution_failures": 3,
  "enable_anti_hack": ["ast_scan", "kernel_signature"]
}
```

| 字段                    | 说明                                      |
| ----------------------- | ----------------------------------------- |
| `warmup_runs`           | 计时前 warmup 次数                        |
| `iterations`            | 每次负载计时次数                          |
| `seed`                  | 输入生成随机种子                          |
| `timeout`               | 单负载超时                                |
| `compile_timeout`       | Ascend C 编译超时                         |
| `benchmark_reference`   | 是否测量参考实现延迟                      |
| `perf_reference_source` | `measured` 或 `baseline`，默认 `measured` |
| `runner`                | `persistent`（默认）或 `isolated`         |
| `enable_anti_hack`      | 反作弊检查列表，可选 `ast_scan` / `kernel_signature` / `output_caching` / `monkey_patch` / `clock_anomaly`，`[]` 表示关闭 |

---

## 🧪 测试

```bash
# 运行全部测试
uv run pytest tests/

# 运行单个测试文件
uv run pytest tests/npu_kernelbench/data/test_definition.py -v

# 按关键词运行
uv run pytest tests/ -k "test_correctness" -v

# 运行集成测试（需要 NPU）
uv run pytest tests/test_integration.py -v
```

需要 NPU 或 Ascend C 编译环境的测试会在依赖不可用时自动跳过。

---

## 📦 构建 Wheel

```bash
pip install build
python -m build --wheel
```

---

## 📁 参考示例

`tests/samples/` 目录包含参考算子：

| 目录                  | 说明                    | 语言     |
| --------------------- | ----------------------- | -------- |
| `gelu/`               | GELU 激活函数           | —        |
| `identity/`           | 恒等函数                | —        |
| `matmul_pytorch/`     | 矩阵乘法                | PyTorch  |
| `matmul_triton/`      | 矩阵乘法（Triton 算子） | Triton   |
| `vector_add_ascendc/` | 向量加法                | Ascend C |

---

## 🙏 Acknowledgements

NPUKernelBench 的整体架构与部分实现模式参考了 [FlashInfer-Bench](https://github.com/flashinfer-ai/flashinfer-bench)（文档），该项目是以 Apache 2.0 协议开源的 LLM 算子基准测试套件。