# NPUKernelBench

昇腾 NPU 算子评测与基准测试框架。

通过对比参考实现，评估自定义 NPU 算子方案（PyTorch/torch_npu、Ascend C/CANN、Triton）的正确性与性能。

## 安装

```bash
git clone <repo-url>
cd npu-kernelbench
uv sync
```

需要 Python >=3.10，Ascend NPU 硬件，已安装 torch（CPU 版）和 torch-npu（两者版本号须一致）以及 triton-ascend。

## 快速开始

```bash
# 评测单个算子（problem_dir 模式）
uv run npu-kernelbench eval tests/samples/matmul_triton \
    --solution tests/samples/matmul_triton/solution.json

# 显式指定路径
uv run npu-kernelbench eval \
    --definition tests/samples/gelu/definition.json \
    --workload tests/samples/gelu/workload.jsonl \
    --solution path/to/solution.json

# 批量评测数据集
uv run npu-kernelbench run-dataset data/kernel_generator --level level1

# 多级别 + 限制数量
uv run npu-kernelbench run-dataset data/kernel_generator \
    --level level1 --level level2 --limit 10

# 使用 isolated runner（默认 persistent）
uv run npu-kernelbench eval tests/samples/matmul_triton \
    --solution solution.json --runner isolated

# 指定 trace 输出路径（默认 /tmp/npu-kernelbench/traces.jsonl）
uv run npu-kernelbench eval tests/samples/gelu \
    --solution solution.json -o my_traces.jsonl

# 保存日志
uv run npu-kernelbench eval tests/samples/gelu \
    --solution solution.json --log-file eval.log --verbose
```

## CLI 命令

| 命令 | 说明 |
|------|------|
| `eval` | 评测单个算子方案 |
| `run-dataset` | 批量评测数据集目录下的所有算子 |

### eval 选项

```
--definition PATH       定义文件 definition.json 路径
--workload PATH         用例文件 workload.jsonl 路径
--solution PATH         方案文件 solution.json 路径（必填）
--config PATH           评测配置 JSON 路径
--compile-timeout N     编译超时秒数（默认 120）
--timeout N             评测子进程超时秒数（默认 600）
--limit N               最多评测的 workload 数量
--runner MODE           运行模式：persistent（默认）或 isolated
--npu-ids IDS           NPU 设备 ID 列表，逗号分隔（默认 0）
--profile-dir PATH      profiling 结果输出目录
--work-dir PATH         编译构建工作目录（默认 /tmp/npu-kernelbench）
--log-file PATH         日志输出文件
--json                  将 trace JSON 打印到 stdout
--verbose, -v           启用 debug 日志
-o, --output PATH       trace JSONL 输出路径（默认 /tmp/npu-kernelbench/traces.jsonl）
--html PATH              生成 HTML 报告到指定路径
--resume                 跳过 -o 中已有结果（中断后续测，不重跑已完成的算子/workload）
```

### run-dataset 选项

```
-l, --level TEXT        评测级别（默认 level1，可重复指定）
--limit N               每个算子最多评测的 workload 数量
--solution-dir PATH     按算子存放 solution.json 的目录
--timeout N             评测子进程超时秒数（默认 600）
--runner MODE           运行模式：persistent（默认）或 isolated
--npu-ids IDS           NPU 设备 ID 列表，逗号分隔（默认 0）
--profile-dir PATH      profiling 结果输出目录
--work-dir PATH         编译构建工作目录（默认 /tmp/npu-kernelbench）
--log-file PATH         日志输出文件
--verbose, -v           显示详细输出
-o, --output PATH       trace JSONL 输出路径（默认 /tmp/npu-kernelbench/traces.jsonl）
--html PATH             生成聚合 HTML 报告到指定路径
--resume                跳过 -o 中已有结果（中断后续测，不重跑已完成的算子/workload）
```

Trace 始终输出到 JSONL 文件，默认路径 `/tmp/npu-kernelbench/traces.jsonl`。

### 生成 HTML 报告

```bash
# 评测同时生成 HTML 报告
uv run npu-kernelbench eval tests/samples/matmul_triton \
    --solution solution.json --html report.html
```
浏览器打开 `report.html` 即可查看结果。

### 断点续测

```bash
# 批量运行被中断——续测，不重跑已完成的工作。
# -o 中已有的 workload（以及全部完成的算子）会被跳过；
# 输出文件保留已有结果，只追加新跑的。
uv run npu-kernelbench run-dataset data/kernel_generator \
    --level level1 -o runs.jsonl --resume

# eval 同样支持；数据集新增算子后也只会跑缺失的部分。
uv run npu-kernelbench eval tests/samples/matmul_triton \
    --solution solution.json -o runs.jsonl --resume
```

续测从 `-o` 文件读取（默认 `/tmp/npu-kernelbench/traces.jsonl`）。
续测同一次运行请使用相同的 `-o`；想要全新开始则先删除该文件。

## 配置

参考 `config.json`：

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
  "max_solution_failures": 3
}
```

## 架构

```
cli/main.py                  Click 命令行入口（eval, run-dataset）
    │
    ▼
runner/                      评测编排
├── runner.py                Runner 抽象基类
├── isolated_runner.py       每次评测独立子进程（NPU 完全隔离）
├── persistent_runner.py     长生命周期 worker 进程（默认，低开销）
└── eval.py                  独立可执行评测脚本（拷贝到 work_dir 用于问题复现）
    │
    ├── builder/             方案 → 可执行体 构建
    │   ├── python_builder.py   PyTorch、Triton、TileLang（importlib 加载）
    │   └── ascendc_builder.py  Ascend C（CANN 编译）
    │
    ├── evaluator/           评测流水线
    │   ├── generator.py     输入张量生成（gen_inputs、_rand_tensor、load_safetensors）
    │   ├── correctness.py   数值正确性检查
    │   ├── performance.py   NPU profiler 性能统计
    │   ├── anti_hack.py     反作弊检测
    │   ├── evaluator.py     Evaluator 抽象基类 + BaselineHandle
    │   ├── default.py       DefaultEvaluator（通用算子）
    │   └── registry.py      EvaluatorRegistry（按优先级匹配）
    │
    ├── data/                Pydantic v2 数据模型
    │   ├── definition.py   Definition（算子规格、符号化维度）
    │   ├── workload.py     Workload（InputSpec 类型、容差）
    │   ├── solution.py     Solution（源码文件、构建规格）
    │   ├── trace.py        Trace、Evaluation、Correctness、Performance
    │   ├── shapes.py       维度类型（const/var/expr）+ 表达式求值
    │   └── dtypes.py       数据类型枚举 + torch/python 转换
    │
    └── report/              结果输出
        ├── formatter.py     Rich 表格输出
        ├── score.py         评分计算
        ├── json_reporter.py JSON 导出
        └── html_reporter.py HTML 报告生成
```

### 执行流程

```
CLI → Runner.pre_build（拷贝源码和评测文件、编译、加载 .so）
    → Runner.run_workload（逐个 workload）
      → 子进程 worker：
         → Builder 加载 Solution
         → Evaluator.build_baseline：gen_inputs → 运行 reference
         → Evaluator.evaluate：运行 solution → 形状/dtype 检查 → 正确性 → 性能 → Trace
    → Report 格式化结果（Rich 表格 + 评分摘要）
    → Trace 保存到 JSONL（默认 /tmp/npu-kernelbench/traces.jsonl）
```

### 问题复现

评测完成后，`work_dir` 目录自包含所有所需文件：

```
work_dir/
├── definition.json
├── workload.jsonl
├── solution.json
├── eval.py              ← cd work_dir && python eval.py 即可复现
├── sources/
├── kernel.so            （仅 AscendC）
└── profile/
```

## 数据格式

### definition.json

定义算子的接口：符号化维度、张量规格、参考实现。

```json
{
  "name": "matmul_triton",
  "op_type": "matmul",
  "axes": {
    "M": {"type": "var", "description": "A 和 C 的行数"},
    "K": {"type": "var", "description": "内积维度"},
    "N": {"type": "var", "description": "B 和 C 的列数"}
  },
  "inputs": {
    "a": {"shape": ["M", "K"], "dtype": "float32"},
    "b": {"shape": ["K", "N"], "dtype": "float32"}
  },
  "outputs": {
    "c": {"shape": ["M", "N"], "dtype": "float32"}
  },
  "reference": "import torch\ndef run(a, b):\n    return torch.mm(a, b)\n"
}
```

维度类型：
- `"const"` — 固定值，如 `{"type": "const", "value": 2048}`
- `"var"` — 由 workload 在运行时确定
- `"expr"` — 由其他维度计算得出，如 `{"type": "expr", "expression": "M * 2"}`

### workload.jsonl（每行一个 JSON 对象）

指定具体的维度值和输入数据生成策略。

```jsonl
{"axes": {"M": 64, "K": 128, "N": 64}, "inputs": {"a": {"type": "random", "dtype": "float32"}, "b": {"type": "random", "dtype": "float32"}}, "uuid": "mm-001"}
{"axes": {"M": 128, "K": 256, "N": 128}, "inputs": {"a": {"type": "random", "dtype": "float16"}, "b": {"type": "random", "dtype": "float16"}}, "uuid": "mm-002"}
{"axes": {"M": 128, "K": 256, "N": 128}, "inputs": {"a": {"type": "random", "dtype": "bfloat16"}, "b": {"type": "random", "dtype": "bfloat16"}}, "uuid": "mm-003"}
```

InputSpec 类型：
- `{"type": "random", "dtype": "float16"}` — 随机张量，可覆盖 definition 中的 dtype
- `{"type": "scalar", "value": 2.5}` — 标量字面值
- `{"type": "custom"}` — 通过 `custom_inputs_entrypoint` 生成
- `{"type": "safetensors", "path": "...", "tensor_key": "..."}` — 从 safetensors 文件加载

### solution.json

具体的算子实现方案。

```json
{
  "name": "my_matmul",
  "definition": "matmul_triton",
  "author": "developer",
  "spec": {
    "languages": ["triton"],
    "target_hardware": ["ascend_910b"],
    "entry_point": "kernel.py::matmul",
    "destination_passing_style": false
  },
  "sources": [
    {
      "path": "kernel.py",
      "content": "import torch\nimport triton\n...\ndef matmul(a, b):\n    ...\n"
    }
  ]
}
```

支持的语言：`python`、`pytorch`、`triton`、`ascendc`。
支持的硬件：`ascend_910b`、`ascend_910c`、`LOCAL`。

## 示例问题

`tests/samples/` 目录包含参考问题：

| 目录 | 说明 | 语言 |
|------|------|------|
| `gelu/` | GELU 激活函数 | — |
| `identity/` | 恒等函数 | — |
| `matmul_pytorch/` | 矩阵乘法 | PyTorch |
| `matmul_triton/` | 矩阵乘法（Triton 算子） | Triton |
| `vector_add_ascendc/` | 向量加法 | Ascend C |

## 测试

```bash
# 运行全部测试
uv run pytest tests/

# 运行单个测试文件
uv run pytest tests/npu_kernelbench/data/test_definition.py -v

# 按关键字筛选
uv run pytest tests/ -k "test_correctness" -v

# 集成测试（需要 NPU）
uv run pytest tests/test_integration.py -v
```

标记为 `requires_npu` 或 `ascend_c` 的测试在缺少对应依赖时会自动跳过。

## 构建 Wheel

```bash
pip install build
python -m build --wheel
```
