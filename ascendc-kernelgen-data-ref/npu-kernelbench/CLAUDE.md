# CLAUDE.md

## Project Overview

npu-kernelbench is an Ascend NPU kernel evaluation and benchmarking framework. It evaluates custom NPU kernel solutions (PyTorch/torch_npu, Ascend C/CANN, Triton) for correctness and performance by comparing them against reference implementations.

## Build & Run Commands

```bash
# Install dependencies
uv sync

# Run the CLI
uv run npu-kernelbench eval <problem_dir> --solution solution.json

# Run all tests
uv run pytest tests/

# Run a single test file
uv run pytest tests/npu_kernelbench/data/test_definition.py

# Run tests matching a keyword
uv run pytest tests/ -k "test_correctness"

# Lint and format
uv run ruff check .
uv run ruff format .
```

## Architecture

Six top-level modules:

1. **cli** (`cli/`) — Click-based CLI entry point (`npu-kernelbench`). Two commands: `eval` (single kernel) and `run-dataset` (batch). Also houses `BenchmarkConfig`. Default runner mode is `persistent`, traces are written to `work_dir/traces.jsonl` by default.

2. **data** (`data/`) — Pydantic v2 models:
   - `definition.py` — Definition (kernel spec, symbolic axes: const/var/expr, reference code)
   - `workload.py` — Workload (concrete axis values, InputSpec types: random/scalar/custom/safetensors, tolerance)
   - `solution.py` — Solution (source files, build spec, entry point, languages: python/pytorch/triton/ascendc)
   - `trace.py` — Trace, Evaluation, Correctness, Performance, Environment
   - `shapes.py` — AxisConst, AxisVar, AxisExpr + `resolve_shape_expression()`
   - `dtypes.py` — DType enum (float32/16/bf16/fp8/fp4/int/bool) + torch conversions
   - `json_utils.py` — JSON/JSONL save/load/append utilities

3. **runner** (`runner/`) — Orchestrates evaluation:
   - `runner.py` — Runner ABC
   - `isolated_runner.py` — Process-per-evaluation (full NPU isolation)
   - `persistent_runner.py` — Long-lived worker processes with health checks and failure tracking
   - `eval.py` — Standalone eval driver, written to `work_dir` for reproduction (`cd work_dir && python eval.py`)

4. **builder** (`builder/`) — Transforms Solution → Runnable:
   - `python_builder.py` — PyTorch, Triton, TileLang (importlib-based)
   - `ascendc_builder.py` — Ascend C (CANN compilation)

5. **evaluator** (`evaluator/`) — Evaluation pipeline:
   - `generator.py` — Input tensor generation: `gen_inputs()` dispatches by InputSpec type, `_rand_tensor()` handles dtype-aware random generation, `load_safetensors()` loads safetensors files
   - `correctness.py` — `set_seed()`, `check_tensor_sanity()`, `compute_error_stats()`
   - `performance.py` — `profile_kernel()` (torch_npu profiler primary, NPU Event fallback)
   - `anti_hack.py` — Reward hack detection (output caching, monkey-patching, clock anomaly, kernel signature mismatch)
   - `evaluator.py` — Evaluator ABC + `BaselineHandle` dataclass
   - `default.py` — `DefaultEvaluator` (catch-all for general-purpose kernels)
   - `registry.py` — `EvaluatorRegistry` (priority-ordered, singleton)

6. **report** (`report/`) — Output formatting:
   - `formatter.py` — Rich table output
   - `score.py` — Score computation and summary
   - `json_reporter.py` — JSON export
   - `html_reporter.py` — Standalone HTML report generation

### Execution flow

```
CLI → Runner.pre_build (stage sources + eval files, compile, load .so)
    → Runner.run_workload (per workload)
      → subprocess worker:
         → importlib loads Solution entry_point
         → Evaluator.build_baseline:
            → gen_inputs (dispatch by InputSpec type, dtype-aware random)
            → run reference → BaselineHandle(inputs, outputs)
         → Evaluator.evaluate:
            → run solution → shape/dtype check → correctness → performance → Trace
    → Report formats results (Rich table + score)
    → Traces saved to JSONL (default: work_dir/traces.jsonl)
```

### Standalone reproduction

After evaluation, `work_dir` is self-contained:
```
work_dir/
├── definition.json
├── workload.jsonl
├── solution.json
├── eval.py              ← cd work_dir && python eval.py
├── sources/
├── kernel.so            (AscendC)
└── profile/
```

### Key design decisions

- **Input dtype**: `definition.inputs[].dtype` is the default; `workload.inputs[].RandomInput.dtype` overrides it for multi-dtype evaluation
- **BaselineHandle.inputs** is `list` (ordered by `definition.inputs`), not `dict`
- **Default runner** is `persistent` (worker reuse); use `--runner isolated` for process isolation
- **Trace output** is always written; default path is `/tmp/npu-kernelbench/traces.jsonl`, use `-o` to override

## Data format

See `tests/samples/` for working examples. Key conventions:

- `definition.json` — axes can be `const` (fixed value), `var` (runtime), `expr` (computed from others)
- `workload.jsonl` — one JSON object per line; `inputs[].dtype` on `RandomInput` overrides definition default
- `solution.json` — `entry_point` format: `"file_path::function_name"`; `languages` determines builder selection

## Style

Ruff for linting and formatting. Rule E741 is ignored. Exclude `data/` from lint.

## Dependencies

- Python >=3.10, torch (CPU), torch-npu (must match torch version), pydantic >=2.12.5, click >=8.0, rich >=13.0, triton-ascend >=3.2.0
- Optional: `safetensors` (for SafetensorsInput workloads)
