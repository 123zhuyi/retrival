# CLI Reference

All commands are run from the project root and use `uv run npu-kernelbench ...`.

## Commands

| Command | Purpose |
|---------|---------|
| `eval` | Evaluate a single kernel solution. |
| `run-dataset` | Batch evaluate all kernels in a dataset level. |
| `report` | Generate an HTML report from an existing JSONL trace file. |

## `eval`

### Positional form

```bash
uv run npu-kernelbench eval <problem_dir> --solution solution.json
```

Looks for `definition.json`, `workload.jsonl`, and optionally `solution.json` inside `<problem_dir>`.

### Explicit form

```bash
uv run npu-kernelbench eval \
    --definition path/to/definition.json \
    --workload path/to/workload.jsonl \
    --solution path/to/solution.json
```

### Common options

| Option | Default | Description |
|--------|---------|-------------|
| `--solution PATH` | — | **Required.** Path to `solution.json`. |
| `--config PATH` | — | Path to `config.json`. |
| `--timeout N` | 600 | Subprocess timeout in seconds. |
| `--compile-timeout N` | 120 | Compilation timeout in seconds. |
| `--limit N` | — | Evaluate at most the first N workloads. |
| `--runner MODE` | `persistent` | `persistent` (worker reuse) or `isolated` (process per eval). |
| `--npu-ids IDS` | `0` | Comma-separated NPU device IDs. |
| `--profile-dir PATH` | — | Save raw profiling data. |
| `--work-dir PATH` | `/tmp/npu-kernelbench` | Staging directory. |
| `--log-file PATH` | — | Write logs to file. |
| `-o PATH` | `/tmp/npu-kernelbench/traces.jsonl` | Output trace JSONL. |
| `--html PATH` | — | Generate HTML report. |
| `--resume` | false | Skip workloads already present in `-o`. |
| `--mode correctness_only` | — | Skip performance + anti-hack. |
| `--perf-reference-source` | `measured` | `measured` or `baseline` for speedup. |
| `--enable-anti-hack NAME` | `ast_scan,kernel_signature` | Repeatable; see anti-hack reference. |
| `--verbose, -v` | false | Debug logging. |
| `--json` | false | Print trace JSON to stdout. |

### Examples

```bash
# Basic single evaluation
uv run npu-kernelbench eval tests/samples/gelu \
    --solution tests/samples/gelu/solution.json

# With HTML report
uv run npu-kernelbench eval tests/samples/gelu \
    --solution tests/samples/gelu/solution.json \
    --html /tmp/gelu_report.html

# Limit to first workload, correctness only
uv run npu-kernelbench eval tests/samples/gelu \
    --solution tests/samples/gelu/solution.json \
    --limit 1 --mode correctness_only

# Full anti-cheat
uv run npu-kernelbench eval tests/samples/matmul_triton \
    --solution tests/samples/matmul_triton/solution.json \
    --enable-anti-hack ast_scan \
    --enable-anti-hack kernel_signature \
    --enable-anti-hack output_caching \
    --enable-anti-hack monkey_patch \
    --enable-anti-hack clock_anomaly

# Resume
uv run npu-kernelbench eval tests/samples/gelu \
    --solution tests/samples/gelu/solution.json \
    -o /tmp/gelu.jsonl --resume
```

## `run-dataset`

```bash
uv run npu-kernelbench run-dataset <data_dir> \
    --scenario <scenario> \
    --level level1 \
    --solution-dir <solution_dir>
```

Directory layout expected:

```
data_dir/
└── scenario/
    └── level1/
        └── OperatorName/
            ├── definition.json
            └── workload.jsonl
```

If `--solution-dir` is given, the CLI looks for `solution.json` at:

```
<solution_dir>/<level>/<OperatorName>/solution.json
```

Otherwise it looks inside the operator directory.

### Common options

| Option | Default | Description |
|--------|---------|-------------|
| `--scenario NAME` | — | `ascendc_migration`, `kernel_generator`, or `triton_migration`. |
| `-l, --level TEXT` | `level1` | Repeatable. |
| `--limit N` | — | Max workloads per kernel. |
| `--solution-dir PATH` | — | External solution directory. |
| `--timeout N` | 600 | Subprocess timeout. |
| `--runner MODE` | `persistent` | `persistent` or `isolated`. |
| `--npu-ids IDS` | `0` | Comma-separated NPU IDs. |
| `--profile-dir PATH` | — | Save profiling data. |
| `--work-dir PATH` | `/tmp/npu-kernelbench` | Staging directory. |
| `--log-file PATH` | — | Write logs to file. |
| `-o PATH` | `/tmp/npu-kernelbench/traces.jsonl` | Output trace JSONL. |
| `--html PATH` | — | Generate aggregated HTML report. |
| `--resume` | false | Skip completed kernels/workloads. |
| `--mode correctness_only` | — | Skip performance + anti-hack. |
| `--perf-reference-source` | `measured` | `measured` or `baseline`. |
| `--enable-anti-hack NAME` | `ast_scan,kernel_signature` | Repeatable. |
| `--verbose, -v` | false | Verbose output. |

### Examples

```bash
# Single level
uv run npu-kernelbench run-dataset data/ascendc_migration \
    --scenario ascendc_migration \
    --level level1 \
    --solution-dir ./solutions

# Multiple levels, multi-NPU
uv run npu-kernelbench run-dataset data/kernel_generator \
    --scenario kernel_generator \
    --level level1 --level level2 \
    --npu-ids 0,1,2,3 \
    --solution-dir ./solutions \
    -o /tmp/kg_l1l2.jsonl \
    --html /tmp/kg_l1l2.html

# Resume
uv run npu-kernelbench run-dataset data/kernel_generator \
    --scenario kernel_generator \
    --level level2 \
    --solution-dir ./solutions \
    -o /tmp/kg_l2.jsonl --resume
```

## `report`

Generate HTML from an existing trace JSONL file.

```bash
uv run npu-kernelbench report /tmp/gelu.jsonl --html /tmp/gelu_report.html
```

## `config.json` schema

A `config.json` placed next to a problem or passed via `--config` can contain any of the following fields:

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
  "enable_anti_hack": ["ast_scan", "kernel_signature"],
  "perf_reference_source": "measured"
}
```

| Field | Default | Description |
|-------|---------|-------------|
| `warmup_runs` | 5 | Warmup iterations before timing. |
| `iterations` | 10 | Timing iterations. |
| `seed` | 200 | Random seed for input generation. |
| `timeout` | 600 | Subprocess timeout. |
| `compile_timeout` | 120 | Compilation timeout. |
| `benchmark_reference` | true | Whether to time the reference implementation. |
| `npu_id` | 0 | Deprecated; use `npu_ids`. |
| `npu_ids` | `[0]` | List of NPU device IDs. |
| `runner` | `persistent` | `persistent` or `isolated`. |
| `work_dir` | `/tmp/npu-kernelbench` | Staging directory. |
| `profile_dir` | `""` | Profiling output directory. |
| `health_check_interval` | 10 | Seconds between persistent worker health checks. |
| `max_solution_failures` | 3 | Max consecutive failures before a persistent worker restarts. |
| `enable_anti_hack` | `["ast_scan", "kernel_signature"]` | Enabled anti-hack checks. |
| `perf_reference_source` | `measured` | `measured` or `baseline`. |

## Runner choice

| Runner | Use when |
|--------|----------|
| `persistent` | Default. Faster for many workloads because workers stay alive. |
| `isolated` | Need maximum isolation; debugging crashes or state leakage. |

For batch dataset runs with many kernels, `persistent` is usually fastest. If you see mysterious crashes or ACL initialization errors, try `isolated`.
