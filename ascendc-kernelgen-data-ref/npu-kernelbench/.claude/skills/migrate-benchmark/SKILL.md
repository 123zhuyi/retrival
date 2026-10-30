---
name: migrate-benchmark
description: >
  Create npu-kernelbench benchmark problem directories (definition.json + workload.jsonl + solution.json)
  from user-provided reference implementations and operator code. Use this skill whenever the user provides
  a reference kernel implementation and wants to benchmark it, asks to create a benchmark problem, mentions
  setting up definition.json, workload.jsonl, or solution.json files, wants to evaluate a custom NPU kernel
  against a reference, or needs to convert any operator into npu-kernelbench format.
---

# Create Benchmark Problem for npu-kernelbench

## Overview

This skill guides the creation of a complete npu-kernelbench benchmark problem from user-provided materials. Given:

1. **Reference implementation** — correct (but possibly slow) PyTorch/Python code the solution will be compared against
2. **Workload descriptions** — set of test cases with shapes, dtypes, and scalar parameters
3. **Operator/kernel code** (optional) — the optimized implementation to benchmark

The skill produces:

```
problem_dir/
├── definition.json    ← formal kernel specification + reference code
├── workload.jsonl     ← concrete test cases (one per line)
└── solution.json      ← wrapped operator code (if provided)
```

**Default style is NPU.** Shape and dtype go on each workload line, not in `definition.json`. Use Community style (symbolic axes) only for simple, uniform operators.

## Step 1: Gather Information

Before writing any files, clarify with the user:

### Reference code
The reference must be a standalone function `def run(...)` that takes tensors + scalars and returns tensor(s). If the user provides a PyTorch `nn.Module.forward`, transform it:
- Drop `self` → standalone function
- Keep default values for non-tensor parameters
- Remove docstrings and comments (move those to `definition.description`)

### Workload descriptions
The user should describe test cases. Each test case needs:
- Concrete tensor shapes (list of ints)
- Dtype for each tensor (float16, bfloat16, float32, int64, etc.)
- Scalar parameter values
- Optional: tolerance overrides

If the user describes workloads loosely (e.g., "test shapes from 128 to 4096 in powers of 2, float16"), expand that into concrete test cases.

### Operator code (optional)
If the user provides an optimized kernel, ask:
- What language? (python, triton, ascendc, tilelang, catlass)
- What hardware? (default: `ascend910b`)
- Entry point: which file + which function?
- Any additional source files or build flags?

## Step 2: Analyze the Operator

Read the reference code and answer these questions:

| Question | How to determine |
|---|---|
| What does this compute? | Read the function body |
| Which params are tensors vs scalars? | Tensors have shape/dtype concerns; scalars are just values |
| What is the output shape relative to inputs? | Trace the tensor operations |
| What is the output dtype? | Usually same as input; bool for predicates; int64 for indices |
| Do dtypes vary across test cases? | If yes → NPU style required |

### Choose Definition Style

| Criterion | NPU Style (default) | Community Style |
|---|---|---|
| Dtype varies across test cases | ✅ | ❌ |
| Complex shape variation | ✅ | ❌ |
| Simple, uniform operator (same dtype, clean shape variation) | 🟡 Works | ✅ Simpler |
| Need symbolic axis validation | ❌ | ✅ |

**Decision rule:** Default to NPU style. Use Community style only when ALL test cases share the same dtype AND shape variation can be cleanly expressed via const/var/expr axes.

### Derive Axis Names (Community style only)

For Community style, assign meaningful axis names based on operator category:

| Operator category | Axis naming | Example |
|---|---|---|
| Elementwise activations (GELU, ReLU, etc.) | `D0`, `D1`, ... | `["D0", "D1"]` |
| Transformer norms (LayerNorm, RMSNorm) | `batch`, `seq_len`, `hidden` | `["batch", "seq_len", "hidden"]` |
| Image norms | `N`, `C`, `H`, `W` | `["N", "C", "H", "W"]` |
| Matmul | `M`, `K`, `N` | `["M", "K"], ["K", "N"]` |
| Binary elementwise (Add, Mul) with broadcasting | Primary: e.g. `["D0", "D1"]`, secondary may use `["D0", "1"]` | |
| Reduction (Sum, Cumsum) | `dim` for the reduced dimension | |
| Pooling (AvgPool3d, MaxPool3d) | `N`, `C`, `D`, `H`, `W` | |

Determine const vs var: scan ALL shapes for that dimension across ALL test cases. Same value every time → `const`. Varies → `var`.

## Step 3: Write `definition.json`

### NPU style (default)

Shape and dtype are **omitted** from TensorSpec. Semantic capability is described in `description`. `axes` is `{}`.

```json
{
  "name": "OperatorName",
  "op_type": "elementwise|matmul|norm|attention|...",
  "axes": {},
  "inputs": {
    "param_name": {"description": "shape (B, S, D); supports float16, bfloat16"},
    "other_param": {"description": "same shape and dtype as param_name"}
  },
  "outputs": {
    "output_name": {"description": "same shape and dtype as inputs"}
  },
  "reference": "import torch\n\ndef run(x):\n    return torch.nn.functional.gelu(x)\n",
  "description": "Full operator documentation including math formulas, parameter docs, and edge cases."
}
```

- `axes` is empty `{}`
- `shape` and `dtype` are omitted in all TensorSpecs
- `description` on each tensor carries human-readable capability notes
- `description` on the definition carries the full operator documentation
- For non-tensor (scalar/attr) inputs, use `description` to note type and valid range

### Community style

Shape and dtype are specified in Definition. Axes are defined with types.

```json
{
  "name": "OperatorName",
  "axes": {
    "axis_name": {"type": "var", "description": "what this axis represents"}
  },
  "inputs": {
    "param_name": {"shape": ["axis1", "axis2"], "dtype": "float32", "description": "brief"}
  },
  "outputs": {
    "output_name": {"shape": ["axis1", "axis2"], "dtype": "float32", "description": "brief"}
  },
  "reference": "import torch\n\ndef run(x):\n    return torch.nn.functional.gelu(x)\n",
  "description": "Full operator documentation."
}
```

### Reference code rules
- Must define a top-level `def run(...)` function
- Use only standard torch ops (`torch.*`, `torch.nn.functional.*`)
- **Exception:** Preserve vendor-specific APIs (`torch_npu.*`) in reference code as-is. Document the pure-PyTorch equivalent in `description`.
- Strip all docstrings and comments from reference — move documentation to `definition.description`
- The `description` field should contain the complete operator documentation: math formulas (LaTeX), parameter descriptions, return value notes, edge cases

### Output naming
- Output name must not clash with any input name
- Default: use `"y"`, `"output"`, or a single letter not used by inputs
- Output shape derived from input shapes based on operator semantics
- Output dtype: same as input for elementwise; `int64` for index-producing ops; `bool` for predicates

## Step 4: Write `workload.jsonl`

One JSON object per line. No trailing newline at end of file.

### NPU style (default)

`axes` is `null`. Every `RandomInput` carries concrete `shape` and `dtype`.

```jsonl
{"uuid": "OpName-001", "axes": null, "inputs": {"x": {"type": "random", "shape": [128, 256], "dtype": "float16"}, "alpha": {"type": "scalar", "value": 1.0}}}
{"uuid": "OpName-002", "axes": null, "inputs": {"x": {"type": "random", "shape": [512, 1024], "dtype": "float32"}, "alpha": {"type": "scalar", "value": 2.0}}}
```

### Community style

`axes` resolves symbolic axis values. `RandomInput` omits shape/dtype (from definition).

```jsonl
{"uuid": "OpName-001", "axes": {"M": 64, "K": 128, "N": 64}, "inputs": {"a": {"type": "random"}, "b": {"type": "random"}}}
```

### UUID

Format: `"{OperatorName}-{line_number:03d}"` (1-indexed). Must be the **first field** in every workload. Must be unique across all lines.

### Input types reference

| InputSpec type | When to use | JSON shape |
|---|---|---|
| `RandomInput` | Regular tensor, random data | `{"type": "random", "shape": [M, N], "dtype": "float16"}` |
| `ScalarInput` | Non-tensor parameter | `{"type": "scalar", "value": 1.0}` |
| `CustomInput` | Custom generation from entrypoint | `{"type": "custom"}` |
| `SafetensorsInput` | Load from file | `{"type": "safetensors", "path": "...", "tensor_key": "..."}` |
| `TensorListInput` | Variable-length tensor list | `{"type": "tensor_list", "dtype": "float32", "shapes": [[128], [128]]}` |

**RandomInput fields (NPU style):**

| Field | Required | Description |
|---|---|---|
| `shape` | Yes | List of ints, concrete tensor shape |
| `dtype` | Yes | String dtype (e.g. `"float16"`) |
| `range` | No | `[low, high)` for integer tensor generation (exclusive upper, matches `torch.randint`) |

## Step 5: Write `solution.json` (if operator code provided)

Wrap the user's optimized kernel code as a Solution.

### Solution source file rules (MANDATORY)

**`sources[]` must contain ONLY:**
1. **Kernel source code** — the actual kernel implementation (e.g., `.py` Triton kernel, `.cpp` Ascend C kernel, `.h` header files)
2. **`main.py`** — the Python entry point that imports/calls the kernel and exposes `def run(...)`

**`sources[]` must NOT contain:**
- Build scripts, Makefiles, CMakeLists.txt, setup.py, pyproject.toml, or any compilation/building configuration files
- Custom Python packages or modules with `__init__.py` (no package-style directory structures)
- Test files, examples, or documentation
- Any file not directly needed to run the kernel at evaluation time

**`main.py` requirements:**
- **File name must be exactly `main.py`** — no other names allowed
- **Entry function must be `def run(...)`** — the function that evaluates the kernel
- **No custom package imports** — do NOT use relative imports (`from . import ...`), do NOT create package directories (`__init__.py`), and do NOT assume any project-specific import paths. All imports must be from:
  - Standard library packages
  - Installed third-party packages (torch, torch_npu, triton, triton_ascend, etc.)
  - Other source files listed in `sources[]` using plain top-level module imports (e.g., `import kernel` or `from kernel import my_func` where `kernel.py` is also in `sources[]`)

### Determine BuildSpec

| Language | entry_point | Notes |
|---|---|---|
| `python` | `main.py::run` | Pure PyTorch/Python implementations |
| `triton` | `main.py::run` | Triton kernels |
| `tilelang` | `main.py::run` | TileLang DSL |
| `ascendc` | `main.py::run` | Ascend C kernels; may need `bisheng_cflags`, `ld_flags` |
| `catlass` | `main.py::run` | Catlass C++ templates |

- **`entry_point` is always `main.py::run`** — non-negotiable for all languages
- `target_hardware`: default `["ascend910b"]`; ask user if uncertain
- `destination_passing_style`: `true` if the kernel writes output into a pre-allocated buffer; `false` if it returns a new tensor
- For Ascend C kernels, collect `bisheng_cflags` and `ld_flags` if needed (these go in `spec`, NOT as build files in `sources[]`)

### Solution structure

```json
{
  "name": "operator_name_language",
  "definition": "OperatorName",
  "author": "user",
  "spec": {
    "languages": ["triton"],
    "target_hardware": ["ascend910b"],
    "entry_point": "main.py::run",
    "destination_passing_style": false
  },
  "sources": [
    {
      "path": "main.py",
      "content": "import torch\nimport triton\n\ndef run(x):\n    ...\n"
    }
  ],
  "description": "Optional description of the solution approach."
}
```

- `name`: typically `"{definition_name}_{language}"` (e.g., `"matmul_triton"`)
- `definition`: must match the `name` in `definition.json` exactly
- **Python entry file must always be named `main.py`** — no exceptions
- **Entry function must always be `run`** — `entry_point` is always `main.py::run`
- Each source file goes in `sources[]` with its relative `path` and full `content`
- Multi-file solutions: add one `SourceFile` per file (e.g., `kernel.py` for kernel logic + `main.py` for the `run` wrapper); use plain `import kernel` in `main.py`, NOT package-relative imports
- For Ascend C multi-file: include `.cpp`/`.h` kernel files and `main.py` Python wrapper; do NOT include build scripts or CMake files

## Step 6: Validate

After writing all files, validate with the data models:

```python
from npu_kernelbench.data import Definition, Workload, Solution
import json
from pathlib import Path

# Validate definition
Definition(**json.loads(Path("definition.json").read_text()))

# Validate each workload
for line in Path("workload.jsonl").read_text().splitlines():
    if line.strip():
        Workload(**json.loads(line))

# Validate solution (if present)
if Path("solution.json").exists():
    Solution(**json.loads(Path("solution.json").read_text()))
```

### Pre-validation checklist

**NPU style:**
- `TensorSpec.shape` and `TensorSpec.dtype` are omitted/null in definition; capability in `description`
- `axes` is `{}` in definition and `null` in every workload
- Every `RandomInput` in every workload carries both `shape` (list of ints) and `dtype` (string)
- Integer `RandomInput` tensors carry `range: [low, high)` for valid indices
- Reference code compiles as valid Python with a `run` function
- `uuid` is unique across all workloads and appears as the first field
- `definition.description` contains complete operator documentation (not an abbreviated summary)

**Community style:**
- Every axis in input/output shapes exists in `axes`
- Every `var` axis appears in every workload's `axes` dict
- `uuid` is unique across all workloads
- Reference parses as valid Python with a `run` function

**Solution:**
- Python entry file is **named exactly `main.py`** (not `solution.py`, `run.py`, etc.)
- Entry function is **named exactly `run`** — `entry_point` is `main.py::run`
- `main.py` exists in `sources[]` and contains a `def run(...)` function
- `definition` name matches the `name` in `definition.json`
- `spec.languages` matches the actual content of the sources
- No duplicate source file paths
- **No build/compilation files** in `sources[]` (no Makefiles, CMakeLists.txt, setup.py, etc.)
- **No custom package imports** in `main.py` (no `from . import`, no `__init__.py`-based packages)
- Multi-file imports use plain top-level module imports (e.g., `import kernel` for `kernel.py`)

## Edge Cases

### tensor_list inputs (Cat, Stack)
Use `TensorListInput`:

```jsonl
{"tensors": {"type": "tensor_list", "dtype": "float32", "shapes": [[128], [256], [512]]}}
```

The reference `run()` receives a Python `list[Tensor]`.

### Tuple outputs (Split, AdamW)
Return a tuple from `run()`. The evaluator auto-maps to `{"0": tensor0, "1": tensor1, ...}`.

```python
def run(x, split_size_or_sections, dim=0):
    return torch.split(x, split_size_or_sections, dim=dim)
```

### In-place ops (IndexPut, Scatter)
Clone the input before in-place modification:

```python
def run(x, dim, index, src, reduce=None):
    x = x.clone()
    x.scatter_(dim, index, src, reduce=reduce)
    return x
```

### Integer index tensors (Index, Gather, Scatter)
Index tensors are `int64` with a `range` constraint. Use exclusive upper bound `[0, N)` (matching `torch.randint` convention):

```jsonl
{"index": {"type": "random", "shape": [64], "dtype": "int64", "range": [0, 128]}}
```

### Optional tensors (weight/bias in LayerNorm, GroupNorm)
Include in `definition.inputs` with a description noting they're optional. In workloads where absent, **omit the key entirely** from the inputs dict. In reference `run()`, make it a default-`None` kwarg:

```jsonl
# With weight + bias (both present):
{"weight": {"type": "random", "shape": [4096], "dtype": "float16"}, "bias": {"type": "random", "shape": [4096], "dtype": "float16"}}
# Weight only (bias omitted):
{"weight": {"type": "random", "shape": [4096], "dtype": "float16"}}
```

### Vendor-specific ops (torch_npu.*)
**Do NOT replace vendor APIs.** Keep `torch_npu` calls in reference code as-is. Write the pure-PyTorch equivalent in `description` for documentation purposes.

### Multi-output ops (Sort, TopK)
These return `(values, indices)` tuples. If only evaluating one component, extract it:

```python
def run(x, dim=-1, descending=False):
    return torch.sort(x, dim=dim, descending=descending)[0]
```

### Boolean output ops (Nonzero, comparison ops)
Set output dtype to `"bool"` or `"int64"` (for indices).

### Variable-rank operators (Community style only)
When test cases span multiple ranks (1D, 2D, 3D, 4D) for Community style:
- **Preferred:** Group by rank into separate directories (`Op_2d/`, `Op_3d/`)
- **Alternative (elementwise only):** Pad to max rank with `1`s for unused axes — safe because elementwise ops compute identically on any shape
- **NPU style handles this naturally** — each workload carries its own concrete shape list

### ScalarInput with list/tuple values
Use native Python lists, not JSON strings:

```jsonl
{"dim": {"type": "scalar", "value": [4096]}}
```
