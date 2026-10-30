# solution.json Format Reference

`solution.json` is the canonical packaging format for a kernel implementation in npu-kernelbench. It is a JSON serialization of the `Solution` Pydantic model (`src/npu_kernelbench/data/solution.py`).

## Top-level schema

```json
{
  "name": "my_matmul_triton",
  "definition": "matmul_triton",
  "author": "developer",
  "spec": {
    "languages": ["triton"],
    "target_hardware": ["ascend910b"],
    "entry_point": "main.py::run",
    "destination_passing_style": true,
    "dependencies": [],
    "build_method": null,
    "bisheng_cflags": [],
    "ld_flags": []
  },
  "sources": [
    {
      "path": "main.py",
      "content": "import torch\n..."
    }
  ],
  "declared_kernel": null,
  "description": "Optional human-readable summary of the approach."
}
```

## Field reference

| Field | Required | Type | Description |
|-------|----------|------|-------------|
| `name` | Yes | string | Unique solution name. Prefer `{definition_name}_{language}`. |
| `definition` | Yes | string | Must exactly match the `name` field of the target `definition.json`. |
| `author` | Yes | string | Author or agent name. |
| `spec` | Yes | object | `BuildSpec` — see below. |
| `sources` | Yes | list | One `SourceFile` per file needed to run the solution. |
| `declared_kernel` | No | string | Optional declared kernel name for anti-hack signature verification. |
| `description` | No | string | Free-form notes about the approach. |

## BuildSpec

| Field | Required | Type | Description |
|-------|----------|------|-------------|
| `languages` | Yes | list of strings | One or more of: `python`, `pytorch`, `triton`, `tilelang`, `ascendc`, `catlass`. |
| `target_hardware` | Yes | list of strings | One or more of: `ascend910b`, `ascend910_93`, `ascend950`, `LOCAL`. |
| `entry_point` | Yes | string | Format `file_path::function_name`. For npu-kernelbench this is **always** `main.py::run`. |
| `destination_passing_style` | No | bool | Default `true`. Set `true` if `run()` writes into a pre-allocated output buffer; `false` if it returns a fresh tensor. |
| `dependencies` | No | list of strings | Optional Python packages or libraries required by the solution. |
| `build_method` | No | string | AscendC only: `bisheng` (default) or `cmake`. |
| `bisheng_cflags` | No | list of strings | Extra compiler flags for bisheng. |
| `ld_flags` | No | list of strings | Extra linker flags. |

## SourceFile

| Field | Required | Type | Description |
|-------|----------|------|-------------|
| `path` | Yes | string | Relative path inside the staging directory. Must not be absolute and must not contain `..`. |
| `content` | Yes | string | Full source text of the file. |

## Hard rules

1. **`entry_point` is always `main.py::run`.** No exceptions.
2. **The file `main.py` must exist in `sources` and must define `def run(...)`.**
3. **`sources` must not contain** build scripts, `CMakeLists.txt`, `Makefile`, `setup.py`, `pyproject.toml`, or any `__init__.py`.
4. **No package-relative imports** (`from . import ...`). Use plain top-level imports such as `import kernel` when `kernel.py` is also in `sources`.
5. **Source paths must be relative** and must not traverse above the solution root (`..` is forbidden).

## Per-language examples

### Pure PyTorch / `python`

```json
{
  "name": "gelu_python",
  "definition": "gelu",
  "author": "example",
  "spec": {
    "languages": ["python"],
    "target_hardware": ["ascend910b"],
    "entry_point": "main.py::run",
    "destination_passing_style": false
  },
  "sources": [
    {
      "path": "main.py",
      "content": "import torch\n\ndef run(x):\n    return torch.nn.functional.gelu(x)\n"
    }
  ]
}
```

### Triton — single file

```json
{
  "name": "gelu_triton",
  "definition": "gelu",
  "author": "example",
  "spec": {
    "languages": ["triton"],
    "target_hardware": ["ascend910b"],
    "entry_point": "main.py::run",
    "destination_passing_style": false
  },
  "sources": [
    {
      "path": "main.py",
      "content": "import torch\nimport triton\nimport triton.language as tl\n\n@triton.jit\ndef gelu_kernel(x_ptr, out_ptr, n, BLOCK_SIZE: tl.constexpr):\n    pid = tl.program_id(0)\n    offsets = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)\n    mask = offsets < n\n    x = tl.load(x_ptr + offsets, mask=mask)\n    y = 0.5 * x * (1 + tl.math.erf(x / tl.sqrt(2.0)))\n    tl.store(out_ptr + offsets, y, mask=mask)\n\ndef run(x):\n    out = torch.empty_like(x)\n    n = x.numel()\n    gelu_kernel[(triton.cdiv(n, 1024),)](x, out, n, BLOCK_SIZE=1024)\n    return out\n"
    }
  ]
}
```

### Triton — multi-file

`kernel.py` holds the Triton kernel; `main.py` imports it and exposes `run`.

```json
{
  "name": "matmul_triton",
  "definition": "matmul_triton",
  "author": "example",
  "spec": {
    "languages": ["triton"],
    "target_hardware": ["ascend910b"],
    "entry_point": "main.py::run",
    "destination_passing_style": false
  },
  "sources": [
    {
      "path": "kernel.py",
      "content": "import triton\nimport triton.language as tl\n\n@triton.jit\ndef matmul_kernel(...):\n    ...\n"
    },
    {
      "path": "main.py",
      "content": "import torch\nimport kernel\n\ndef run(a, b):\n    return kernel.matmul(a, b)\n"
    }
  ],
  "declared_kernel": "matmul_kernel"
}
```

### TileLang

TileLang is handled by the Python builder, so the layout is the same as Triton.

```json
{
  "name": "softmax_tilelang",
  "definition": "softmax",
  "author": "example",
  "spec": {
    "languages": ["tilelang"],
    "target_hardware": ["ascend910b"],
    "entry_point": "main.py::run",
    "destination_passing_style": false
  },
  "sources": [
    {
      "path": "main.py",
      "content": "import torch\nimport tilelang\n\ndef run(x):\n    ...\n"
    }
  ]
}
```

### AscendC — bisheng (default)

Include the C++ kernel/host files and a Python `main.py` wrapper. Build flags go in `spec`, not in `sources`.

```json
{
  "name": "vector_add_ascendc",
  "definition": "vector_add",
  "author": "example",
  "spec": {
    "languages": ["ascendc"],
    "target_hardware": ["ascend910b"],
    "entry_point": "main.py::run",
    "destination_passing_style": true,
    "build_method": "bisheng",
    "bisheng_cflags": ["-O2"],
    "ld_flags": []
  },
  "sources": [
    {
      "path": "op_kernel/add_custom.cpp",
      "content": "..."
    },
    {
      "path": "op_host/my_add.cpp",
      "content": "..."
    },
    {
      "path": "main.py",
      "content": "import torch\nimport torch_npu\n\ndef run(a, b, out):\n    torch.ops.vector_add_ascendc.my_add(a, b, out)\n"
    }
  ]
}
```

### AscendC — cmake

Same file layout; only `build_method` changes.

```json
{
  "name": "vector_add_ascendc_cmake",
  "definition": "vector_add",
  "author": "example",
  "spec": {
    "languages": ["ascendc"],
    "target_hardware": ["ascend910b"],
    "entry_point": "main.py::run",
    "destination_passing_style": true,
    "build_method": "cmake"
  },
  "sources": [
    {
      "path": "op_kernel/add_custom.cpp",
      "content": "..."
    },
    {
      "path": "op_host/my_add.cpp",
      "content": "..."
    },
    {
      "path": "main.py",
      "content": "..."
    }
  ]
}
```

### Catlass

Catlass is built as AscendC under the hood, so the spec is nearly identical to AscendC.

```json
{
  "name": "matmul_catlass",
  "definition": "matmul",
  "author": "example",
  "spec": {
    "languages": ["catlass"],
    "target_hardware": ["ascend910b"],
    "entry_point": "main.py::run",
    "destination_passing_style": true,
    "build_method": "bisheng"
  },
  "sources": [
    {
      "path": "main.py",
      "content": "..."
    }
  ]
}
```

## Validation

Validate a `solution.json` against the data model:

```bash
uv run python -c "
from npu_kernelbench.data import Solution
import json
Solution(**json.loads(open('solution.json').read()))
print('OK')
"
```

Or use the skill helper:

```bash
uv run python .claude/skills/npu-kernelbench/scripts/validate_solution.py solution.json
```
