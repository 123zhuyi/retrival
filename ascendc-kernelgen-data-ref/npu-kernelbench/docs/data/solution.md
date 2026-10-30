# Solution（算子方案）

## 概述

`Solution` 是针对某个 [Definition](definition.md) 的具体实现。每个 Solution 包含完整的源码和构建元数据，用于编译、加载和评测。

## JSON Schema

### 顶层结构

| 字段 | 类型 | 必填 | 说明 |
|------|------|------|------|
| `name` | string | 是 | 唯一名称，如 `"matmul_triton"` |
| `definition` | string | 是 | 对应的 Definition 名称 |
| `author` | string | 是 | 作者或 Agent 名称 |
| `description` | string | 否 | 方案描述 |
| `spec` | object | 是 | 构建和运行规格 |
| `sources` | array | 是 | 源码文件列表 |

### sources：源码文件

| 字段 | 类型 | 必填 | 说明 |
|------|------|------|------|
| `path` | string | 是 | 相对路径，如 `"kernel.py"`、`"src/add_custom.asc"` |
| `content` | string | 是 | 文件完整内容 |

### spec：构建规格

| 字段 | 类型 | 必填 | 说明 |
|------|------|------|------|
| `languages` | array | 是 | 编程语言列表：`python`、`triton`、`tilelang`、`ascendc` |
| `target_hardware` | array | 是 | 目标硬件：`ascend910b`、`ascend910_93`、`ascend950`、`LOCAL` |
| `entry_point` | string | 是 | 入口函数，格式 `"{file}::{func}"`，如 `"kernel.py::run"` |
| `destination_passing_style` | bool | 否 | 是否使用 DPS 风格（默认 `true`） |
| `dependencies` | array | 否 | 依赖列表 |
| `bisheng_cflags` | array | 否 | bisheng 编译器额外参数（默认 `[]`） |
| `ld_flags` | array | 否 | 额外链接参数（默认 `[]`） |

### Destination Passing Style (DPS)

| 风格 | `destination_passing_style` | 函数签名 |
|------|----------------------------|----------|
| DPS | `true`（默认） | `run(input1, input2, ..., output1, output2, ...)` |
| 返回值 | `false` | `run(input1, input2, ...) -> output` |

## 示例

### Triton 方案

```json
{
  "name": "matmul_triton",
  "definition": "matmul_triton",
  "author": "example",
  "spec": {
    "languages": ["triton"],
    "target_hardware": ["ascend910b"],
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

### Ascend C 方案

```json
{
  "name": "vector_add_ascendc",
  "definition": "vector_add",
  "author": "example",
  "spec": {
    "languages": ["ascendc"],
    "target_hardware": ["ascend910b"],
    "entry_point": "main.py::run",
    "destination_passing_style": false,
  },
  "sources": [
    {
      "path": "add_custom.asc",
      "content": "#include <torch/extension.h>\n..."
    },
    {
      "path": "main.py",
      "content": "import torch\n...\ndef run(a, b):\n    ...\n"
    }
  ]
}
```

### TileLang 方案

```json
{
  "name": "identity_sol",
  "definition": "identity",
  "author": "test",
  "spec": {
    "languages": ["tilelang"],
    "target_hardware": ["ascend910b"],
    "entry_point": "main.py::run"
  },
  "sources": [
    {
      "path": "main.py",
      "content": "def run(x): return x + 0\n"
    }
  ]
}
```
