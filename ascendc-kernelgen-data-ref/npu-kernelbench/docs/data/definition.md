# Definition（算子定义）

## 概述

`Definition` 是算子的正式规格说明，定义了张量格式、维度语义和参考实现。它是算子开发和优化的唯一数据源。

Definition 不包含具体的输入数据，具体数据由 [Workload](workload.md) 提供。

## JSON Schema

### 顶层结构

| 字段 | 类型 | 必填 | 说明 |
|------|------|------|------|
| `name` | string | 是 | 唯一名称，如 `"matmul_triton"`、`"gelu"` |
| `op_type` | string | 否 | 计算类别，如 `"matmul"`、`"elementwise"` |
| `description` | string | 否 | 可读描述 |
| `axes` | object | 是 | 符号化维度定义，key 为维度名 |
| `inputs` | object | 是 | 输入张量定义 |
| `outputs` | object | 是 | 输出张量定义 |
| `reference` | string | 是 | 参考实现代码，必须包含 `run` 函数 |
| `custom_inputs_entrypoint` | string | 否 | 自定义输入生成函数名 |

### axes：维度定义

支持三种类型：

#### `const` — 常量维度

| 字段 | 类型 | 必填 | 说明 |
|------|------|------|------|
| `type` | string | 是 | 固定值 `"const"` |
| `value` | integer | 是 | 常量值 |
| `description` | string | 否 | 描述 |

```json
"hidden_size": {
  "type": "const",
  "value": 2048
}
```

#### `var` — 变量维度

运行时由 Workload 确定。

| 字段 | 类型 | 必填 | 说明 |
|------|------|------|------|
| `type` | string | 是 | 固定值 `"var"` |
| `description` | string | 否 | 描述 |

```json
"M": {
  "type": "var",
  "description": "A 和 C 的行数"
}
```

#### `expr` — 表达式维度

由其他维度通过算术表达式计算得出。

| 字段 | 类型 | 必填 | 说明 |
|------|------|------|------|
| `type` | string | 是 | 固定值 `"expr"` |
| `expression` | string | 是 | 算术表达式，如 `"M * 2"` |

支持运算符：`+`、`-`、`*`、`/`、`//`、`%`、`**`。

### inputs / outputs：张量定义

| 字段 | 类型 | 必填 | 说明 |
|------|------|------|------|
| `shape` | array 或 null | 是 | 维度名数组。`null` 表示标量 |
| `dtype` | string | 是 | 数据类型 |
| `description` | string | 否 | 描述 |

### dtype：支持的数据类型

- `float64`
- `float32`
- `float16`
- `bfloat16`
- `float8_e4m3fn`
- `float8_e5m2`
- `float4_e2m1fn_x2`
- `int64`
- `int32`
- `int16`
- `int8`
- `bool`

### reference：参考实现

- 必须包含名为 `run` 的顶层函数作为入口
- 应为纯 PyTorch 实现，作为数学规格的正式定义
- 优先使用显式步骤而非高层封装

### custom_inputs_entrypoint：自定义输入

可选的字符串，指定 `reference` 代码中用于生成自定义输入的函数名。函数签名为 `(axes, device) -> dict[str, Tensor]`。Workload 中使用 `{"type": "custom"}` 时触发。

## 完整示例

### GEMM

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

### 带标量输入的 LayerNorm

```json
{
  "name": "layernorm",
  "op_type": "norm",
  "axes": {
    "N": {"type": "var", "description": "batch size"},
    "D": {"type": "var", "description": "hidden dim"}
  },
  "inputs": {
    "x": {"shape": ["N", "D"], "dtype": "float16"},
    "weight": {"shape": ["D"], "dtype": "float16"},
    "eps": {"shape": null, "dtype": "float32"}
  },
  "outputs": {
    "y": {"shape": ["N", "D"], "dtype": "float16"}
  },
  "reference": "import torch\ndef run(x, weight, eps):\n    mean = x.mean(-1, keepdim=True)\n    var = x.var(-1, keepdim=True, unbiased=False)\n    return weight * (x - mean) / torch.sqrt(var + eps)\n"
}
```

### 带表达式维度的算子

```json
{
  "name": "attention_matmul",
  "op_type": "attention",
  "axes": {
    "batch": {"type": "var"},
    "seq_len": {"type": "var"},
    "num_heads": {"type": "const", "value": 32},
    "head_dim": {"type": "const", "value": 128},
    "total_dim": {"type": "expr", "expression": "num_heads * head_dim"}
  },
  "inputs": {
    "q": {"shape": ["batch", "seq_len", "total_dim"], "dtype": "float16"},
    "k": {"shape": ["batch", "seq_len", "total_dim"], "dtype": "float16"}
  },
  "outputs": {
    "scores": {"shape": ["batch", "seq_len", "seq_len"], "dtype": "float32"}
  },
  "reference": "import torch\ndef run(q, k):\n    return torch.matmul(q, k.transpose(-2, -1)) / (128 ** 0.5)\n"
}
```
