# Workload（评测用例）

## 概述

`Workload` 是 [Definition](definition.md) 的具体化实例：将变量维度绑定到具体整数值，并指定每个输入的来源策略。

一个 Definition 通常对应多个 Workload，覆盖不同尺寸和数据类型。Workload 按 JSONL 格式存储，每行一个 Workload 对象。

## JSON Schema

### 顶层结构

| 字段 | 类型 | 必填 | 说明 |
|------|------|------|------|
| `uuid` | string | 是 | 唯一标识，如 `"mm-001"` |
| `axes` | object | 是 | 变量维度到整数值的映射 |
| `inputs` | object | 是 | 输入数据来源描述 |
| `tolerance` | object | 否 | 正确性容差配置 |

### axes：维度映射

将 Definition 中的 `var` 类型维度绑定到具体值：

```json
"axes": {
  "M": 1024,
  "K": 4096,
  "N": 1024
}
```

### inputs：输入描述

每个输入通过 `type` 字段指定来源类型。

#### random — 随机生成

| 字段 | 类型 | 必填 | 说明 |
|------|------|------|------|
| `type` | string | 是 | 固定值 `"random"` |
| `dtype` | string | 否 | 覆盖 Definition 中的默认 dtype |

```json
"inputs": {
  "a": {"type": "random"},
  "b": {"type": "random", "dtype": "float16"}
}
```

#### scalar — 标量字面值

| 字段 | 类型 | 必填 | 说明 |
|------|------|------|------|
| `type` | string | 是 | 固定值 `"scalar"` |
| `value` | int / float / bool / string | 是 | 标量值 |

```json
"inputs": {
  "alpha": {"type": "scalar", "value": 2.5},
  "mode": {"type": "scalar", "value": "tanh"}
}
```

#### custom — 自定义生成

| 字段 | 类型 | 必填 | 说明 |
|------|------|------|------|
| `type` | string | 是 | 固定值 `"custom"` |

通过 Definition 中的 `custom_inputs_entrypoint` 函数生成。Workload 中所有输入要么全是 custom，要么全不是（不允许混合）。

#### safetensors — 文件加载

| 字段 | 类型 | 必填 | 说明 |
|------|------|------|------|
| `type` | string | 是 | 固定值 `"safetensors"` |
| `path` | string | 是 | safetensors 文件路径 |
| `tensor_key` | string | 是 | 文件中的张量 key |

### tolerance：容差配置

| 字段 | 类型 | 默认值 | 说明 |
|------|------|--------|------|
| `max_atol` | float | 0.01 | 最大绝对误差 |
| `max_rtol` | float | 0.01 | 最大相对误差 |
| `required_matched_ratio` | float | 0.99 | 通过元素比例 |
| `max_error_cap` | float | null | 绝对误差硬上限 |
| `allow_negative_inf` | bool | false | 是否允许匹配 -inf |

## 完整示例

### 多 dtype GEMM 用例（JSONL 格式）

```jsonl
{"axes": {"M": 64, "K": 128, "N": 64}, "inputs": {"a": {"type": "random", "dtype": "float32"}, "b": {"type": "random", "dtype": "float32"}}, "uuid": "mm-001"}
{"axes": {"M": 128, "K": 256, "N": 128}, "inputs": {"a": {"type": "random", "dtype": "float16"}, "b": {"type": "random", "dtype": "float16"}}, "uuid": "mm-002"}
{"axes": {"M": 128, "K": 256, "N": 128}, "inputs": {"a": {"type": "random", "dtype": "bfloat16"}, "b": {"type": "random", "dtype": "bfloat16"}}, "uuid": "mm-003"}
```

### 带标量的用例

```jsonl
{"axes": {"N": 8}, "inputs": {"x": {"type": "random"}, "alpha": {"type": "scalar", "value": 2.5}}, "uuid": "scale-001"}
```

### 带容差的用例

```json
{
  "axes": {"N": 1024},
  "inputs": {"a": {"type": "random"}, "b": {"type": "random"}},
  "uuid": "vecadd-001",
  "tolerance": {
    "max_atol": 0.001,
    "max_rtol": 0.001,
    "required_matched_ratio": 0.999
  }
}
```
