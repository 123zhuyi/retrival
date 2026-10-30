# Trace（评测结果）

## 概述

`Trace` 记录一次评测的完整结果：某个 [Solution](solution.md) 在特定 [Definition](definition.md) 下、使用特定 [Workload](workload.md) 的正确性和性能数据。

Trace 是评测数据库的原子记录单元。所有 Trace 汇总后可进行排序、筛选、评分。

## JSON Schema

### 顶层结构

| 字段 | 类型 | 必填 | 说明 |
|------|------|------|------|
| `definition` | string | 是 | Definition 名称 |
| `workload` | object | 是 | Workload 对象（内联） |
| `solution` | string | 否 | Solution 名称（仅 workload trace 时为 null） |
| `evaluation` | object | 否 | 评测结果（仅 workload trace 时为 null） |

### evaluation：评测结果

| 字段 | 类型 | 必填 | 说明 |
|------|------|------|------|
| `status` | string | 是 | 状态码，见下表 |
| `environment` | object | 是 | 硬件和软件环境信息 |
| `timestamp` | string | 是 | ISO 8601 时间戳 |
| `log` | string | 是 | 评测过程的 stdout/stderr |
| `correctness` | object | 条件必填 | 正确性指标 |
| `performance` | object | 条件必填 | 性能指标 |

### status：状态码

| 状态 | 说明 |
|------|------|
| `PASSED` | 通过 |
| `INCORRECT_SHAPE` | 输出形状不匹配 |
| `INCORRECT_DTYPE` | 输出数据类型不匹配 |
| `INCORRECT_NUMERICAL` | 数值误差超出容差 |
| `RUNTIME_ERROR` | 运行时异常 |
| `COMPILE_ERROR` | 编译失败 |
| `TIMEOUT` | 超时 |
| `REWARD_HACK` | 反作弊检测触发 |
| `INVALID_REFERENCE` | 参考实现无效 |

### correctness：正确性指标

| 字段 | 类型 | 说明 |
|------|------|------|
| `max_absolute_error` | float | 最大绝对误差 |
| `max_relative_error` | float | 最大相对误差 |
| `has_nan` | bool | 输出是否包含 NaN |
| `has_inf` | bool | 输出是否包含 Inf |
| `extra` | object | 额外指标（可选） |

### performance：性能指标

| 字段 | 类型 | 说明 |
|------|------|------|
| `latency_ms` | float | 方案执行延迟（毫秒） |
| `reference_latency_ms` | float | 参考实现延迟（毫秒） |
| `speedup_factor` | float | 加速比（reference / solution） |

### environment：环境信息

| 字段 | 类型 | 说明 |
|------|------|------|
| `hardware` | string | 硬件标识，如 `"Ascend910B3"` |
| `libs` | object | 库名到版本的映射 |

### status 与 correctness/performance 的关系

| status | correctness | performance |
|--------|-------------|-------------|
| PASSED | 必填 | 必填 |
| INCORRECT_NUMERICAL | 必填 | null |
| INCORRECT_SHAPE | null | null |
| INCORRECT_DTYPE | null | null |
| RUNTIME_ERROR | null | null |
| COMPILE_ERROR | null | null |
| TIMEOUT | null | null |

## 完整示例

### PASSED

```json
{
  "definition": "matmul_triton",
  "solution": "matmul_triton",
  "workload": {
    "uuid": "mm-001",
    "axes": {"M": 64, "K": 128, "N": 64},
    "inputs": {"a": {"type": "random", "dtype": "float32"}, "b": {"type": "random", "dtype": "float32"}},
    "tolerance": {"max_atol": 0.01, "max_rtol": 0.01, "required_matched_ratio": 0.99}
  },
  "evaluation": {
    "status": "PASSED",
    "environment": {
      "hardware": "Ascend910B3",
      "libs": {"torch": "2.6.0", "torch_npu": "2.6.0", "triton": "3.2.0"}
    },
    "timestamp": "2026-06-09T12:25:00+00:00",
    "log": "",
    "correctness": {
      "max_relative_error": 0.0,
      "max_absolute_error": 0.0,
      "has_nan": false,
      "has_inf": false
    },
    "performance": {
      "latency_ms": 0.002,
      "reference_latency_ms": 0.003,
      "speedup_factor": 1.50
    }
  }
}
```

### RUNTIME_ERROR

```json
{
  "definition": "matmul_triton",
  "solution": "matmul_triton",
  "workload": {
    "uuid": "mm-002",
    "axes": {"M": 128, "K": 256, "N": 128},
    "inputs": {"a": {"type": "random"}, "b": {"type": "random"}},
    "tolerance": {"max_atol": 0.01, "max_rtol": 0.01, "required_matched_ratio": 0.99}
  },
  "evaluation": {
    "status": "RUNTIME_ERROR",
    "environment": {"hardware": "Ascend910B3", "libs": {}},
    "timestamp": "2026-06-09T12:25:10+00:00",
    "log": "cbuf overflow, requires 6291456 bits while 4194304 bits available!",
    "correctness": null,
    "performance": null
  }
}
```

### Workload-only Trace

仅用于存储 Workload 数据的 Trace（solution 和 evaluation 均为 null）：

```json
{
  "definition": "matmul_triton",
  "workload": {
    "uuid": "mm-001",
    "axes": {"M": 64, "K": 128, "N": 64},
    "inputs": {"a": {"type": "random", "dtype": "float32"}, "b": {"type": "random", "dtype": "float32"}},
    "tolerance": {"max_atol": 0.01, "max_rtol": 0.01, "required_matched_ratio": 0.99}
  },
  "solution": null,
  "evaluation": null
}
```
