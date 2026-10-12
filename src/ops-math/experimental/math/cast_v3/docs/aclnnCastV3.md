# aclnnCastV3

## 产品支持情况

| 产品 | 是否支持 |
| ---- | :----: |
| Atlas 310P 推理系列产品 | √ |

## 功能说明

- 接口功能：对输入张量进行数据类型转换，输出与输入形状相同、数据类型为目标类型的张量。本算子为面向 Ascend 310P 优化的 Cast v3 版本实现。
- 当前算子提供的 ACLNN 接口为 `aclnnCastV3GetWorkspaceSize` 和 `aclnnCastV3` 两段式接口。

## 函数原型

每个算子分为两段式接口，必须先调用 `aclnnCastV3GetWorkspaceSize` 获取计算所需 workspace 大小以及执行器，再调用 `aclnnCastV3` 执行计算。

```Cpp
aclnnStatus aclnnCastV3GetWorkspaceSize(
    const aclTensor* x,
    int64_t dstType,
    const aclTensor* y,
    uint64_t* workspaceSize,
    aclOpExecutor** executor);
```

```Cpp
aclnnStatus aclnnCastV3(
    void* workspace,
    uint64_t workspaceSize,
    aclOpExecutor* executor,
    aclrtStream stream);
```

## aclnnCastV3GetWorkspaceSize

- 参数说明

| 参数名 | 输入/输出 | 描述 | 使用说明 | 数据类型 | 数据格式 | 维度(shape) | 非连续Tensor |
| ---- | ---- | ---- | ---- | ---- | ---- | ---- | ---- |
| `x` | 输入 | 输入张量。 | 支持 0-8 维。 | FLOAT、FLOAT16、INT8、INT32、INT16、UINT8、BOOL、BF16、INT64 | ND | 0-8 维 | √ |
| `dstType` | 输入 | 目标数据类型枚举值。 | 指定输出张量的数据类型。 | INT64 | - | - | - |
| `y` | 输出 | 输出张量。 | shape 与 x 相同。 | 与目标类型一致 | ND | 0-8 维 | - |
| `workspaceSize` | 输出 | 返回需要在 Device 侧申请的 workspace 大小。 | - | - | - | - | - |
| `executor` | 输出 | 返回算子执行器。 | 包含算子计算流程。 | - | - | - | - |

- 返回值

  `aclnnStatus`：返回状态码，具体参见 aclnn 返回码。

## aclnnCastV3

- 参数说明

| 参数名 | 输入/输出 | 描述 |
| ---- | ---- | ---- |
| `workspace` | 输入 | Device 侧申请的 workspace 内存地址。 |
| `workspaceSize` | 输入 | Device 侧申请的 workspace 大小，由 `aclnnCastV3GetWorkspaceSize` 获取。 |
| `executor` | 输入 | 算子执行器，包含算子计算流程。 |
| `stream` | 输入 | 指定执行任务的 Stream。 |

- 返回值

  `aclnnStatus`：返回状态码，具体参见 aclnn 返回码。

## 约束说明

- 当前仅支持 ascend310p 芯片。
- 输入输出格式仅支持 ND。
- 支持非连续 Tensor。
- 支持 53 种输入输出类型组合，详见 README.md。
- `dst_type` 属性必须指定有效的目标数据类型枚举值。
- 输出张量 y 的数据类型必须与 `dstType` 指定的类型一致。

## 调用说明

| 调用方式 | 调用样例 | 说明 |
| ---- | ---- | ---- |
| ACLNN 调用 | [test_aclnn_cast_v3.cpp](../examples/test_aclnn_cast_v3.cpp) | 通过 `aclnnCastV3GetWorkspaceSize` 和 `aclnnCastV3` 两段式接口调用 CastV3 算子。 |
