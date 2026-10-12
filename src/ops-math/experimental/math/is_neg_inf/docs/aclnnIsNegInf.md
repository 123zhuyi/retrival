# aclnnIsNegInf

## 功能说明

`aclnnIsNegInf` 判断输入张量每个元素是否为负无穷，并将结果写入 `BOOL` 类型输出张量。

## 函数原型

```cpp
aclnnStatus aclnnIsNegInfGetWorkspaceSize(
    const aclTensor* self,
    aclTensor* out,
    uint64_t* workspaceSize,
    aclOpExecutor** executor);
```

```cpp
aclnnStatus aclnnIsNegInf(
    void* workspace,
    uint64_t workspaceSize,
    aclOpExecutor* executor,
    const aclrtStream stream);
```

## 参数说明

| 参数名 | 输入/输出 | 描述 | 使用说明 | 数据类型 | 数据格式 | 维度(shape) | 非连续 Tensor |
| :-- | :-- | :-- | :-- | :-- | :-- | :-- | :--: |
| `self` | 输入 | 待判断张量 | 不支持 broadcast | `FLOAT16/FLOAT/BFLOAT16/BOOL/INT8/UINT8/INT16/INT32/INT64` | `ND` | `0-8` 维 | √ |
| `out` | 输出 | 判断结果张量 | shape 与 `self` 完全一致，dtype 必须为 `BOOL` | `BOOL` | `ND` | 与 `self` 相同 | √ |
| `workspaceSize` | 输出 | 设备侧 workspace 大小 | 第一段接口返回 | - | - | - | - |
| `executor` | 输出 | 执行器 | 第一段接口返回 | - | - | - | - |

## 返回值

第一段接口完成参数检查并构造执行流程。典型报错场景如下：

| 返回码 | 错误码 | 描述 |
| :-- | :-- | :-- |
| `ACLNN_ERR_PARAM_NULLPTR` | `161001` | `self` 或 `out` 是空指针 |
| `ACLNN_ERR_PARAM_INVALID` | `161002` | `self/out` dtype 不支持 |
| `ACLNN_ERR_PARAM_INVALID` | `161002` | `out` dtype 不是 `BOOL` |
| `ACLNN_ERR_PARAM_INVALID` | `161002` | `self/out` shape 不一致 |
| `ACLNN_ERR_PARAM_INVALID` | `161002` | 维度数超过 8 |
| `ACLNN_ERR_PARAM_INVALID` | `161002` | 当前设备不在 `910B/910C` 支持范围 |

## 计算流程

- 浮点输入：
  - `Contiguous(self)`
  - `l0op::IsNegInf`
  - `ViewCopy(result, out)`
- 非浮点有界输入：
  - 直接构造全 `false` 结果
  - `ViewCopy(result, out)`

## 约束说明

- 算子为确定性实现。
- 空 Tensor 直接返回成功，`workspaceSize = 0`。
- `BFLOAT16` 仅在 `910B/910C` 浮点路径支持。

## 调用示例

参考 [examples/test_aclnn_is_neg_inf.cpp](../examples/test_aclnn_is_neg_inf.cpp)。
