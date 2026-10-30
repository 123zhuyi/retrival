# IsNegInf

## 产品支持情况

| 产品 | 是否支持 |
| :-- | :--: |
| Atlas A2 训练系列产品 / Atlas A2 推理系列产品 | √ |
| Atlas A3 训练系列产品 / Atlas A3 推理系列产品 | √ |

当前工程按需求仅面向 `Ascend 910B/910C` 路径生成 ACLNN 接口与 AiCore 实现。

## 功能说明

- 算子功能：逐元素判断输入张量是否为负无穷。
- 对浮点输入，返回 `input_i == -inf` 的布尔结果。
- 对 `bool/int8/uint8/int16/int32/int64` 等有界类型，返回同 shape 的全 `false` 布尔张量。

## 数学表达

若输入为浮点型：

$$
out_i = (self_i == -\infty)
$$

若输入为支持的非浮点有界类型：

$$
out_i = \mathrm{false}
$$

## 接口说明

对外暴露两段式 ACLNN 接口：

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

| 参数名 | 输入/输出 | 描述 | 数据类型 | 数据格式 |
| :-- | :-- | :-- | :-- | :-- |
| `self` | 输入 | 待判断是否为负无穷的张量 | `FLOAT16/FLOAT/BFLOAT16/BOOL/INT8/UINT8/INT16/INT32/INT64` | `ND` |
| `out` | 输出 | 逐元素判断结果 | `BOOL` | `ND` |

## 约束说明

- `self` 和 `out` 不能为空指针。
- `self` 与 `out` shape 必须完全一致，不支持 broadcast。
- 维度数不超过 8，支持标量和空 Tensor。
- `out` dtype 必须为 `BOOL`。
- `self` 非连续时先转连续；`out` 非连续时通过 `ViewCopy` 回写。
- `BFLOAT16` 浮点路径依赖 `910B/910C` 类 SoC 支持。

## 调用说明

| 调用方式 | 调用样例 | 说明 |
| :-- | :-- | :-- |
| ACLNN | [examples/test_aclnn_is_neg_inf.cpp](./examples/test_aclnn_is_neg_inf.cpp) | 两段式调用 `aclnnIsNegInfGetWorkspaceSize` 和 `aclnnIsNegInf` |
