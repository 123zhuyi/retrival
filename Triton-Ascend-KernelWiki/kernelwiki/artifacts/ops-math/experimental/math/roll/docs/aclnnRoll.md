# aclnnRoll

## 产品支持情况

| 产品 | 是否支持 |
| :--- | :---: |
| Atlas A2 训练系列产品 | 支持 |

## 功能说明

- 接口功能：沿给定维度对输入 Tensor 执行循环位移。

- 语义说明：

  - `dims` 非空时，`shifts[i]` 作用于 `dims[i]` 对应维度。
  - `dims` 为空时，先对输入按逻辑视图展平，再执行一维 roll。
  - 重复维度会在 Host 侧归一化合并。

## 函数原型

```cpp
aclnnStatus aclnnRollGetWorkspaceSize(
    const aclTensor* x,
    const aclIntArray* shifts,
    const aclIntArray* dims,
    aclTensor* out,
    uint64_t* workspaceSize,
    aclOpExecutor** executor)
```

```cpp
aclnnStatus aclnnRoll(
    void* workspace,
    uint64_t workspaceSize,
    aclOpExecutor* executor,
    aclrtStream stream)
```

## aclnnRollGetWorkspaceSize

| 参数名 | 输入/输出 | 描述 | 数据类型 | 数据格式 | 维度 |
| --- | --- | --- | --- | --- | --- |
| x | 输入 | 输入张量 | uint8, int8, bfloat16, float16, float32, int32, uint32 | ND | 0-8 维 |
| shifts | 输入 | 每个目标维度上的循环位移量 | aclIntArray* | - | - |
| dims | 输入 | 循环位移维度，可省略或传空数组 | aclIntArray* | - | - |
| out | 输出 | 输出张量，shape 和 dtype 与 x 一致 | 与 x 相同 | ND | 0-8 维 |
| workspaceSize | 输出 | 需要申请的 workspace 大小 | uint64_t* | - | - |
| executor | 输出 | 执行器 | aclOpExecutor** | - | - |

- 返回值

  `aclnnStatus`。第一段接口完成参数校验，出现以下场景时返回错误：

  - `ACLNN_ERR_PARAM_NULLPTR`：`x`、`shifts`、`out`、`workspaceSize`、`executor` 为空。
  - `ACLNN_ERR_PARAM_INVALID`：
    - 输入或输出 dtype 不在支持范围内。
    - 输入与输出 dtype 不一致。
    - 输入或输出格式不是 `ND`。
    - 输入与输出 shape 不一致。
    - rank 大于 8。
    - 0 维输入时，`shifts` 长度不为 1 或 `dims` 非空。
    - `dims` 为空但 `shifts` 长度不为 1。
    - `dims` 非空但 `shifts` 与 `dims` 长度不一致。
    - `dims` 元素越界。

## aclnnRoll

| 参数名 | 输入/输出 | 描述 |
| --- | --- | --- |
| workspace | 输入 | Device 侧 workspace 地址 |
| workspaceSize | 输入 | Device 侧 workspace 大小 |
| executor | 输入 | 执行器 |
| stream | 输入 | 执行 stream |

## 约束说明

- 仅支持 `ND`。
- 仅支持 `uint8`、`int8`、`bfloat16`、`float16`、`float32`、`int32`、`uint32`。
- 0 维输入时，`shifts` 长度必须为 1，且 `dims` 为空。
- 非连续输入会先整理为连续视图后执行。
- 非连续输出会在算子结果生成后做回写。

## 调用示例

请参考 [test_aclnn_roll.cpp](../examples/test_aclnn_roll.cpp)。
