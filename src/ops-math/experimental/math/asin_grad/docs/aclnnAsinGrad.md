# aclnnAsinGrad

## 产品支持情况

- <term>Atlas A3 训练系列产品/Atlas A3 推理系列产品</term>：支持
- <term>Atlas A2 训练系列产品/Atlas A2 推理系列产品</term>：支持

## 功能说明

aclnnAsinGrad用于计算反正弦函数 Asin 的反向梯度。

计算公式如下：

```text
z = dy / sqrt(1 - y * y)
```

其中 `y` 表示 Asin 反向计算的输入张量，`dy` 表示上游梯度张量，`z` 表示输出梯度张量。

## 函数原型

每个算子分为两段式接口，必须先调用“aclnnAsinGradGetWorkspaceSize”接口获取计算所需workspace大小以及包含了算子计算流程的执行器，再调用“aclnnAsinGrad”接口执行计算。

```cpp
aclnnStatus aclnnAsinGradGetWorkspaceSize(
    const aclTensor *y,
    const aclTensor *dy,
    aclTensor *z,
    uint64_t *workspaceSize,
    aclOpExecutor **executor);

aclnnStatus aclnnAsinGrad(
    void *workspace,
    uint64_t workspaceSize,
    aclOpExecutor *executor,
    const aclrtStream stream);
```

## aclnnAsinGradGetWorkspaceSize

### 参数说明

| 参数名 | 输入/输出 | 描述 | 使用说明 | 数据类型 | 数据格式 | 维度 / (shape) | 非连续Tensor |
| --- | --- | --- | --- | --- | --- | --- | --- |
| y（const aclTensor*） | 输入 | 表示 Asin 反向计算的输入张量，对应计算公式中的 `y`。 | 支持空Tensor；数据类型、数据格式和shape需与 `dy`、`z` 保持一致。 | FLOAT、FLOAT16、BFLOAT16 | ND | 0-8维 | √ |
| dy（const aclTensor*） | 输入 | 表示上游梯度张量，对应计算公式中的 `dy`。 | 支持空Tensor；数据类型、数据格式和shape需与 `y`、`z` 保持一致。 | 数据类型与 `y` 保持一致 | ND | Shape与 `y` 保持一致 | √ |
| z（aclTensor*） | 输出 | 表示输出梯度张量，对应计算公式中的 `z`。 | 支持空Tensor；数据类型、数据格式和shape需与 `y`、`dy` 保持一致。 | 数据类型与 `y` 保持一致 | ND | Shape与 `y` 保持一致 | √ |
| workspaceSize（uint64_t*） | 输出 | 返回需要在Device侧申请的workspace大小。 | - | - | - | - | - |
| executor（aclOpExecutor**） | 输出 | 返回op执行器，包含了算子计算流程。 | - | - | - | - | - |

### 返回值

aclnnStatus：返回状态码，具体参见aclnn返回码。

第一段接口完成入参校验，出现以下场景时报错：

| 返回值 | 错误码 | 描述 |
| --- | --- | --- |
| ACLNN_ERR_PARAM_NULLPTR | 161001 | `y`、`dy`、`z`、`workspaceSize` 或 `executor` 存在空指针。 |
| ACLNN_ERR_PARAM_INVALID | 161002 | `y`、`dy` 或 `z` 的数据类型不在支持范围内。 |
| ACLNN_ERR_PARAM_INVALID | 161002 | `y`、`dy` 和 `z` 的数据类型不一致。 |
| ACLNN_ERR_PARAM_INVALID | 161002 | `y`、`dy` 和 `z` 的shape不一致。 |
| ACLNN_ERR_PARAM_INVALID | 161002 | `y`、`dy` 或 `z` 的维度超过8维。 |

## aclnnAsinGrad

### 参数说明

| 参数名 | 输入/输出 | 描述 |
| --- | --- | --- |
| workspace | 输入 | 在Device侧申请的workspace内存地址。 |
| workspaceSize | 输入 | 在Device侧申请的workspace大小，由第一段接口aclnnAsinGradGetWorkspaceSize获取。 |
| executor | 输入 | op执行器，包含了算子计算流程。 |
| stream | 输入 | 指定执行任务的Stream。 |

### 返回值

aclnnStatus：返回状态码，具体参见aclnn返回码。

## 约束说明

- aclnnAsinGrad默认确定性实现。
- `y`、`dy`、`z` 的数据类型需要保持一致，支持 FLOAT、FLOAT16、BFLOAT16。
- `y`、`dy`、`z` 的shape需要保持一致，不支持broadcast。
- 仅支持ND数据格式。
- 支持0-8维输入。
- FLOAT16路径在half精度上直接计算；BFLOAT16路径将输入转换为FLOAT后计算，再将结果转换回BFLOAT16。
- 当 `1 - y * y` 小于0或分母为0时，结果遵循硬件 `sqrt` 和 `div` 指令对NaN/Inf的处理行为。

## 调用示例

示例代码如下，仅供参考，具体编译和执行过程请参考算子目录下的调用样例。

```cpp
// 调用aclnnAsinGrad第一段接口
uint64_t workspaceSize = 0;
aclOpExecutor *executor = nullptr;
auto ret = aclnnAsinGradGetWorkspaceSize(y, dy, z, &workspaceSize, &executor);

// 根据第一段接口返回的workspaceSize申请workspace，并调用第二段接口执行计算
void *workspaceAddr = nullptr;
if (workspaceSize > 0) {
    ret = aclrtMalloc(&workspaceAddr, workspaceSize, ACL_MEM_MALLOC_HUGE_FIRST);
}
ret = aclnnAsinGrad(workspaceAddr, workspaceSize, executor, stream);
```

完整样例请参考：[test_aclnn_asin_grad.cpp](../examples/test_aclnn_asin_grad.cpp)。
