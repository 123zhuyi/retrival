# aclnnSquaredDifference

## 产品支持情况

| 产品 | 是否支持 |
| :--- | :---: |
| Atlas A2 训练系列产品（Ascend 910B） | √ |

本接口对应 `experimental/math/squared_difference`，当前只注册 `ascend910b`。

## 功能说明

接口计算两个 Tensor 的逐元素平方差：

```text
y = (x1 - x2) * (x1 - x2)
```

`x1` 和 `x2` 遵循 NumPy broadcast 规则。两者 dtype 必须一致，并且必须是 FLOAT、FLOAT16、BFLOAT16、INT32 或 INT64；Tensor 格式为 ND。输入 rank 可为 0-16，内部合轴后的维度数最多为 8。输出 `y` 的 dtype 与输入相同，shape 为 broadcast 后的 shape。

## 函数原型

算子使用两段式 ACLNN 接口。第一段完成参数检查、shape/broadcast 校验和执行器构造；第二段在指定 stream 上提交执行器。

```cpp
aclnnStatus aclnnSquaredDifferenceGetWorkspaceSize(
    const aclTensor *x1,
    const aclTensor *x2,
    const aclTensor *out,
    uint64_t *workspaceSize,
    aclOpExecutor **executor);
```

```cpp
aclnnStatus aclnnSquaredDifference(
    void *workspace,
    uint64_t workspaceSize,
    aclOpExecutor *executor,
    aclrtStream stream);
```

该实现的 workspace size 为 0。调用第二段时可将 `workspace` 设为 `nullptr`、`workspaceSize` 设为 0。

## aclnnSquaredDifferenceGetWorkspaceSize

| 参数 | 方向 | 说明 |
| :--- | :---: | :--- |
| `x1` | 输入 | 第一个操作数；支持 FLOAT、FLOAT16、BFLOAT16、INT32、INT64，ND 格式。 |
| `x2` | 输入 | 第二个操作数；dtype/格式约束与 `x1` 相同。 |
| `out` | 输出 | 预先创建的输出 Tensor，dtype 与输入相同，shape 必须等于 broadcast 后的 shape。 |
| `workspaceSize` | 输出 | 返回 workspace 字节数；当前为 0。 |
| `executor` | 输出 | 返回第二段接口使用的执行器。 |

第一段接口在以下情况返回参数错误或执行失败：Tensor 指针为空；输入 dtype 不一致或不在支持范围；格式不是 ND；输入 shape 不可 broadcast；输出 shape/dtype 不匹配；rank 超出约束；合轴后维度数超过 8；平台不支持该算子。

## aclnnSquaredDifference

| 参数 | 方向 | 说明 |
| :--- | :---: | :--- |
| `workspace` | 输入 | 第一段返回的 workspace 地址。当前 workspace size 为 0，可传 `nullptr`。 |
| `workspaceSize` | 输入 | 第一段返回的 workspace 大小。 |
| `executor` | 输入 | 第一段返回的执行器。 |
| `stream` | 输入 | 提交算子任务的 ACL stream。 |

返回值为 `aclnnStatus`。具体错误码请参见 CANN ACLNN 返回码文档。

## 调用示例

下面给出调用顺序的核心片段；完整资源管理和 Tensor 创建代码见 [`../examples/test_aclnn_squared_difference.cpp`](../examples/test_aclnn_squared_difference.cpp)。

```cpp
uint64_t workspaceSize = 0;
aclOpExecutor *executor = nullptr;

aclnnStatus ret = aclnnSquaredDifferenceGetWorkspaceSize(
    x1, x2, out, &workspaceSize, &executor);
// workspaceSize == 0 for the current implementation.

void *workspace = nullptr;
ret = aclnnSquaredDifference(workspace, workspaceSize, executor, stream);
```

调用第二段后，应使用 `aclrtSynchronizeStream(stream)` 等待任务完成，再从 `out` 的 device 地址拷贝结果，并按 ACLNN 生命周期释放 Tensor、executor、stream 和 device 内存。

## 约束与实现路径

- OneDim 路径处理合轴后单维的 same-shape 和标量广播。
- BRC 路径处理多维广播；单轴广播在支持的 dtype 上优先使用 BRCFast 优化路径。
- FP16/BF16 使用 FP32 中间计算后转换回输入 dtype；INT64 使用标量计算路径。
- 输出为空时不访问输入、输出或 UB 缓冲区。
