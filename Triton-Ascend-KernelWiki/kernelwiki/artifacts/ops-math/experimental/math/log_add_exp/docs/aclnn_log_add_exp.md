# aclnnLogAddExp

## 支持的产品型号

| 产品                                                         | 是否支持 |
| :----------------------------------------------------------- | :------: |
| <term>Atlas A2 训练系列产品/Atlas A2 推理系列产品</term> |    √     |

## 功能描述

计算输入张量 `x` 与 `y` 的 LogAddExp，即 $out = \log(e^x + e^y)$。

计算时采用数值稳定公式：

$$out = \max(x, y) + \ln\!\left(1 + e^{-|x-y|}\right)$$

- 支持广播：`x` 与 `y` 的 shape 须满足 NumPy 广播规则，输出 shape 为广播后的 shape。
- 支持数据类型：float32、float16、bfloat16。

## 函数原型

```cpp
aclnnStatus aclnnLogAddExpGetWorkspaceSize(
    const aclTensor *x,
    const aclTensor *y,
    const aclTensor *out,
    uint64_t *workspaceSize,
    aclOpExecutor **executor);

aclnnStatus aclnnLogAddExp(
    void *workspace,
    uint64_t workspaceSize,
    aclOpExecutor *executor,
    aclrtStream stream);
```

## aclnnLogAddExpGetWorkspaceSize

### 参数说明

| 参数名 | 输入/输出 | 描述 |
|-------|---------|------|
| x | 输入 | 数据类型：float32、float16、bfloat16。数据格式：ND。支持非连续 tensor。最大支持 8 维。 |
| y | 输入 | 数据类型：float32、float16、bfloat16，须与 x 相同。数据格式：ND。支持非连续 tensor。最大支持 8 维。支持广播。 |
| out | 输出 | 数据类型与 x、y 相同。数据格式：ND。shape 须等于 broadcast(x.shape, y.shape)。支持非连续 tensor。 |
| workspaceSize | 输出 | 算子执行所需 workspace 大小，单位为 Byte。由本函数返回，调用方须据此分配 workspace 内存。 |
| executor | 输出 | 算子执行器，包含算子计算流信息，由本函数返回后传入 aclnnLogAddExp 执行。 |

### 返回值说明

返回 `aclnnStatus` 错误码，详见 [aclnn 错误码](#错误码)。

## aclnnLogAddExp

### 参数说明

| 参数名 | 输入/输出 | 描述 |
|-------|---------|------|
| workspace | 输入 | workspace 内存地址。若 workspaceSize 为 0，可传入 nullptr。 |
| workspaceSize | 输入 | workspace 大小，由 aclnnLogAddExpGetWorkspaceSize 返回。 |
| executor | 输入 | 算子执行器，由 aclnnLogAddExpGetWorkspaceSize 返回。 |
| stream | 输入 | ACL stream，用于异步调度算子执行。 |

### 返回值说明

返回 `aclnnStatus` 错误码，详见 [aclnn 错误码](#错误码)。

## 错误码

| 错误码 | 描述 |
|-------|------|
| ACLNN_SUCCESS（0） | 执行成功。 |
| ACLNN_ERR_PARAM_NULLPTR | 输入/输出 tensor 指针为空。 |
| ACLNN_ERR_PARAM_INVALID | 参数非法，包括：数据类型不支持、x 与 y 数据类型不一致、维度超过 8、out shape 与广播结果不一致等。 |
| ACLNN_ERR_INNER_CREATE_EXECUTOR | 内部创建算子执行器失败。 |
| ACLNN_ERR_INNER_NULLPTR | 内部 tensor 分配失败。 |
| ACLNN_ERR_INNER_INFERSHAPE_ERROR | 内部 InferShape 失败。 |

## 约束说明

- `x` 与 `y` ��为相同数据类型（不支持隐式类型提升）。
- `x` 与 `y` 最大支持 8 维。
- `out` 的 shape 须等于 `broadcast(x.shape, y.shape)`，否则返回 `ACLNN_ERR_PARAM_INVALID`。
- 支持空 tensor（元素数为 0），此时 workspaceSize 为 0，直接返回成功。
- workspace 须在调用 `aclnnLogAddExp` 之前分配，在 stream 中算子执行完成后方可释放。

## 调用示例

以下示例展示了 LogAddExp 算子的完整调用流程：

```cpp
#include <cstdio>
#include <vector>
#include "acl/acl.h"
#include "aclnn_log_add_exp.h"

int main() {
    // 1. 初始化 ACL 及设备
    aclInit(nullptr);
    aclrtSetDevice(0);
    aclrtStream stream;
    aclrtCreateStream(&stream);

    // 2. 准备输入数据（fp32，shape=[4]，同 shape）
    //    x = [1.0, 2.0, 3.0, 4.0]
    //    y = [1.0, 1.0, 1.0, 1.0]
    //    out = logaddexp(x, y) = ln(e^x + e^y)
    int64_t shape[] = {4};
    int64_t strides[] = {1};
    float x_host[] = {1.0f, 2.0f, 3.0f, 4.0f};
    float y_host[] = {1.0f, 1.0f, 1.0f, 1.0f};
    float out_host[4] = {0};

    void *x_dev = nullptr, *y_dev = nullptr, *out_dev = nullptr;
    size_t nbytes = 4 * sizeof(float);
    aclrtMalloc(&x_dev,   nbytes, ACL_MEM_MALLOC_NORMAL_ONLY);
    aclrtMalloc(&y_dev,   nbytes, ACL_MEM_MALLOC_NORMAL_ONLY);
    aclrtMalloc(&out_dev, nbytes, ACL_MEM_MALLOC_NORMAL_ONLY);
    aclrtMemcpy(x_dev, nbytes, x_host, nbytes, ACL_MEMCPY_HOST_TO_DEVICE);
    aclrtMemcpy(y_dev, nbytes, y_host, nbytes, ACL_MEMCPY_HOST_TO_DEVICE);

    // 3. 创建 aclTensor
    aclTensor *x = aclCreateTensor(shape, 1, ACL_FLOAT, strides, 0,
                                   ACL_FORMAT_ND, shape, 1, x_dev);
    aclTensor *y = aclCreateTensor(shape, 1, ACL_FLOAT, strides, 0,
                                   ACL_FORMAT_ND, shape, 1, y_dev);
    aclTensor *out = aclCreateTensor(shape, 1, ACL_FLOAT, strides, 0,
                                     ACL_FORMAT_ND, shape, 1, out_dev);

    // 4. 查询 workspace 大小并分配
    uint64_t workspaceSize = 0;
    aclOpExecutor *executor = nullptr;
    aclnnLogAddExpGetWorkspaceSize(x, y, out, &workspaceSize, &executor);

    void *workspace = nullptr;
    if (workspaceSize > 0)
        aclrtMalloc(&workspace, workspaceSize, ACL_MEM_MALLOC_NORMAL_ONLY);

    // 5. 执行算子
    aclnnLogAddExp(workspace, workspaceSize, executor, stream);
    aclrtSynchronizeStream(stream);

    // 6. 取回结果
    aclrtMemcpy(out_host, nbytes, out_dev, nbytes, ACL_MEMCPY_DEVICE_TO_HOST);
    printf("out = [%.4f, %.4f, %.4f, %.4f]\n",
           out_host[0], out_host[1], out_host[2], out_host[3]);
    // 期望: [1.3133, 2.1269, 3.0486, 4.0183]

    // 7. 释放资源
    if (workspace) aclrtFree(workspace);
    aclrtFree(x_dev); aclrtFree(y_dev); aclrtFree(out_dev);
    aclDestroyTensor(x); aclDestroyTensor(y); aclDestroyTensor(out);
    aclrtDestroyStream(stream);
    aclrtResetDevice(0);
    aclFinalize();
    return 0;
}
```

> **广播示例**：若 `x` 的 shape 为 `[4, 1]`，`y` 的 shape 为 `[1]`，则 `out` 的 shape 为 `[4, 1]`，`y` 自动沿第 0 维广播。此场景等价于 `softplus(x) = ln(e^x + 1)`。
