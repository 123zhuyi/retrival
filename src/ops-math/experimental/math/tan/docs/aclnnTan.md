# aclnnTan

## 支持的产品型号

| 产品                                                         | 是否支持 |
| :----------------------------------------------------------- | :------: |
| <term>Atlas A2 训练系列产品/Atlas A2 推理系列产品</term> |    √     |

## 功能描述

计算输入张量 `x` 的逐元素正切值，即 $out = \tan(x) = \frac{\sin(x)}{\cos(x)}$。

- 不支持广播：`out` 的 shape 与 `x` 相同。
- 支持数据类型：float32、float16。

## 函数原型

```cpp
aclnnStatus aclnnTanGetWorkspaceSize(
    const aclTensor *x,
    const aclTensor *out,
    uint64_t *workspaceSize,
    aclOpExecutor **executor);

aclnnStatus aclnnTan(
    void *workspace,
    uint64_t workspaceSize,
    aclOpExecutor *executor,
    aclrtStream stream);
```

## aclnnTanGetWorkspaceSize

### 参数说明

| 参数名 | 输入/输出 | 描述 |
|-------|---------|------|
| x | 输入 | 数据类型：float32、float16。数据格式：ND。支持非连续 tensor。 |
| out | 输出 | 数据类型与 x 相同。数据格式：ND。shape 须与 x 相同。支持非连续 tensor。 |
| workspaceSize | 输出 | 算子执行所需 workspace 大小，单位为 Byte。由本函数返回，调用方须据此分配 workspace 内存。 |
| executor | 输出 | 算子执行器，包含算子计算流信息，由本函数返回后传入 aclnnTan 执行。 |

### 返回值说明

返回 `aclnnStatus` 错误码，详见 [aclnn 错误码](#错误码)。

## aclnnTan

### 参数说明

| 参数名 | 输入/输出 | 描述 |
|-------|---------|------|
| workspace | 输入 | workspace 内存地址。若 workspaceSize 为 0，可传入 nullptr。 |
| workspaceSize | 输入 | workspace 大小，由 aclnnTanGetWorkspaceSize 返回。 |
| executor | 输入 | 算子执行器，由 aclnnTanGetWorkspaceSize 返回。 |
| stream | 输入 | ACL stream，用于异步调度算子执行。 |

### 返回值说明

返回 `aclnnStatus` 错误码，详见 [aclnn 错误码](#错误码)。

## 错误码

| 错误码 | 描述 |
|-------|------|
| ACLNN_SUCCESS（0） | 执行成功。 |
| ACLNN_ERR_PARAM_NULLPTR | 输入/输出 tensor 指针为空。 |
| ACLNN_ERR_PARAM_INVALID | 参数非法，包括：数据类型不支持、out shape 与 x 不一致等。 |
| ACLNN_ERR_INNER_CREATE_EXECUTOR | 内部创建算子执行器失败。 |
| ACLNN_ERR_INNER_NULLPTR | 内部 tensor 分配失败。 |
| ACLNN_ERR_INNER_INFERSHAPE_ERROR | 内部 InferShape 失败。 |

## 约束说明

- 输入 `x` 支持 float32 和 float16 数据类型。
- `out` 的数据类型须与 `x` 相同。
- `out` 的 shape 须与 `x` 相同（不支持广播）。
- 支持标量输入（内部转为 shape {1} 处理）。
- 支持空 tensor（元素数为 0），此时 workspaceSize 为 0，直接返回成功。
- workspace 须在调用 `aclnnTan` 之前分配，在 stream 中算子执行完成后方可释放。
- 当输入值接近 $\frac{\pi}{2} + k\pi$（$k$ 为整数）时，正切函数结果趋向无穷大，可能出现精度下降或溢出。

## 调用示例

以下示例展示了 Tan 算子的完整调用流程：

```cpp
#include <cstdio>
#include <vector>
#include "acl/acl.h"
#include "aclnn_tan.h"

int main() {
    // 1. 初始化 ACL 及设备
    aclInit(nullptr);
    aclrtSetDevice(0);
    aclrtStream stream;
    aclrtCreateStream(&stream);

    // 2. 准备输入数据（fp32，shape=[2,4]）
    //    x = [0.0, 0.5, 1.0, -1.0, 0.25, -0.5, 2.0, -2.0]
    //    out = tan(x)
    int64_t shape[] = {2, 4};
    int64_t strides[] = {4, 1};
    float x_host[] = {0.0f, 0.5f, 1.0f, -1.0f, 0.25f, -0.5f, 2.0f, -2.0f};
    float out_host[8] = {0};

    void *x_dev = nullptr, *out_dev = nullptr;
    size_t nbytes = 8 * sizeof(float);
    aclrtMalloc(&x_dev,   nbytes, ACL_MEM_MALLOC_NORMAL_ONLY);
    aclrtMalloc(&out_dev, nbytes, ACL_MEM_MALLOC_NORMAL_ONLY);
    aclrtMemcpy(x_dev, nbytes, x_host, nbytes, ACL_MEMCPY_HOST_TO_DEVICE);

    // 3. 创建 aclTensor
    aclTensor *x = aclCreateTensor(shape, 2, ACL_FLOAT, strides, 0,
                                   ACL_FORMAT_ND, shape, 2, x_dev);
    aclTensor *out = aclCreateTensor(shape, 2, ACL_FLOAT, strides, 0,
                                     ACL_FORMAT_ND, shape, 2, out_dev);

    // 4. 查询 workspace 大小并分配
    uint64_t workspaceSize = 0;
    aclOpExecutor *executor = nullptr;
    aclnnTanGetWorkspaceSize(x, out, &workspaceSize, &executor);

    void *workspace = nullptr;
    if (workspaceSize > 0)
        aclrtMalloc(&workspace, workspaceSize, ACL_MEM_MALLOC_NORMAL_ONLY);

    // 5. 执行算子
    aclnnTan(workspace, workspaceSize, executor, stream);
    aclrtSynchronizeStream(stream);

    // 6. 取回结果
    aclrtMemcpy(out_host, nbytes, out_dev, nbytes, ACL_MEMCPY_DEVICE_TO_HOST);
    printf("out = [%.4f, %.4f, %.4f, %.4f, %.4f, %.4f, %.4f, %.4f]\n",
           out_host[0], out_host[1], out_host[2], out_host[3],
           out_host[4], out_host[5], out_host[6], out_host[7]);
    // 期望: [0.0000, 0.5463, 1.5574, -1.5574, 0.2553, -0.5463, -2.1850, 2.1850]

    // 7. 释放资源
    if (workspace) aclrtFree(workspace);
    aclrtFree(x_dev); aclrtFree(out_dev);
    aclDestroyTensor(x); aclDestroyTensor(out);
    aclrtDestroyStream(stream);
    aclrtResetDevice(0);
    aclFinalize();
    return 0;
}
```
