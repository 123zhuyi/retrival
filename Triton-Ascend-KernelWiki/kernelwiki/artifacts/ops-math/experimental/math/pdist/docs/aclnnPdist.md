# aclnnPdist

## 产品支持情况

| 产品 | 是否支持 |
| :----------------------------------------- | :------:|
| Ascend 950PR/Ascend 950DT | × |
| Atlas A3 训练系列产品/Atlas A3 推理系列产品 | √ |
| Atlas A2 训练系列产品/Atlas A2 推理系列产品 | √ |
| Atlas 200I/500 A2 推理产品 | × |
| Atlas 推理系列产品 | × |
| Atlas 训练系列产品 | × |

## 功能说明

计算输入二维 tensor 各行之间的 p-范数成对距离。等价于 PyTorch 的 `torch.nn.functional.pdist`。

计算公式：

  $$
  \text{dist}(i, j) =
  \begin{cases}
  \left( \sum_{k=0}^{M-1} |x_{ik} - x_{jk}|^p \right)^{1/p} & 0 < p < \infty \\
  \sum_{k=0}^{M-1} \mathbb{1}(x_{ik} \neq x_{jk}) & p = 0 \\
  \max_{k=0}^{M-1} |x_{ik} - x_{jk}| & p = \infty
  \end{cases}
  $$

其中 x 为输入 tensor，形状 (N, M)；p 为距离参数（标量，p ≥ 0）；输出为一维 tensor，形状 (N*(N-1)/2,)，按上三角行优先顺序排列。

## 函数原型

每个算子分为两段式接口，必须先调用"aclnnPdistGetWorkspaceSize"接口获取计算所需 workspace 大小以及包含了算子计算流程的执行器，再调用"aclnnPdist"接口执行计算。

```cpp
aclnnStatus aclnnPdistGetWorkspaceSize(
    const aclTensor  *self,
    double            p,
    aclTensor        *out,
    uint64_t         *workspaceSize,
    aclOpExecutor   **executor)
```

```cpp
aclnnStatus aclnnPdist(
    void           *workspace,
    uint64_t        workspaceSize,
    aclOpExecutor  *executor,
    aclrtStream     stream)
```

## aclnnPdistGetWorkspaceSize

- **参数说明**

| 参数名 | 输入/输出 | 描述 | 数据类型 | 数据格式 | 维度(shape) |
|--------|---------|------|---------|---------|------------|
| self（const aclTensor*） | 输入 | 输入二维tensor，对应公式中x。 | FLOAT、FLOAT16 | ND | 2维 (N, M)，N≥2, M≥1 |
| p（double） | 输入 | 距离参数，对应公式中p。p≥0，含inf。 | - | - | - |
| out（aclTensor*） | 输出 | 输出一维tensor，对应公式中dist。 | FLOAT、FLOAT16 | ND | 1维 (N*(N-1)/2,) |
| workspaceSize（uint64_t*） | 输出 | 返回需要在Device侧申请的workspace大小。 | - | - | - |
| executor（aclOpExecutor**） | 输出 | 返回op执行器，包含了算子计算流程。 | - | - | - |

- **返回值**

  aclnnStatus：返回状态码。

  第一段接口完成入参校验，出现以下场景时报错：

| 返回值 | 错误码 | 描述 |
|--------|--------|------|
| ACLNN_ERR_PARAM_NULLPTR | 161001 | self、out存在空指针。 |
| ACLNN_ERR_PARAM_INVALID | 161002 | self的数据类型不在支持的范围之内。 |
| ACLNN_ERR_PARAM_INVALID | 161002 | self不是二维tensor或N<2。 |
| ACLNN_ERR_PARAM_INVALID | 161002 | p < 0。 |
| ACLNN_ERR_PARAM_INVALID | 161002 | out的shape与N*(N-1)/2不匹配。 |

## aclnnPdist

- **参数说明**

| 参数名 | 输入/输出 | 描述 |
|--------|---------|------|
| workspace | 输入 | 在Device侧申请的workspace内存地址。 |
| workspaceSize | 输入 | 在Device侧申请的workspace大小，由第一段接口aclnnPdistGetWorkspaceSize获取。 |
| executor | 输入 | op执行器，包含了算子计算流程。 |
| stream | 输入 | 指定执行任务的Stream。 |

- **返回值**

  aclnnStatus：返回状态码。

## 调用说明

| 调用方式 | 样例代码 | 说明 |
| :------- | :------- | :--- |
| aclnn接口 | [test_aclnn_pdist](../examples/test_aclnn_pdist.cpp) | 通过 aclnnPdist 接口调用 Pdist 算子。 |

## 约束说明

- self 必须为二维 tensor，且 dim(0) ≥ 2。
- out 的数据类型必须与 self 一致。
- p 的取值范围为 [0, +inf]。
- aclnnPdist默认确定性实现。

## 调用示例

示例代码如下，仅供参考，具体编译和执行过程请参考编译与运行样例。

```Cpp
#include <iostream>
#include <vector>
#include "acl/acl.h"
#include "aclnn_pdist.h"

// 省略 Init、CreateAclTensor 等通用函数，完整代码见 examples/test_aclnn_pdist.cpp

int main()
{
    int32_t deviceId = 0;
    aclrtStream stream;
    Init(deviceId, &stream);

    // 构造输入: 4x3 矩阵
    std::vector<int64_t> inputShape = {4, 3};
    std::vector<float> inputData = {1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12};
    aclTensor* inputTensor = nullptr;
    void* inputDeviceAddr = nullptr;
    CreateAclTensor(inputData, inputShape, &inputDeviceAddr, ACL_FLOAT, &inputTensor);

    // 构造输出: N*(N-1)/2 = 6
    std::vector<int64_t> outputShape = {6};
    std::vector<float> outputData(6, 0);
    aclTensor* outputTensor = nullptr;
    void* outputDeviceAddr = nullptr;
    CreateAclTensor(outputData, outputShape, &outputDeviceAddr, ACL_FLOAT, &outputTensor);

    // 调用第一段接口
    float p = 2.0f;
    uint64_t workspaceSize = 0;
    aclOpExecutor* executor;
    aclnnPdistGetWorkspaceSize(inputTensor, p, outputTensor, &workspaceSize, &executor);

    void* workspaceAddr = nullptr;
    if (workspaceSize > 0) {
        aclrtMalloc(&workspaceAddr, workspaceSize, ACL_MEM_MALLOC_HUGE_FIRST);
    }

    // 调用第二段接口
    aclnnPdist(workspaceAddr, workspaceSize, executor, stream);
    aclrtSynchronizeStream(stream);

    // 释放资源
    aclDestroyTensor(inputTensor);
    aclDestroyTensor(outputTensor);
    aclrtFree(inputDeviceAddr);
    aclrtFree(outputDeviceAddr);
    if (workspaceSize > 0) aclrtFree(workspaceAddr);
    aclrtDestroyStream(stream);
    aclrtResetDevice(deviceId);
    aclFinalize();
    return 0;
}
```
