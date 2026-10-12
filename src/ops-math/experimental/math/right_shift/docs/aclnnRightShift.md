# aclnnRightShift

## 产品支持情况

| 产品                                                               | 是否支持 |
| ------------------------------------------------------------------ | :------: |
| Atlas A2 训练系列产品/Atlas 800I A2 推理产品/A200I A2 Box 异构组件 |    √     |

## 功能说明

- 算子功能：对输入张量 `input` 中每个元素，按照 `shiftBits` 对应位置的移位位数进行按位右移。
- 计算公式：

  $$
  out_i = input_i \gg shiftBits_i
  $$

- `input` 与 `shiftBits` 支持 broadcast，输出 `out` 的 shape 为二者 broadcast 后的 shape。
- 对有符号整数执行算术右移；对无符号整数执行逻辑右移。
- 当移位位数非法时，输出遵循以下规则：
  - 有符号整数：若 `shiftBits_i < 0` 或 `shiftBits_i >= bitWidth`，则 `input_i < 0` 时输出 `-1`，否则输出 `0`。
  - 无符号整数：若 `shiftBits_i >= bitWidth`，则输出 `0`。

## 函数原型

每个算子分为[两段式接口](../../../../docs/zh/context/two_phase_api.md)，必须先调用 `aclnnRightShiftGetWorkspaceSize` 接口获取计算所需 workspace 大小以及包含算子计算流程的执行器，再调用 `aclnnRightShift` 接口执行计算。

```c++
aclnnStatus aclnnRightShiftGetWorkspaceSize(
    const aclTensor *input,
    const aclTensor *shiftBits,
    aclTensor *out,
    uint64_t *workspaceSize,
    aclOpExecutor **executor)
```

```c++
aclnnStatus aclnnRightShift(
    void *workspace,
    uint64_t workspaceSize,
    aclOpExecutor *executor,
    aclrtStream stream)
```

## aclnnRightShiftGetWorkspaceSize

- 参数说明：

| 参数名        | 输入/输出 | 描述                                        | 使用说明                                                        | 数据类型                                                 | 数据格式 | 维度(shape) | 非连续Tensor |
| ------------- | --------- | ------------------------------------------- | --------------------------------------------------------------- | -------------------------------------------------------- | -------- | ----------- | ------------ |
| input         | 输入      | 待右移的输入张量，对应公式中的 `input`。    | 需要与 `shiftBits` 满足 broadcast 关系。                        | INT8、UINT8、INT16、UINT16、INT32、UINT32、INT64、UINT64 | ND       | 0-8         | √            |
| shiftBits     | 输入      | 右移位数张量，对应公式中的 `shiftBits`。    | 需要与 `input` 满足 broadcast 关系。                            | INT8、UINT8、INT16、UINT16、INT32、UINT32、INT64、UINT64 | ND       | 0-8         | √            |
| out           | 输出      | 右移计算结果，对应公式中的 `out`。          | shape 需要与 `input` 和 `shiftBits` broadcast 后的 shape 一致。 | INT8、UINT8、INT16、UINT16、INT32、UINT32、INT64、UINT64 | ND       | 0-8         | √            |
| workspaceSize | 输出      | 返回需要在 Device 侧申请的 workspace 大小。 | -                                                               | -                                                        | -        | -           | -            |
| executor      | 输出      | 返回 op 执行器，包含算子计算流程。          | -                                                               | -                                                        | -        | -           | -            |

- 返回值：

  aclnnStatus：返回状态码，具体参见[aclnn返回码](../../../../docs/zh/context/aclnn_return_code.md)。

  第一段接口会完成入参校验，出现以下场景时报错：

| 返回码                          | 错误码 | 描述                                                                                 |
| ------------------------------- | ------ | ------------------------------------------------------------------------------------ |
| ACLNN_ERR_PARAM_NULLPTR         | 161001 | `input`、`shiftBits`、`out`、`workspaceSize` 或 `executor` 为空指针。                |
| ACLNN_ERR_PARAM_INVALID         | 161002 | `input`、`shiftBits` 或 `out` 的数据类型不在支持范围内。                             |
| ACLNN_ERR_PARAM_INVALID         | 161002 | `input` 或 `shiftBits` 的维度超过 8 维。                                             |
| ACLNN_ERR_PARAM_INVALID         | 161002 | `input` 与 `shiftBits` 不满足 broadcast 关系。                                       |
| ACLNN_ERR_PARAM_INVALID         | 161002 | `out` 的 shape 与 broadcast 后的 shape 不一致。                                      |
| ACLNN_ERR_PARAM_INVALID         | 161002 | `input` 与 `shiftBits` 无法完成类型推导，或推导后的类型无法转换为 `out` 的数据类型。 |
| ACLNN_ERR_INNER_CREATE_EXECUTOR | -      | 创建执行器失败。                                                                     |
| ACLNN_ERR_INNER_NULLPTR         | -      | 内部计算流程创建失败。                                                               |

## aclnnRightShift

- 参数说明：

| 参数名        | 输入/输出 | 描述                                                                                     |
| ------------- | --------- | ---------------------------------------------------------------------------------------- |
| workspace     | 输入      | 在 Device 侧申请的 workspace 内存地址。                                                  |
| workspaceSize | 输入      | 在 Device 侧申请的 workspace 大小，由第一段接口 `aclnnRightShiftGetWorkspaceSize` 获取。 |
| executor      | 输入      | op 执行器，包含算子计算流程。                                                            |
| stream        | 输入      | 指定执行任务的 Stream。                                                                  |

- 返回值：

  aclnnStatus：返回状态码，具体参见[aclnn返回码](../../../../docs/zh/context/aclnn_return_code.md)。

## 约束说明

- 仅支持整数类型：INT8、UINT8、INT16、UINT16、INT32、UINT32、INT64、UINT64。
- `input`、`shiftBits`、`out` 均支持 0-8 维 Tensor，0 维表示标量。
- `input` 与 `shiftBits` 需要满足 broadcast 关系，`out` 的 shape 需要等于 broadcast 后的 shape。
- 支持空 Tensor，元素个数为 0 时跳过计算并返回空结果。
- 支持非连续 Tensor，接口内部会进行连续化处理。

## 调用示例

示例代码如下，仅供参考，具体编译和执行过程请参考[编译与运行样例](../../../../docs/zh/context/compile_and_run_sample.md)。

```c++
#include <cstdint>
#include <cstdio>
#include <vector>

#include "acl/acl.h"
#include "aclnnop/aclnn_right_shift.h"

#define CHECK_RET(cond, return_expr) \
    do {                             \
        if (!(cond)) {               \
            return_expr;             \
        }                            \
    } while (0)

int main()
{
    constexpr int32_t deviceId = 0;
    aclrtStream stream = nullptr;
    auto ret = aclInit(nullptr);
    CHECK_RET(ret == ACL_SUCCESS, return ret);
    ret = aclrtSetDevice(deviceId);
    CHECK_RET(ret == ACL_SUCCESS, return ret);
    ret = aclrtCreateStream(&stream);
    CHECK_RET(ret == ACL_SUCCESS, return ret);

    const std::vector<int64_t> shape = {2, 4};
    const std::vector<int32_t> inputHost = {-16, -8, -1, 0, 1, 8, 16, 32};
    const std::vector<int32_t> shiftHost = {0, 1, 2, 3, -1, 32, 4, 5};
    std::vector<int32_t> outHost(inputHost.size(), 0);

    const size_t bytes = inputHost.size() * sizeof(int32_t);
    void *inputDevice = nullptr;
    void *shiftDevice = nullptr;
    void *outDevice = nullptr;
    ret = aclrtMalloc(&inputDevice, bytes, ACL_MEM_MALLOC_HUGE_FIRST);
    CHECK_RET(ret == ACL_SUCCESS, return ret);
    ret = aclrtMalloc(&shiftDevice, bytes, ACL_MEM_MALLOC_HUGE_FIRST);
    CHECK_RET(ret == ACL_SUCCESS, return ret);
    ret = aclrtMalloc(&outDevice, bytes, ACL_MEM_MALLOC_HUGE_FIRST);
    CHECK_RET(ret == ACL_SUCCESS, return ret);

    ret = aclrtMemcpy(inputDevice, bytes, inputHost.data(), bytes, ACL_MEMCPY_HOST_TO_DEVICE);
    CHECK_RET(ret == ACL_SUCCESS, return ret);
    ret = aclrtMemcpy(shiftDevice, bytes, shiftHost.data(), bytes, ACL_MEMCPY_HOST_TO_DEVICE);
    CHECK_RET(ret == ACL_SUCCESS, return ret);

    std::vector<int64_t> strides = {4, 1};
    aclTensor *input = aclCreateTensor(
        shape.data(), shape.size(), ACL_INT32, strides.data(), 0, ACL_FORMAT_ND, shape.data(), shape.size(),
        inputDevice);
    aclTensor *shiftBits = aclCreateTensor(
        shape.data(), shape.size(), ACL_INT32, strides.data(), 0, ACL_FORMAT_ND, shape.data(), shape.size(),
        shiftDevice);
    aclTensor *out = aclCreateTensor(
        shape.data(), shape.size(), ACL_INT32, strides.data(), 0, ACL_FORMAT_ND, shape.data(), shape.size(),
        outDevice);

    uint64_t workspaceSize = 0;
    aclOpExecutor *executor = nullptr;
    ret = aclnnRightShiftGetWorkspaceSize(input, shiftBits, out, &workspaceSize, &executor);
    CHECK_RET(ret == ACL_SUCCESS, return ret);

    void *workspace = nullptr;
    if (workspaceSize > 0) {
        ret = aclrtMalloc(&workspace, workspaceSize, ACL_MEM_MALLOC_HUGE_FIRST);
        CHECK_RET(ret == ACL_SUCCESS, return ret);
    }

    ret = aclnnRightShift(workspace, workspaceSize, executor, stream);
    CHECK_RET(ret == ACL_SUCCESS, return ret);
    ret = aclrtSynchronizeStream(stream);
    CHECK_RET(ret == ACL_SUCCESS, return ret);

    ret = aclrtMemcpy(outHost.data(), bytes, outDevice, bytes, ACL_MEMCPY_DEVICE_TO_HOST);
    CHECK_RET(ret == ACL_SUCCESS, return ret);
    for (auto value : outHost) {
        std::printf("%d ", value);
    }
    std::printf("\n");

    aclDestroyTensor(input);
    aclDestroyTensor(shiftBits);
    aclDestroyTensor(out);
    aclrtFree(inputDevice);
    aclrtFree(shiftDevice);
    aclrtFree(outDevice);
    if (workspace != nullptr) {
        aclrtFree(workspace);
    }
    aclrtDestroyStream(stream);
    aclrtResetDevice(deviceId);
    aclFinalize();
    return ACL_SUCCESS;
}
```
