/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include <iostream>
#include <vector>
#include "acl/acl.h"
#include "aclnnop/aclnn_isin.h"

#define CHECK_RET(cond, return_expr) \
  do {                               \
    if (!(cond)) {                   \
      return_expr;                   \
    }                                \
  } while (0)

#define LOG_PRINT(message, ...)     \
  do {                              \
    printf(message, ##__VA_ARGS__); \
  } while (0)

int64_t GetShapeSize(const std::vector<int64_t>& shape)
{
    int64_t shape_size = 1;
    for (auto i : shape) {
        shape_size *= i;
    }
    return shape_size;
}

int Init(int32_t deviceId, aclrtStream* stream)
{
    // 固定写法，资源初始化
    auto ret = aclInit(nullptr);
    CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclInit failed. ERROR: %d\n", ret); return ret);
    ret = aclrtSetDevice(deviceId);
    CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclrtSetDevice failed. ERROR: %d\n", ret); return ret);
    ret = aclrtCreateStream(stream);
    CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclrtCreateStream failed. ERROR: %d\n", ret); return ret);
    return 0;
}

template <typename T>
int CreateAclTensor(
    const std::vector<T>& hostData, const std::vector<int64_t>& shape, void** deviceAddr, aclDataType dataType,
    aclTensor** tensor)
{
    auto size = GetShapeSize(shape) * sizeof(T);
    // 调用aclrtMalloc申请device侧内存
    auto ret = aclrtMalloc(deviceAddr, size, ACL_MEM_MALLOC_HUGE_FIRST);
    CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclrtMalloc failed. ERROR: %d\n", ret); return ret);

    // 调用aclrtMemcpy将host侧数据拷贝到device侧内存上
    ret = aclrtMemcpy(*deviceAddr, size, hostData.data(), size, ACL_MEMCPY_HOST_TO_DEVICE);
    CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclrtMemcpy failed. ERROR: %d\n", ret); return ret);

    // 计算连续tensor的strides
    std::vector<int64_t> strides(shape.size(), 1);
    for (int64_t i = shape.size() - 2; i >= 0; i--) {
        strides[i] = shape[i + 1] * strides[i + 1];
    }

    // 调用aclCreateTensor接口创建aclTensor
    *tensor = aclCreateTensor(
        shape.data(), shape.size(), dataType, strides.data(), 0, aclFormat::ACL_FORMAT_ND, shape.data(), shape.size(),
        *deviceAddr);
    return 0;
}

aclError InitAcl(int32_t deviceId, aclrtStream* stream)
{
    auto ret = Init(deviceId, stream);
    CHECK_RET(ret == 0, LOG_PRINT("Init acl failed. ERROR: %d\n", ret); return ret);
    return ACL_SUCCESS;
}

aclError CreateInputs(
    std::vector<int64_t>& testElementsShape, std::vector<int64_t>& outShape, void** testElementsDeviceAddr,
    void** outDeviceAddr, aclTensor** testElements, aclTensor** out, aclScalar** element, bool& assumeUnique,
    bool& invert)
{
    std::vector<double> testElementsHostData = {1.0, 2.0, 3.0};
    std::vector<char> outHostData = {0};
    double elementValue = 4.0;

    // 创建 testElements Tensor
    auto ret = CreateAclTensor(
        testElementsHostData, testElementsShape, testElementsDeviceAddr, aclDataType::ACL_DOUBLE, testElements);
    CHECK_RET(ret == ACL_SUCCESS, return ret);

    // 创建 element Scalar
    *element = aclCreateScalar(&elementValue, aclDataType::ACL_DOUBLE);
    CHECK_RET(*element != nullptr, return ACL_ERROR_INVALID_PARAM);

    // 创建 out Tensor
    ret = CreateAclTensor(outHostData, outShape, outDeviceAddr, aclDataType::ACL_BOOL, out);
    CHECK_RET(ret == ACL_SUCCESS, return ret);

    return ACL_SUCCESS;
}

aclError ExecOpApi(
    aclScalar* element, aclTensor* testElements, bool assumeUnique, bool invert, aclTensor* out,
    void** workspaceAddrOut, uint64_t& workspaceSize, void* outDeviceAddr, aclrtStream stream)
{
    aclOpExecutor* executor;

    // 第一段接口
    auto ret = aclnnIsInScalarTensorGetWorkspaceSize(
        element, testElements, assumeUnique, invert, out, &workspaceSize, &executor);
    CHECK_RET(
        ret == ACL_SUCCESS, LOG_PRINT("aclnnIsInScalarTensorGetWorkspaceSize failed. ERROR: %d\n", ret); return ret);

    // 分配 workspace
    void* workspaceAddr = nullptr;
    if (workspaceSize > 0) {
        ret = aclrtMalloc(&workspaceAddr, workspaceSize, ACL_MEM_MALLOC_HUGE_FIRST);
        CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("allocate workspace failed. ERROR: %d\n", ret); return ret);
    }
    *workspaceAddrOut = workspaceAddr;

    // 第二段接口
    ret = aclnnIsInScalarTensor(workspaceAddr, workspaceSize, executor, stream);
    CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclnnIsInScalarTensor failed. ERROR: %d\n", ret); return ret);

    // 同步
    ret = aclrtSynchronizeStream(stream);
    CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclrtSynchronizeStream failed. ERROR: %d\n", ret); return ret);

    // 输出拷贝
    char resultData = 0;
    ret = aclrtMemcpy(&resultData, sizeof(char), outDeviceAddr, sizeof(char), ACL_MEMCPY_DEVICE_TO_HOST);
    CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("copy result from device to host failed. ERROR: %d\n", ret); return ret);

    LOG_PRINT("result is: %d\n", static_cast<bool>(resultData));

    return ACL_SUCCESS;
}

int main()
{
    int32_t deviceId = 0;
    aclrtStream stream;

    auto ret = InitAcl(deviceId, &stream);
    CHECK_RET(ret == ACL_SUCCESS, return ret);

    std::vector<int64_t> testElementsShape = {3};
    std::vector<int64_t> outShape = {};
    void* testElementsDeviceAddr = nullptr;
    void* outDeviceAddr = nullptr;
    aclTensor* testElements = nullptr;
    aclTensor* out = nullptr;
    aclScalar* element = nullptr;

    bool assumeUnique = false;
    bool invert = true;

    ret = CreateInputs(
        testElementsShape, outShape, &testElementsDeviceAddr, &outDeviceAddr, &testElements, &out, &element, assumeUnique,
        invert);
    CHECK_RET(ret == ACL_SUCCESS, return ret);

    uint64_t workspaceSize = 0;
    void* workspaceAddr = nullptr;

    ret =
        ExecOpApi(element, testElements, assumeUnique, invert, out, &workspaceAddr, workspaceSize, outDeviceAddr, stream);
    CHECK_RET(ret == ACL_SUCCESS, return ret);

    // 释放
    aclDestroyScalar(element);
    aclDestroyTensor(testElements);
    aclDestroyTensor(out);

    aclrtFree(testElementsDeviceAddr);
    aclrtFree(outDeviceAddr);

    if (workspaceSize > 0) {
        aclrtFree(workspaceAddr);
    }

    aclrtDestroyStream(stream);
    aclrtResetDevice(deviceId);
    aclFinalize();
    return 0;
}