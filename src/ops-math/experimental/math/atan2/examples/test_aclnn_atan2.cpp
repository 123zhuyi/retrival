/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

/*!
 * \file test_aclnn_atan2.cpp
 * \brief aclnn example for Atan2 operator.
 *
 * Tests atan2(y, x) with a 4-D float32 tensor.
 * Expected: out[i] = std::atan2(y[i], x[i])
 */

#include <iostream>
#include <vector>
#include <cmath>
#include "acl/acl.h"
#include "aclnn_atan2.h"

using DataType = float;

#define CHECK_RET(cond, return_expr) \
    do {                             \
        if (!(cond)) {               \
            return_expr;             \
        }                            \
    } while (0)

#define LOG_PRINT(message, ...)         \
    do {                                \
        printf(message, ##__VA_ARGS__); \
    } while (0)

int64_t GetShapeSize(const std::vector<int64_t>& shape)
{
    int64_t shapeSize = 1;
    for (auto i : shape) {
        shapeSize *= i;
    }
    return shapeSize;
}

void PrintOutResult(std::vector<int64_t>& shape, void** deviceAddr)
{
    auto size = GetShapeSize(shape);
    std::vector<DataType> resultData(size, 0);
    auto ret = aclrtMemcpy(
        resultData.data(), resultData.size() * sizeof(DataType), *deviceAddr, size * sizeof(DataType),
        ACL_MEMCPY_DEVICE_TO_HOST);
    CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("copy result from device to host failed. ERROR: %d\n", ret); return);
    for (int64_t i = 0; i < size && i < 16; i++) {
        LOG_PRINT("result[%ld] = %f\n", i, resultData[i]);
    }
}

int Init(int32_t deviceId, aclrtStream* stream)
{
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
    auto ret = aclrtMalloc(deviceAddr, size, ACL_MEM_MALLOC_HUGE_FIRST);
    CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclrtMalloc failed. ERROR: %d\n", ret); return ret);
    ret = aclrtMemcpy(*deviceAddr, size, hostData.data(), size, ACL_MEMCPY_HOST_TO_DEVICE);
    CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclrtMemcpy failed. ERROR: %d\n", ret); return ret);

    std::vector<int64_t> strides(shape.size(), 1);
    for (int64_t i = (int64_t)shape.size() - 2; i >= 0; i--) {
        strides[i] = shape[i + 1] * strides[i + 1];
    }
    *tensor = aclCreateTensor(
        shape.data(), shape.size(), dataType, strides.data(), 0, aclFormat::ACL_FORMAT_ND, shape.data(), shape.size(),
        *deviceAddr);
    return 0;
}

int main()
{
    // 1. ACL 初始化
    int32_t deviceId = 0;
    aclrtStream stream;
    auto ret = Init(deviceId, &stream);
    CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("Init acl failed. ERROR: %d\n", ret); return ret);

    // 2. 构造输入输出张量
    //    shape: [4, 4, 4, 4] = 256 elements
    std::vector<int64_t> tensorShape = {4, 4, 4, 4};
    int64_t numElements = GetShapeSize(tensorShape);

    // x1 = y argument: values in [-3, 3]
    std::vector<DataType> x1HostData(numElements);
    for (int64_t i = 0; i < numElements; i++) {
        x1HostData[i] = static_cast<float>(i % 7) - 3.0f; // -3,-2,...,3,-3,...
    }

    // x2 = x argument: values spanning all quadrants including 0 and negatives
    std::vector<DataType> x2HostData(numElements);
    for (int64_t i = 0; i < numElements; i++) {
        x2HostData[i] = static_cast<float>((i + 3) % 7) - 3.0f;
    }

    std::vector<DataType> outHostData(numElements, 0.0f);

    aclTensor* x1Tensor = nullptr;
    void* x1DeviceAddr = nullptr;
    ret = CreateAclTensor(x1HostData, tensorShape, &x1DeviceAddr, aclDataType::ACL_FLOAT, &x1Tensor);
    CHECK_RET(ret == ACL_SUCCESS, return ret);

    aclTensor* x2Tensor = nullptr;
    void* x2DeviceAddr = nullptr;
    ret = CreateAclTensor(x2HostData, tensorShape, &x2DeviceAddr, aclDataType::ACL_FLOAT, &x2Tensor);
    CHECK_RET(ret == ACL_SUCCESS, return ret);

    aclTensor* outTensor = nullptr;
    void* outDeviceAddr = nullptr;
    ret = CreateAclTensor(outHostData, tensorShape, &outDeviceAddr, aclDataType::ACL_FLOAT, &outTensor);
    CHECK_RET(ret == ACL_SUCCESS, return ret);

    // 3. 调用 aclnnAtan2 第一段接口获取 workspaceSize
    uint64_t workspaceSize = 0;
    aclOpExecutor* executor = nullptr;
    ret = aclnnAtan2GetWorkspaceSize(x1Tensor, x2Tensor, outTensor, &workspaceSize, &executor);
    CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclnnAtan2GetWorkspaceSize failed. ERROR: %d\n", ret); return ret);

    // 4. 申请 workspace
    void* workspaceAddr = nullptr;
    if (workspaceSize > 0) {
        ret = aclrtMalloc(&workspaceAddr, workspaceSize, ACL_MEM_MALLOC_HUGE_FIRST);
        CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("allocate workspace failed. ERROR: %d\n", ret); return ret);
    }

    // 5. 调用 aclnnAtan2 第二段接口执行计算
    ret = aclnnAtan2(workspaceAddr, workspaceSize, executor, stream);
    CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclnnAtan2 failed. ERROR: %d\n", ret); return ret);

    // 6. 同步等待
    ret = aclrtSynchronizeStream(stream);
    CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclrtSynchronizeStream failed. ERROR: %d\n", ret); return ret);

    // 7. 打印结果（前16个）
    LOG_PRINT("=== Atan2 output (first 16 elements) ===\n");
    PrintOutResult(tensorShape, &outDeviceAddr);

    // 8. 释放资源
    aclDestroyTensor(x1Tensor);
    aclDestroyTensor(x2Tensor);
    aclDestroyTensor(outTensor);
    aclrtFree(x1DeviceAddr);
    aclrtFree(x2DeviceAddr);
    aclrtFree(outDeviceAddr);
    if (workspaceSize > 0) {
        aclrtFree(workspaceAddr);
    }
    aclrtDestroyStream(stream);
    aclrtResetDevice(deviceId);
    aclFinalize();

    return 0;
}
