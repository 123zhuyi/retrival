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
#include "aclnn_remainder.h"

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

int Init(int32_t deviceId, aclrtStream* stream)
{
    auto ret = aclInit(nullptr);
    if (ret != ACL_SUCCESS) {
        LOG_PRINT("aclInit failed. ERROR: %d\n", ret);
        return ret;
    }
    ret = aclrtSetDevice(deviceId);
    if (ret != ACL_SUCCESS) {
        LOG_PRINT("aclrtSetDevice failed. ERROR: %d\n", ret);
        return ret;
    }
    ret = aclrtCreateStream(stream);
    if (ret != ACL_SUCCESS) {
        LOG_PRINT("aclrtCreateStream failed. ERROR: %d\n", ret);
        return ret;
    }
    return 0;
}

template <typename T>
int CreateAclTensor(
    const std::vector<T>& hostData, const std::vector<int64_t>& shape, void** deviceAddr, aclDataType dataType,
    aclTensor** tensor)
{
    auto size = GetShapeSize(shape) * sizeof(T);
    auto ret = aclrtMalloc(deviceAddr, size, ACL_MEM_MALLOC_HUGE_FIRST);
    if (ret != ACL_SUCCESS) {
        LOG_PRINT("aclrtMalloc failed. ERROR: %d\n", ret);
        return ret;
    }
    ret = aclrtMemcpy(*deviceAddr, size, hostData.data(), size, ACL_MEMCPY_HOST_TO_DEVICE);
    if (ret != ACL_SUCCESS) {
        LOG_PRINT("aclrtMemcpy failed. ERROR: %d\n", ret);
        return ret;
    }
    std::vector<int64_t> strides(shape.size(), 1);
    for (int64_t i = shape.size() - 2; i >= 0; i--) {
        strides[i] = shape[i + 1] * strides[i + 1];
    }
    *tensor = aclCreateTensor(
        shape.data(), shape.size(), dataType, strides.data(), 0, aclFormat::ACL_FORMAT_ND, shape.data(), shape.size(),
        *deviceAddr);
    return 0;
}

int Compute(aclrtStream stream, aclTensor* self, aclTensor* other, aclTensor* out)
{
    uint64_t workspaceSize = 0;
    aclOpExecutor* executor;
    auto ret = aclnnRemainderTensorTensorGetWorkspaceSize(self, other, out, &workspaceSize, &executor);
    if (ret != ACL_SUCCESS) {
        LOG_PRINT("aclnnRemainderTensorTensorGetWorkspaceSize failed. ERROR: %d\n", ret);
        return ret;
    }

    void* workspaceAddr = nullptr;
    if (workspaceSize > 0) {
        ret = aclrtMalloc(&workspaceAddr, workspaceSize, ACL_MEM_MALLOC_HUGE_FIRST);
        if (ret != ACL_SUCCESS) {
            LOG_PRINT("allocate workspace failed. ERROR: %d\n", ret);
            return ret;
        }
    }

    ret = aclnnRemainderTensorTensor(workspaceAddr, workspaceSize, executor, stream);
    if (ret != ACL_SUCCESS) {
        LOG_PRINT("aclnnRemainderTensorTensor failed. ERROR: %d\n", ret);
        return ret;
    }

    ret = aclrtSynchronizeStream(stream);
    if (ret != ACL_SUCCESS) {
        LOG_PRINT("aclrtSynchronizeStream failed. ERROR: %d\n", ret);
        return ret;
    }

    if (workspaceSize > 0) {
        aclrtFree(workspaceAddr);
    }
    return 0;
}

int PrintResult(void* outDeviceAddr, const std::vector<int64_t>& outShape)
{
    auto size = GetShapeSize(outShape);
    std::vector<float> resultData(size, 0);
    auto ret = aclrtMemcpy(
        resultData.data(), resultData.size() * sizeof(resultData[0]), outDeviceAddr, size * sizeof(resultData[0]),
        ACL_MEMCPY_DEVICE_TO_HOST);
    if (ret != ACL_SUCCESS) {
        LOG_PRINT("copy result from device to host failed. ERROR: %d\n", ret);
        return ret;
    }
    for (int64_t i = 0; i < size; i++) {
        LOG_PRINT("result[%ld] is: %f\n", i, resultData[i]);
    }
    return 0;
}

void DestroyResources(
    void* selfAddr, void* otherAddr, void* outAddr, aclTensor* self, aclTensor* other, aclTensor* out,
    aclrtStream stream, int32_t deviceId)
{
    aclDestroyTensor(self);
    aclDestroyTensor(other);
    aclDestroyTensor(out);
    aclrtFree(selfAddr);
    aclrtFree(otherAddr);
    aclrtFree(outAddr);
    aclrtDestroyStream(stream);
    aclrtResetDevice(deviceId);
    aclFinalize();
}

int main()
{
    int32_t deviceId = 0;
    aclrtStream stream;
    if (Init(deviceId, &stream) != 0)
        return -1;

    std::vector<int64_t> selfShape = {4, 4};
    std::vector<int64_t> otherShape = {4, 4};
    std::vector<int64_t> outShape = {4, 4};
    void *selfAddr = nullptr, *otherAddr = nullptr, *outAddr = nullptr;
    aclTensor *self = nullptr, *other = nullptr, *out = nullptr;

    std::vector<float> selfData = {5.5, -11.51, 36.23, 7, -10, -8, -15, -7, 10, 8, 15, 7, -10, -8, -15, -7};
    std::vector<float> otherData = {2, 3, -24.1, 2, 3, 5, 4, 2, -3, -5, -4, -2, -3, -5, -4, -2};
    std::vector<float> outData(selfData.size(), 0);

    if (CreateAclTensor(selfData, selfShape, &selfAddr, aclDataType::ACL_FLOAT, &self) != 0)
        return -1;
    if (CreateAclTensor(otherData, otherShape, &otherAddr, aclDataType::ACL_FLOAT, &other) != 0)
        return -1;
    if (CreateAclTensor(outData, outShape, &outAddr, aclDataType::ACL_FLOAT, &out) != 0)
        return -1;

    // 执行计算
    if (Compute(stream, self, other, out) != 0)
        return -1;

    // 打印结果
    if (PrintResult(outAddr, outShape) != 0)
        return -1;

    // 释放资源
    DestroyResources(selfAddr, otherAddr, outAddr, self, other, out, stream, deviceId);
    return 0;
}