/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
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
#include "aclnn_view_copy.h"

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

int Init(int32_t deviceId, aclrtStream* stream)
{
    auto ret = aclInit(nullptr);
    CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclInit failed. ERROR: %d\n", ret); return ret);
    ret = aclrtSetDevice(deviceId);
    if (ret != ACL_SUCCESS) {
        LOG_PRINT("aclrtSetDevice failed. ERROR: %d\n", ret);
        aclFinalize();
        return ret;
    }
    ret = aclrtCreateStream(stream);
    if (ret != ACL_SUCCESS) {
        LOG_PRINT("aclrtCreateStream failed. ERROR: %d\n", ret);
        aclrtResetDevice(deviceId);
        aclFinalize();
        return ret;
    }
    return ACL_SUCCESS;
}

template <typename T>
int CreateAclTensor(const std::vector<T>& hostData, const std::vector<int64_t>& shape, void** deviceAddr,
                    aclDataType dataType, aclTensor** tensor)
{
    const auto size = static_cast<size_t>(GetShapeSize(shape)) * sizeof(T);
    auto ret = aclrtMalloc(deviceAddr, size, ACL_MEM_MALLOC_HUGE_FIRST);
    CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclrtMalloc failed. ERROR: %d\n", ret); return ret);
    ret = aclrtMemcpy(*deviceAddr, size, hostData.data(), size, ACL_MEMCPY_HOST_TO_DEVICE);
    if (ret != ACL_SUCCESS) {
        LOG_PRINT("aclrtMemcpy failed. ERROR: %d\n", ret);
        aclrtFree(*deviceAddr);
        *deviceAddr = nullptr;
        return ret;
    }

    std::vector<int64_t> strides(shape.size(), 1);
    for (int64_t i = static_cast<int64_t>(shape.size()) - 2; i >= 0; i--) {
        strides[static_cast<size_t>(i)] = shape[static_cast<size_t>(i + 1)] * strides[static_cast<size_t>(i + 1)];
    }

    *tensor = aclCreateTensor(shape.data(), shape.size(), dataType, strides.data(), 0, aclFormat::ACL_FORMAT_ND,
                              shape.data(), shape.size(), *deviceAddr);
    if (*tensor == nullptr) {
        LOG_PRINT("aclCreateTensor failed.\n");
        aclrtFree(*deviceAddr);
        *deviceAddr = nullptr;
        return ACL_ERROR_INVALID_PARAM;
    }
    return ACL_SUCCESS;
}

void DestroyTensorAndFree(aclTensor*& tensor, void*& deviceAddr)
{
    if (tensor != nullptr) {
        aclDestroyTensor(tensor);
        tensor = nullptr;
    }
    if (deviceAddr != nullptr) {
        aclrtFree(deviceAddr);
        deviceAddr = nullptr;
    }
}

int main()
{
    int32_t deviceId = 0;
    aclrtStream stream = nullptr;
    int ret = Init(deviceId, &stream);
    CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("Init acl failed. ERROR: %d\n", ret); return ret);

    aclTensor* dst = nullptr;
    void* dstDeviceAddr = nullptr;
    aclTensor* dstSize = nullptr;
    void* dstSizeDeviceAddr = nullptr;
    aclTensor* dstStride = nullptr;
    void* dstStrideDeviceAddr = nullptr;
    aclTensor* dstStorageOffset = nullptr;
    void* dstStorageOffsetDeviceAddr = nullptr;
    aclTensor* src = nullptr;
    void* srcDeviceAddr = nullptr;
    aclTensor* srcSize = nullptr;
    void* srcSizeDeviceAddr = nullptr;
    aclTensor* srcStride = nullptr;
    void* srcStrideDeviceAddr = nullptr;
    aclTensor* srcStorageOffset = nullptr;
    void* srcStorageOffsetDeviceAddr = nullptr;
    aclTensor* y = nullptr;
    void* yDeviceAddr = nullptr;
    void* workspaceAddr = nullptr;
    uint64_t workspaceSize = 0;
    aclOpExecutor* executor = nullptr;

    std::vector<int64_t> dstShape = {5};
    std::vector<float> dstHostData(5, 1);
    std::vector<int64_t> dstSizeShape = {1};
    std::vector<int32_t> dstSizeHostData(1, 1);
    std::vector<int64_t> dstStrideShape = {1};
    std::vector<int32_t> dstStrideHostData(1, 1);
    std::vector<int64_t> dstStorageOffsetShape = {1};
    std::vector<int32_t> dstStorageOffsetHostData(1, 1);
    std::vector<int64_t> srcShape = {5};
    std::vector<float> srcHostData(5, 1);
    std::vector<int64_t> srcSizeShape = {1};
    std::vector<int32_t> srcSizeHostData(1, 1);
    std::vector<int64_t> srcStrideShape = {1};
    std::vector<int32_t> srcStrideHostData(1, 1);
    std::vector<int64_t> srcStorageOffsetShape = {1};
    std::vector<int32_t> srcStorageOffsetHostData(1, 1);
    std::vector<int64_t> yShape = {5};
    std::vector<float> yHostData(5, 0);

    ret = CreateAclTensor(dstHostData, dstShape, &dstDeviceAddr, aclDataType::ACL_FLOAT, &dst);
    CHECK_RET(ret == ACL_SUCCESS, goto cleanup);

    ret = CreateAclTensor(dstSizeHostData, dstSizeShape, &dstSizeDeviceAddr, aclDataType::ACL_INT32, &dstSize);
    CHECK_RET(ret == ACL_SUCCESS, goto cleanup);

    ret = CreateAclTensor(dstStrideHostData, dstStrideShape, &dstStrideDeviceAddr, aclDataType::ACL_INT32, &dstStride);
    CHECK_RET(ret == ACL_SUCCESS, goto cleanup);

    ret = CreateAclTensor(dstStorageOffsetHostData, dstStorageOffsetShape, &dstStorageOffsetDeviceAddr,
                          aclDataType::ACL_INT32, &dstStorageOffset);
    CHECK_RET(ret == ACL_SUCCESS, goto cleanup);

    ret = CreateAclTensor(srcHostData, srcShape, &srcDeviceAddr, aclDataType::ACL_FLOAT, &src);
    CHECK_RET(ret == ACL_SUCCESS, goto cleanup);

    ret = CreateAclTensor(srcSizeHostData, srcSizeShape, &srcSizeDeviceAddr, aclDataType::ACL_INT32, &srcSize);
    CHECK_RET(ret == ACL_SUCCESS, goto cleanup);

    ret = CreateAclTensor(srcStrideHostData, srcStrideShape, &srcStrideDeviceAddr, aclDataType::ACL_INT32, &srcStride);
    CHECK_RET(ret == ACL_SUCCESS, goto cleanup);

    ret = CreateAclTensor(srcStorageOffsetHostData, srcStorageOffsetShape, &srcStorageOffsetDeviceAddr,
                          aclDataType::ACL_INT32, &srcStorageOffset);
    CHECK_RET(ret == ACL_SUCCESS, goto cleanup);

    ret = CreateAclTensor(yHostData, yShape, &yDeviceAddr, aclDataType::ACL_FLOAT, &y);
    CHECK_RET(ret == ACL_SUCCESS, goto cleanup);

    ret = aclnnViewCopyGetWorkspaceSize(dst, dstSize, dstStride, dstStorageOffset, src, srcSize, srcStride,
                                        srcStorageOffset, y, &workspaceSize, &executor);
    CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclnnViewCopyGetWorkspaceSize failed. ERROR: %d\n", ret); goto cleanup);

    if (workspaceSize > 0) {
        ret = aclrtMalloc(&workspaceAddr, workspaceSize, ACL_MEM_MALLOC_HUGE_FIRST);
        CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("allocate workspace failed. ERROR: %d\n", ret); goto cleanup);
    }

    ret = aclnnViewCopy(workspaceAddr, workspaceSize, executor, stream);
    CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclnnViewCopy failed. ERROR: %d\n", ret); goto cleanup);

    ret = aclrtSynchronizeStream(stream);
    CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclrtSynchronizeStream failed. ERROR: %d\n", ret); goto cleanup);

cleanup:
    if (workspaceAddr != nullptr) {
        aclrtFree(workspaceAddr);
        workspaceAddr = nullptr;
    }
    DestroyTensorAndFree(y, yDeviceAddr);
    DestroyTensorAndFree(srcStorageOffset, srcStorageOffsetDeviceAddr);
    DestroyTensorAndFree(srcStride, srcStrideDeviceAddr);
    DestroyTensorAndFree(srcSize, srcSizeDeviceAddr);
    DestroyTensorAndFree(src, srcDeviceAddr);
    DestroyTensorAndFree(dstStorageOffset, dstStorageOffsetDeviceAddr);
    DestroyTensorAndFree(dstStride, dstStrideDeviceAddr);
    DestroyTensorAndFree(dstSize, dstSizeDeviceAddr);
    DestroyTensorAndFree(dst, dstDeviceAddr);
    if (stream != nullptr) {
        aclrtDestroyStream(stream);
    }
    aclrtResetDevice(deviceId);
    aclFinalize();

    return ret;
}
