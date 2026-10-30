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
 * \file test_aclnn_as_strided.cpp
 * \brief
 */

#include <iostream>
#include <vector>
#include "acl/acl.h"
#include "aclnn_as_strided.h"

#define CHECK_RET(cond, expr) \
    do {                      \
        if (!(cond)) {        \
            expr;             \
        }                     \
    } while (0)

#define LOG_PRINT(fmt, ...) printf(fmt, ##__VA_ARGS__)

int64_t GetShapeSize(const std::vector<int64_t>& shape)
{
    int64_t size = 1;
    for (auto v : shape) {
        size *= v;
    }
    return size;
}

int Init(int32_t deviceId, aclrtStream* stream)
{
    auto ret = aclInit(nullptr);
    CHECK_RET(ret == ACL_SUCCESS, return ret);
    ret = aclrtSetDevice(deviceId);
    CHECK_RET(ret == ACL_SUCCESS, return ret);
    ret = aclrtCreateStream(stream);
    CHECK_RET(ret == ACL_SUCCESS, return ret);
    return ACL_SUCCESS;
}

template <typename T>
int CreateAclTensor(const std::vector<T>& hostData, const std::vector<int64_t>& shape, void** deviceAddr,
                    aclDataType dataType, aclTensor** tensor)
{
    size_t bytes = GetShapeSize(shape) * sizeof(T);
    auto ret = aclrtMalloc(deviceAddr, bytes, ACL_MEM_MALLOC_HUGE_FIRST);
    CHECK_RET(ret == ACL_SUCCESS, return ret);

    ret = aclrtMemcpy(*deviceAddr, bytes, hostData.data(), bytes, ACL_MEMCPY_HOST_TO_DEVICE);
    CHECK_RET(ret == ACL_SUCCESS, return ret);

    std::vector<int64_t> strides(shape.size(), 1);
    for (int64_t i = static_cast<int64_t>(shape.size()) - 2; i >= 0; --i) {
        strides[i] = shape[i + 1] * strides[i + 1];
    }

    *tensor = aclCreateTensor(shape.data(), shape.size(), dataType, strides.data(), 0, ACL_FORMAT_ND, shape.data(),
                              shape.size(), *deviceAddr);
    CHECK_RET(*tensor != nullptr, return ACL_ERROR_FAILURE);
    return ACL_SUCCESS;
}

int main()
{
    int32_t deviceId = 0;
    aclrtStream stream = nullptr;
    auto ret = Init(deviceId, &stream);
    CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("Init failed: %d\n", ret);
              LOG_PRINT("recent err: %s\n", aclGetRecentErrMsg()); return ret);

    aclTensor* x = nullptr;
    void* xDev = nullptr;
    std::vector<int64_t> xShape = {10};
    std::vector<int32_t> xHost = {0, 1, 2, 3, 4, 5, 6, 7, 8, 9};
    ret = CreateAclTensor(xHost, xShape, &xDev, ACL_INT32, &x);
    CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("Create x failed: %d\n", ret);
              LOG_PRINT("recent err: %s\n", aclGetRecentErrMsg()); return ret);

    aclTensor* out = nullptr;
    void* outDev = nullptr;
    std::vector<int64_t> outShape = {4};
    std::vector<int32_t> outHost(4, 0);
    ret = CreateAclTensor(outHost, outShape, &outDev, ACL_INT32, &out);
    CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("Create out failed: %d\n", ret);
              LOG_PRINT("recent err: %s\n", aclGetRecentErrMsg()); return ret);

    std::vector<int64_t> sizeHost = {4};
    std::vector<int64_t> strideHost = {2};
    std::vector<int64_t> storageOffsetHost = {1};
    aclIntArray* size = aclCreateIntArray(sizeHost.data(), sizeHost.size());
    aclIntArray* stride = aclCreateIntArray(strideHost.data(), strideHost.size());
    aclIntArray* storageOffset = aclCreateIntArray(storageOffsetHost.data(), storageOffsetHost.size());
    CHECK_RET(size != nullptr && stride != nullptr && storageOffset != nullptr,
              LOG_PRINT("Create aclIntArray failed\n");
              return ACL_ERROR_FAILURE);

    uint64_t workspaceSize = 0;
    aclOpExecutor* executor = nullptr;

    ret = aclnnAsStridedGetWorkspaceSize(x, size, stride, storageOffset, out, &workspaceSize, &executor);
    CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("GetWorkspaceSize failed: %d\n", ret);
              LOG_PRINT("recent err: %s\n", aclGetRecentErrMsg()); return ret);

    void* workspace = nullptr;
    if (workspaceSize > 0) {
        ret = aclrtMalloc(&workspace, workspaceSize, ACL_MEM_MALLOC_HUGE_FIRST);
        CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("Malloc workspace failed: %d\n", ret);
                  LOG_PRINT("recent err: %s\n", aclGetRecentErrMsg()); return ret);
    }

    ret = aclnnAsStrided(workspace, workspaceSize, executor, stream);
    CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclnnAsStrided failed: %d\n", ret);
              LOG_PRINT("recent err: %s\n", aclGetRecentErrMsg()); return ret);

    ret = aclrtSynchronizeStream(stream);
    CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("Sync failed: %d\n", ret);
              LOG_PRINT("recent err: %s\n", aclGetRecentErrMsg()); return ret);

    std::vector<int32_t> result(4, 0);
    ret = aclrtMemcpy(result.data(), result.size() * sizeof(int32_t), outDev, result.size() * sizeof(int32_t),
                      ACL_MEMCPY_DEVICE_TO_HOST);
    CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("Memcpy D2H failed: %d\n", ret);
              LOG_PRINT("recent err: %s\n", aclGetRecentErrMsg()); return ret);

    LOG_PRINT("result = [%d, %d, %d, %d]\n", result[0], result[1], result[2], result[3]);
    LOG_PRINT("expect = [1, 3, 5, 7]\n");
    CHECK_RET(result[0] == 1 && result[1] == 3 && result[2] == 5 && result[3] == 7, LOG_PRINT("unexpected result\n");
              return ACL_ERROR_FAILURE);

    aclDestroyIntArray(size);
    aclDestroyIntArray(stride);
    aclDestroyIntArray(storageOffset);
    aclDestroyTensor(x);
    aclDestroyTensor(out);
    aclrtFree(xDev);
    aclrtFree(outDev);
    if (workspace != nullptr) {
        aclrtFree(workspace);
    }
    aclrtDestroyStream(stream);
    aclrtResetDevice(deviceId);
    aclFinalize();
    return 0;
}
