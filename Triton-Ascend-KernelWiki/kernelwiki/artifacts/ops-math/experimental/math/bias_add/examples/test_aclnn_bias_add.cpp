/**
 * This file is contributed to the CANN Open Software.
 *
 * Copyright (c) 2026 Yang Zhenze, Chongqing University of Posts and Telecommunications (CQUPT).
 * All Rights Reserved.
 *
 * Author (account):
 * - Yang Zhenze <@gcw_5x5Ew5Ms>
 *
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include <vector>
#include <iostream>
#include "acl/acl.h"
#include "aclnn_bias_add.h"

#define CHECK_RET(cond, expr) \
    do {                      \
        if (!(cond)) {        \
            expr;             \
        }                     \
    } while (0)
#define LOG_PRINT(message, ...)         \
    do {                                \
        printf(message, ##__VA_ARGS__); \
    } while (0)

int64_t GetShapeSize(const std::vector<int64_t>& shape)
{
    int64_t s = 1;
    for (auto d : shape) {
        s *= d;
    }
    return s;
}

template <typename T>
int CreateAclTensor(const std::vector<T>& hostData, const std::vector<int64_t>& shape, void** devAddr,
                    aclDataType dtype, aclTensor** tensor)
{
    auto size = GetShapeSize(shape) * sizeof(T);
    auto ret = aclrtMalloc(devAddr, size, ACL_MEM_MALLOC_HUGE_FIRST);
    CHECK_RET(ret == ACL_SUCCESS, return ret);
    ret = aclrtMemcpy(*devAddr, size, hostData.data(), size, ACL_MEMCPY_HOST_TO_DEVICE);
    CHECK_RET(ret == ACL_SUCCESS, aclrtFree(*devAddr); *devAddr = nullptr; return ret);
    std::vector<int64_t> strides(shape.size(), 1);
    for (int64_t i = shape.size() - 2; i >= 0; i--) {
        strides[i] = shape[i + 1] * strides[i + 1];
    }
    *tensor = aclCreateTensor(shape.data(), shape.size(), dtype, strides.data(), 0, aclFormat::ACL_FORMAT_ND,
                              shape.data(), shape.size(), *devAddr);
    CHECK_RET(*tensor != nullptr, aclrtFree(*devAddr); *devAddr = nullptr; return ACL_ERROR_FAILURE);
    return ACL_SUCCESS;
}

int main()
{
    // Declare every resource handle up front so all error paths can jump to a single
    // CLEAN_UP exit (avoids the device-memory / ACL-tensor leaks that direct returns cause,
    // and keeps the failure return code consistent).
    int32_t deviceId = 0;
    aclrtStream stream = nullptr;
    void *xDev = nullptr, *biasDev = nullptr, *yDev = nullptr;
    aclTensor *x = nullptr, *bias = nullptr, *y = nullptr;
    uint64_t workspaceSize = 0;
    aclOpExecutor* executor = nullptr;
    void* workspaceAddr = nullptr;
    int result = 0;

    // x:[2,3] NHWC(C=3), bias:[3], y:[2,3]
    std::vector<int64_t> xShape = {2, 3};
    std::vector<int64_t> biasShape = {3};
    std::vector<int64_t> yShape = {2, 3};
    std::vector<float> xHost = {1, 2, 3, 4, 5, 6};
    std::vector<float> biasHost = {10, 20, 30};
    std::vector<float> yHost(6, 0);
    char dataFormat[] = "NHWC"; // aclnnBiasAddGetWorkspaceSize 的 data_format 入参为 char*

    auto ret = aclInit(nullptr);
    CHECK_RET(ret == ACL_SUCCESS, return ret);
    ret = aclrtSetDevice(deviceId);
    CHECK_RET(ret == ACL_SUCCESS, aclFinalize(); return ret);
    ret = aclrtCreateStream(&stream);
    CHECK_RET(ret == ACL_SUCCESS, aclrtResetDevice(deviceId); aclFinalize(); return ret);

    // From here on every failure releases all acquired resources via CLEAN_UP.
    ret = CreateAclTensor(xHost, xShape, &xDev, aclDataType::ACL_FLOAT, &x);
    CHECK_RET(ret == ACL_SUCCESS, result = ret; goto CLEAN_UP);
    ret = CreateAclTensor(biasHost, biasShape, &biasDev, aclDataType::ACL_FLOAT, &bias);
    CHECK_RET(ret == ACL_SUCCESS, result = ret; goto CLEAN_UP);
    ret = CreateAclTensor(yHost, yShape, &yDev, aclDataType::ACL_FLOAT, &y);
    CHECK_RET(ret == ACL_SUCCESS, result = ret; goto CLEAN_UP);

    ret = aclnnBiasAddGetWorkspaceSize(x, bias, dataFormat, y, &workspaceSize, &executor);
    CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("GetWorkspaceSize failed. ret=%d\n", ret); result = ret; goto CLEAN_UP);

    if (workspaceSize > 0) {
        ret = aclrtMalloc(&workspaceAddr, workspaceSize, ACL_MEM_MALLOC_HUGE_FIRST);
        CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("alloc workspace failed. ret=%d\n", ret); result = ret; goto CLEAN_UP);
    }
    ret = aclnnBiasAdd(workspaceAddr, workspaceSize, executor, stream);
    CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclnnBiasAdd failed. ret=%d\n", ret); result = ret; goto CLEAN_UP);
    ret = aclrtSynchronizeStream(stream);
    CHECK_RET(ret == ACL_SUCCESS, result = ret; goto CLEAN_UP);

    ret = aclrtMemcpy(yHost.data(), yHost.size() * sizeof(float), yDev, yHost.size() * sizeof(float),
                      ACL_MEMCPY_DEVICE_TO_HOST);
    CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("copy result back failed. ret=%d\n", ret); result = ret; goto CLEAN_UP);
    for (size_t i = 0; i < yHost.size(); i++) {
        LOG_PRINT("y[%zu] = %f\n", i, yHost[i]);
    }

CLEAN_UP:
    if (x != nullptr) {
        aclDestroyTensor(x);
    }
    if (bias != nullptr) {
        aclDestroyTensor(bias);
    }
    if (y != nullptr) {
        aclDestroyTensor(y);
    }
    if (xDev != nullptr) {
        aclrtFree(xDev);
    }
    if (biasDev != nullptr) {
        aclrtFree(biasDev);
    }
    if (yDev != nullptr) {
        aclrtFree(yDev);
    }
    if (workspaceAddr != nullptr) {
        aclrtFree(workspaceAddr);
    }
    if (stream != nullptr) {
        aclrtDestroyStream(stream);
    }
    aclrtResetDevice(deviceId);
    aclFinalize();
    return result;
}
