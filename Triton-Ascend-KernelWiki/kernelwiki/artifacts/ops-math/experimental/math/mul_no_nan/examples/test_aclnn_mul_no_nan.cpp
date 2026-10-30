/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

/**
 *
 * NOTE: Portions of this code were AI-generated and have been
 * technically reviewed for functional accuracy and security
 */

/**
 * @file test_aclnn_mul_no_nan.cpp
 * @brief MulNoNan ACLNN two-phase invocation example
 *
 * Input data (FP32, shape [2,3]):
 *   x = [[1, 2, 3], [4, 5, 6]]
 *   y = [[2, 0, 1], [0, 3, 0]]
 * Expected output z:
 *   z = [[2, 0, 3], [0, 15, 0]]
 *
 * Core semantics: z = (y == 0) ? 0 : x * y
 */

#include <iostream>
#include <vector>
#include <cmath>
#include <cstring>
#include "acl/acl.h"
#include "../op_api/aclnn_mul_no_nan.h"

#define CHECK_ACL(expr)                                                                                          \
    do {                                                                                                         \
        auto _ret = (expr);                                                                                      \
        if (_ret != ACL_SUCCESS) {                                                                               \
            std::cerr << "[ERROR] " << #expr << " failed, ret=" << _ret << " at " << __FILE__ << ":" << __LINE__ \
                      << std::endl;                                                                              \
            return 1;                                                                                            \
        }                                                                                                        \
    } while (0)

int CreateAclTensor(
    const void* hostData, size_t dataSize, const std::vector<int64_t>& shape, const std::vector<int64_t>& strides,
    void** deviceAddr, aclDataType dataType, aclTensor** tensor)
{
    CHECK_ACL(aclrtMalloc(deviceAddr, dataSize, ACL_MEM_MALLOC_HUGE_FIRST));
    CHECK_ACL(aclrtMemcpy(*deviceAddr, dataSize, hostData, dataSize, ACL_MEMCPY_HOST_TO_DEVICE));

    *tensor = aclCreateTensor(
        shape.data(), shape.size(), dataType, strides.data(), 0, aclFormat::ACL_FORMAT_ND, shape.data(), shape.size(),
        *deviceAddr);
    if (*tensor == nullptr) {
        std::cerr << "[ERROR] aclCreateTensor returned nullptr" << std::endl;
        return 1;
    }
    return 0;
}

int main()
{
    std::cout << "=== MulNoNan ACLNN Example ===" << std::endl;

    CHECK_ACL(aclInit(nullptr));
    int32_t deviceId = 0;
    CHECK_ACL(aclrtSetDevice(deviceId));

    aclrtStream stream = nullptr;
    CHECK_ACL(aclrtCreateStream(&stream));

    const std::vector<int64_t> shape = {2, 3};
    const std::vector<int64_t> strides = {3, 1};
    const int64_t numElements = 2 * 3;
    const size_t dataSize = numElements * sizeof(float);

    std::vector<float> hostX = {1.0f, 2.0f, 3.0f, 4.0f, 5.0f, 6.0f};
    std::vector<float> hostY = {2.0f, 0.0f, 1.0f, 0.0f, 3.0f, 0.0f};
    std::vector<float> expected = {2.0f, 0.0f, 3.0f, 0.0f, 15.0f, 0.0f};
    std::vector<float> hostZ(numElements, -1.0f);

    void* devX = nullptr;
    void* devY = nullptr;
    void* devZ = nullptr;
    aclTensor* tensorX = nullptr;
    aclTensor* tensorY = nullptr;
    aclTensor* tensorZ = nullptr;
    uint64_t workspaceSize = 0;
    aclOpExecutor* executor = nullptr;
    void* workspace = nullptr;
    bool pass = true;
    int ret = 0;

    if (CreateAclTensor(hostX.data(), dataSize, shape, strides, &devX, ACL_FLOAT, &tensorX) != 0) {
        ret = 1;
        goto cleanup;
    }
    if (CreateAclTensor(hostY.data(), dataSize, shape, strides, &devY, ACL_FLOAT, &tensorY) != 0) {
        ret = 1;
        goto cleanup;
    }
    if (CreateAclTensor(hostZ.data(), dataSize, shape, strides, &devZ, ACL_FLOAT, &tensorZ) != 0) {
        ret = 1;
        goto cleanup;
    }

    if (aclnnMulNoNanGetWorkspaceSize(tensorX, tensorY, tensorZ, &workspaceSize, &executor) != 0) {
        std::cerr << "[ERROR] aclnnMulNoNanGetWorkspaceSize failed" << std::endl;
        ret = 1;
        goto cleanup;
    }
    std::cout << "workspaceSize = " << workspaceSize << std::endl;

    if (workspaceSize > 0) {
        if (aclrtMalloc(&workspace, workspaceSize, ACL_MEM_MALLOC_HUGE_FIRST) != ACL_SUCCESS) {
            std::cerr << "[ERROR] aclrtMalloc workspace failed" << std::endl;
            ret = 1;
            goto cleanup;
        }
    }

    if (aclnnMulNoNan(workspace, workspaceSize, executor, stream) != 0) {
        std::cerr << "[ERROR] aclnnMulNoNan failed" << std::endl;
        ret = 1;
        goto cleanup;
    }

    if (aclrtSynchronizeStream(stream) != ACL_SUCCESS) {
        std::cerr << "[ERROR] aclrtSynchronizeStream failed" << std::endl;
        ret = 1;
        goto cleanup;
    }

    if (aclrtMemcpy(hostZ.data(), dataSize, devZ, dataSize, ACL_MEMCPY_DEVICE_TO_HOST) != ACL_SUCCESS) {
        std::cerr << "[ERROR] aclrtMemcpy D2H failed" << std::endl;
        ret = 1;
        goto cleanup;
    }

    std::cout << "Results:" << std::endl;
    for (int64_t i = 0; i < numElements; ++i) {
        bool match = (std::fabs(hostZ[i] - expected[i]) < 1e-6f);
        std::cout << "  z[" << i << "] = " << hostZ[i] << " (expected " << expected[i] << ")"
                  << (match ? "" : "  *** MISMATCH ***") << std::endl;
        if (!match)
            pass = false;
    }

    std::cout << "\nVerification: " << (pass ? "PASS" : "FAIL") << std::endl;
    if (!pass)
        ret = 1;

cleanup:
    if (tensorX)
        aclDestroyTensor(tensorX);
    if (tensorY)
        aclDestroyTensor(tensorY);
    if (tensorZ)
        aclDestroyTensor(tensorZ);

    if (devX)
        aclrtFree(devX);
    if (devY)
        aclrtFree(devY);
    if (devZ)
        aclrtFree(devZ);
    if (workspace)
        aclrtFree(workspace);

    aclrtDestroyStream(stream);
    aclrtResetDevice(deviceId);
    aclFinalize();

    std::cout << "=== Done ===" << std::endl;
    return ret;
}
