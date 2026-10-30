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
 * NOTE: Portions of this code were AI-generated and have been
 * technically reviewed for functional accuracy and security
 */

/**
 * @file test_aclnn_real_v2.cpp
 * @brief ACLNN invocation example for RealV2 operator
 *
 * Demonstrates the two-phase ACLNN call pattern:
 *   1. aclnnRealV2GetWorkspaceSize() -- get workspace size & executor
 *   2. aclnnRealV2()                 -- execute on NPU
 *
 * Tested scenarios:
 *   - FLOAT passthrough (real input -> same output)
 *   - COMPLEX64 real-part extraction
 *
 * Build & Run:
 *   cd example && bash run.sh --eager
 */

#include <algorithm>
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <vector>

#include "acl/acl.h"
#include "aclnn/aclnn_base.h"
#include "aclnn/acl_meta.h"
#include "aclnn_real_v2.h"

// ---------------------------------------------------------------------------
// Error checking macro
// ---------------------------------------------------------------------------
#define CHECK_ACL(expr)                                                                                 \
    do {                                                                                                \
        auto __ret = (expr);                                                                            \
        int32_t __code = static_cast<int32_t>(__ret);                                                   \
        if (__code != 0) {                                                                              \
            fprintf(stderr, "[ERROR] %s failed at %s:%d, ret=%d\n", #expr, __FILE__, __LINE__, __code); \
            const char* msg = aclGetRecentErrMsg();                                                     \
            if (msg)                                                                                    \
                fprintf(stderr, "  Detail: %s\n", msg);                                                 \
            return 1;                                                                                   \
        }                                                                                               \
    } while (0)

// ---------------------------------------------------------------------------
// Helper: compute contiguous strides from shape
// ---------------------------------------------------------------------------
static std::vector<int64_t> ComputeStrides(const std::vector<int64_t>& shape)
{
    std::vector<int64_t> strides(shape.size(), 1);
    for (int64_t i = static_cast<int64_t>(shape.size()) - 2; i >= 0; --i) {
        strides[i] = shape[i + 1] * strides[i + 1];
    }
    return strides;
}

// ---------------------------------------------------------------------------
// Helper: create aclTensor from host data
// ---------------------------------------------------------------------------
static int CreateAclTensor(
    const void* hostData, size_t dataBytes, const std::vector<int64_t>& shape, void** deviceAddr, aclDataType dataType,
    aclTensor** tensor)
{
    auto ret = aclrtMalloc(deviceAddr, dataBytes, ACL_MEM_MALLOC_HUGE_FIRST);
    if (ret != ACL_SUCCESS)
        return ret;
    ret = aclrtMemcpy(*deviceAddr, dataBytes, hostData, dataBytes, ACL_MEMCPY_HOST_TO_DEVICE);
    if (ret != ACL_SUCCESS) {
        aclrtFree(*deviceAddr);
        return ret;
    }
    auto strides = ComputeStrides(shape);
    *tensor = aclCreateTensor(
        shape.data(), shape.size(), dataType, strides.data(), 0, aclFormat::ACL_FORMAT_ND, shape.data(), shape.size(),
        *deviceAddr);
    return ACL_SUCCESS;
}

// ---------------------------------------------------------------------------
// Test 1: FLOAT passthrough
//   Input:  [1.0, -2.5, 3.14, 0.0]  (float32)
//   Expect: [1.0, -2.5, 3.14, 0.0]  (float32, identical)
// ---------------------------------------------------------------------------
static int TestFloatPassthrough(aclrtStream stream)
{
    printf("\n--- Test 1: FLOAT passthrough ---\n");

    const std::vector<int64_t> shape = {4};
    const int64_t numElements = 4;
    const size_t bufferSize = numElements * sizeof(float);

    std::vector<float> inputHost = {1.0f, -2.5f, 3.14f, 0.0f};
    std::vector<float> goldenHost = {1.0f, -2.5f, 3.14f, 0.0f};
    std::vector<float> outputHost(numElements, 0.0f);

    // Create input tensor
    void* selfDev = nullptr;
    aclTensor* selfTensor = nullptr;
    CHECK_ACL(CreateAclTensor(inputHost.data(), bufferSize, shape, &selfDev, ACL_FLOAT, &selfTensor));

    // Create output tensor
    void* outDev = nullptr;
    aclTensor* outTensor = nullptr;
    CHECK_ACL(CreateAclTensor(outputHost.data(), bufferSize, shape, &outDev, ACL_FLOAT, &outTensor));

    // Phase 1: GetWorkspaceSize
    uint64_t workspaceSize = 0;
    aclOpExecutor* executor = nullptr;
    CHECK_ACL(aclnnRealV2GetWorkspaceSize(selfTensor, outTensor, &workspaceSize, &executor));
    void* workspace = nullptr;
    if (workspaceSize > 0) {
        CHECK_ACL(aclrtMalloc(&workspace, workspaceSize, ACL_MEM_MALLOC_HUGE_FIRST));
    }

    // Phase 2: Execute
    CHECK_ACL(aclnnRealV2(workspace, workspaceSize, executor, stream));
    CHECK_ACL(aclrtSynchronizeStream(stream));

    // Copy result back
    CHECK_ACL(aclrtMemcpy(outputHost.data(), bufferSize, outDev, bufferSize, ACL_MEMCPY_DEVICE_TO_HOST));

    // Verify bitwise match
    bool pass = (std::memcmp(outputHost.data(), goldenHost.data(), bufferSize) == 0);
    printf("Output: ");
    for (int i = 0; i < numElements; ++i)
        printf("%.4f ", outputHost[i]);
    printf("\nResult: %s\n", pass ? "PASS" : "FAIL");

    // Cleanup
    aclDestroyTensor(selfTensor);
    aclDestroyTensor(outTensor);
    aclrtFree(selfDev);
    aclrtFree(outDev);
    if (workspace)
        aclrtFree(workspace);

    return pass ? 0 : 1;
}

// ---------------------------------------------------------------------------
// Test 2: COMPLEX64 real-part extraction
//   Input (complex64):  [(1.0+10.0j), (-2.5+20.0j), (3.14+30.0j), (0.0+40.0j)]
//   Memory layout:      [1.0, 10.0, -2.5, 20.0, 3.14, 30.0, 0.0, 40.0]
//   Expect (float32):   [1.0, -2.5, 3.14, 0.0]
// ---------------------------------------------------------------------------
static int TestComplex64Extraction(aclrtStream stream)
{
    printf("\n--- Test 2: COMPLEX64 real-part extraction ---\n");

    const std::vector<int64_t> shape = {4};
    const int64_t numElements = 4;
    const size_t inputBytes = numElements * 2 * sizeof(float); // complex64 = 2 x float32
    const size_t outputBytes = numElements * sizeof(float);

    // complex64 input: [real0, imag0, real1, imag1, ...]
    std::vector<float> inputHost = {1.0f, 10.0f, -2.5f, 20.0f, 3.14f, 30.0f, 0.0f, 40.0f};
    std::vector<float> goldenHost = {1.0f, -2.5f, 3.14f, 0.0f};
    std::vector<float> outputHost(numElements, 0.0f);

    // Create input tensor (COMPLEX64)
    void* selfDev = nullptr;
    aclTensor* selfTensor = nullptr;
    CHECK_ACL(CreateAclTensor(inputHost.data(), inputBytes, shape, &selfDev, ACL_COMPLEX64, &selfTensor));

    // Create output tensor (FLOAT)
    void* outDev = nullptr;
    aclTensor* outTensor = nullptr;
    CHECK_ACL(CreateAclTensor(outputHost.data(), outputBytes, shape, &outDev, ACL_FLOAT, &outTensor));

    // Phase 1: GetWorkspaceSize
    uint64_t workspaceSize = 0;
    aclOpExecutor* executor = nullptr;
    CHECK_ACL(aclnnRealV2GetWorkspaceSize(selfTensor, outTensor, &workspaceSize, &executor));
    void* workspace = nullptr;
    if (workspaceSize > 0) {
        CHECK_ACL(aclrtMalloc(&workspace, workspaceSize, ACL_MEM_MALLOC_HUGE_FIRST));
    }

    // Phase 2: Execute
    CHECK_ACL(aclnnRealV2(workspace, workspaceSize, executor, stream));
    CHECK_ACL(aclrtSynchronizeStream(stream));

    // Copy result back
    CHECK_ACL(aclrtMemcpy(outputHost.data(), outputBytes, outDev, outputBytes, ACL_MEMCPY_DEVICE_TO_HOST));

    // Verify bitwise match
    bool pass = (std::memcmp(outputHost.data(), goldenHost.data(), outputBytes) == 0);
    printf("Output: ");
    for (int i = 0; i < numElements; ++i)
        printf("%.4f ", outputHost[i]);
    printf("\nResult: %s\n", pass ? "PASS" : "FAIL");

    // Cleanup
    aclDestroyTensor(selfTensor);
    aclDestroyTensor(outTensor);
    aclrtFree(selfDev);
    aclrtFree(outDev);
    if (workspace)
        aclrtFree(workspace);

    return pass ? 0 : 1;
}

// ---------------------------------------------------------------------------
// Main
// ---------------------------------------------------------------------------
int32_t main(int32_t argc, char** argv)
{
    printf("========================================\n");
    printf("RealV2 operator ACLNN invocation example\n");
    printf("========================================\n");

    const int32_t deviceId = 0;
    aclrtStream stream = nullptr;

    CHECK_ACL(aclInit(nullptr));
    CHECK_ACL(aclrtSetDevice(deviceId));
    CHECK_ACL(aclrtCreateStream(&stream));

    int failures = 0;
    failures += TestFloatPassthrough(stream);
    failures += TestComplex64Extraction(stream);

    printf("\n========================================\n");
    printf("Overall: %s (%d test(s) failed)\n", failures == 0 ? "PASS" : "FAIL", failures);
    printf("========================================\n");

    CHECK_ACL(aclrtDestroyStream(stream));
    CHECK_ACL(aclrtResetDevice(deviceId));
    CHECK_ACL(aclFinalize());

    return failures == 0 ? 0 : 1;
}
