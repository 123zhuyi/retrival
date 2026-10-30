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
 * @file test_aclnn_complex_v3.cpp
 * @brief ComplexV3 operator example test (aclnn interface)
 *
 * Combines two real tensors (real, imag) into one complex tensor.
 * Formula: output(i) = real(i) + imag(i) * j
 * Storage: interleaved, out[2*i] = real[i], out[2*i+1] = imag[i]
 *
 * Reference: ops-math/experimental/math/complex_v2/examples/test_aclnn_complex_v2.cpp
 */
#include <iostream>
#include <vector>
#include "acl/acl.h"
#include "aclnn_complex_v3.h"

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
    // Complex output: each element = (real, imag), so underlying float count = 2 * shapeSize
    auto size = GetShapeSize(shape) * 2;
    std::vector<float> resultData(size, 0);
    auto ret = aclrtMemcpy(
        resultData.data(), resultData.size() * sizeof(float), *deviceAddr, size * sizeof(float),
        ACL_MEMCPY_DEVICE_TO_HOST);
    CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("copy result from device to host failed. ERROR: %d\n", ret); return);
    for (int64_t i = 0; i < GetShapeSize(shape); i++) {
        LOG_PRINT("result[%ld] = (%f, %f)\n", i, resultData[2 * i], resultData[2 * i + 1]);
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
    for (int64_t i = shape.size() - 2; i >= 0; i--) {
        strides[i] = shape[i + 1] * strides[i + 1];
    }

    *tensor = aclCreateTensor(
        shape.data(), shape.size(), dataType, strides.data(), 0, aclFormat::ACL_FORMAT_ND, shape.data(), shape.size(),
        *deviceAddr);
    return 0;
}

int main()
{
    // 1. Init device and stream
    int32_t deviceId = 0;
    aclrtStream stream;
    auto ret = Init(deviceId, &stream);
    CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("Init acl failed. ERROR: %d\n", ret); return ret);

    // 2. Create input tensors: real and imag (fp32, shape [2, 2])
    std::vector<int64_t> inputShape = {2, 2};
    int64_t numElements = GetShapeSize(inputShape);

    // real = {1.0, 2.0, 3.0, 4.0}
    aclTensor* realTensor = nullptr;
    void* realDeviceAddr = nullptr;
    std::vector<float> realHostData = {1.0f, 2.0f, 3.0f, 4.0f};
    ret = CreateAclTensor(realHostData, inputShape, &realDeviceAddr, aclDataType::ACL_FLOAT, &realTensor);
    CHECK_RET(ret == ACL_SUCCESS, return ret);

    // imag = {5.0, 6.0, 7.0, 8.0}
    aclTensor* imagTensor = nullptr;
    void* imagDeviceAddr = nullptr;
    std::vector<float> imagHostData = {5.0f, 6.0f, 7.0f, 8.0f};
    ret = CreateAclTensor(imagHostData, inputShape, &imagDeviceAddr, aclDataType::ACL_FLOAT, &imagTensor);
    CHECK_RET(ret == ACL_SUCCESS, return ret);

    // 3. Create output tensor (complex64, shape [2, 2], underlying memory = 2*numElements floats)
    aclTensor* outTensor = nullptr;
    void* outDeviceAddr = nullptr;
    int64_t outBytes = numElements * 2 * sizeof(float); // complex64 = 2 floats per element
    ret = aclrtMalloc(&outDeviceAddr, outBytes, ACL_MEM_MALLOC_HUGE_FIRST);
    CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclrtMalloc for output failed. ERROR: %d\n", ret); return ret);
    ret = aclrtMemset(outDeviceAddr, outBytes, 0, outBytes);
    CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclrtMemset for output failed. ERROR: %d\n", ret); return ret);

    std::vector<int64_t> outStrides(inputShape.size(), 1);
    for (int64_t i = inputShape.size() - 2; i >= 0; i--) {
        outStrides[i] = inputShape[i + 1] * outStrides[i + 1];
    }
    outTensor = aclCreateTensor(
        inputShape.data(), inputShape.size(), aclDataType::ACL_COMPLEX64, outStrides.data(), 0,
        aclFormat::ACL_FORMAT_ND, inputShape.data(), inputShape.size(), outDeviceAddr);

    // 4. Call aclnnComplexV3GetWorkspaceSize
    uint64_t workspaceSize = 0;
    aclOpExecutor* executor;
    ret = aclnnComplexV3GetWorkspaceSize(realTensor, imagTensor, outTensor, &workspaceSize, &executor);
    LOG_PRINT(
        "aclnnComplexV3GetWorkspaceSize returned %d, workspaceSize=%llu, executor=%p\n", ret,
        (unsigned long long)workspaceSize, (void*)executor);
    CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclnnComplexV3GetWorkspaceSize failed. ERROR: %d\n", ret); return ret);

    // Allocate workspace if needed
    void* workspaceAddr = nullptr;
    if (workspaceSize > static_cast<uint64_t>(0)) {
        ret = aclrtMalloc(&workspaceAddr, workspaceSize, ACL_MEM_MALLOC_HUGE_FIRST);
        CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("allocate workspace failed. ERROR: %d\n", ret); return ret);
    }

    // 5. Call aclnnComplexV3
    ret = aclnnComplexV3(workspaceAddr, workspaceSize, executor, stream);
    CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclnnComplexV3 failed. ERROR: %d\n", ret); return ret);

    // 6. Synchronize stream
    ret = aclrtSynchronizeStream(stream);
    CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclrtSynchronizeStream failed. ERROR: %d\n", ret); return ret);

    // 7. Print results
    // Expected: result[0]=(1,5), result[1]=(2,6), result[2]=(3,7), result[3]=(4,8)
    std::vector<int64_t> outShape = inputShape;
    PrintOutResult(outShape, &outDeviceAddr);

    // 8. Destroy tensors
    aclDestroyTensor(realTensor);
    aclDestroyTensor(imagTensor);
    aclDestroyTensor(outTensor);

    // 9. Free device memory
    aclrtFree(realDeviceAddr);
    aclrtFree(imagDeviceAddr);
    aclrtFree(outDeviceAddr);
    if (workspaceSize > static_cast<uint64_t>(0)) {
        aclrtFree(workspaceAddr);
    }
    aclrtDestroyStream(stream);
    aclrtResetDevice(deviceId);

    // 10. Finalize
    aclFinalize();

    return 0;
}
