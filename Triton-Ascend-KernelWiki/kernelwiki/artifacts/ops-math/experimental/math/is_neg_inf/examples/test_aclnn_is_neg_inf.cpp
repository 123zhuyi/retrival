/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <limits>
#include <vector>

#include "acl/acl.h"
#include "aclnn_is_neg_inf.h"

namespace {

int64_t GetShapeSize(const std::vector<int64_t> &shape)
{
    int64_t size = 1;
    for (int64_t dim : shape) {
        size *= dim;
    }
    return size;
}

std::vector<int64_t> MakeStrides(const std::vector<int64_t> &shape)
{
    if (shape.empty()) {
        return {};
    }
    std::vector<int64_t> strides(shape.size(), 1);
    for (int64_t i = static_cast<int64_t>(shape.size()) - 2; i >= 0; --i) {
        strides[static_cast<size_t>(i)] = shape[static_cast<size_t>(i + 1)] * strides[static_cast<size_t>(i + 1)];
    }
    return strides;
}

aclError CreateAclTensor(
    const std::vector<int64_t> &shape, aclDataType dtype, void *deviceAddr, aclTensor **tensor)
{
    std::vector<int64_t> strides = MakeStrides(shape);
    const int64_t *shapePtr = shape.empty() ? nullptr : shape.data();
    const int64_t *stridesPtr = strides.empty() ? nullptr : strides.data();
    *tensor = aclCreateTensor(
        shapePtr, shape.size(), dtype, stridesPtr, 0, ACL_FORMAT_ND, shapePtr, shape.size(), deviceAddr);
    return *tensor == nullptr ? ACL_ERROR_FAILURE : ACL_SUCCESS;
}

}  // namespace

int main()
{
    aclError ret = ACL_SUCCESS;
    constexpr int32_t kDeviceId = 0;
    int exitCode = 0;
    bool aclInitialized = false;
    bool deviceSet = false;
    aclrtStream stream = nullptr;
    void *input0Device = nullptr;
    void *outputDevice = nullptr;
    void *workspace = nullptr;
    aclTensor *inputTensor = nullptr;
    aclTensor *outputTensor = nullptr;
    aclOpExecutor *executor = nullptr;
    uint64_t workspaceSize = 0;

    const std::vector<int64_t> shape = {6};
    const float negInf = -std::numeric_limits<float>::infinity();
    const std::vector<float> inputHost = {negInf, -2.0f, 0.0f, 3.0f, negInf, 5.0f};
    const std::vector<uint8_t> expected = {1, 0, 0, 0, 1, 0};
    std::vector<uint8_t> outputHost(expected.size(), 0);
    const size_t inputBytes = inputHost.size() * sizeof(float);
    const size_t outputBytes = outputHost.size() * sizeof(uint8_t);

    ret = aclInit(nullptr);
    if (ret != ACL_SUCCESS) {
        return ret;
    }
    aclInitialized = true;

    ret = aclrtSetDevice(kDeviceId);
    if (ret != ACL_SUCCESS) {
        exitCode = ret;
        goto CLEANUP;
    }
    deviceSet = true;

    ret = aclrtCreateStream(&stream);
    if (ret != ACL_SUCCESS) {
        exitCode = ret;
        goto CLEANUP;
    }

    ret = aclrtMalloc(&input0Device, inputBytes, ACL_MEM_MALLOC_HUGE_FIRST);
    if (ret != ACL_SUCCESS) {
        exitCode = ret;
        goto CLEANUP;
    }
    ret = aclrtMalloc(&outputDevice, outputBytes, ACL_MEM_MALLOC_HUGE_FIRST);
    if (ret != ACL_SUCCESS) {
        exitCode = ret;
        goto CLEANUP;
    }

    ret = aclrtMemcpy(input0Device, inputBytes, inputHost.data(), inputBytes, ACL_MEMCPY_HOST_TO_DEVICE);
    if (ret != ACL_SUCCESS) {
        exitCode = ret;
        goto CLEANUP;
    }
    ret = aclrtMemset(outputDevice, outputBytes, 0, outputBytes);
    if (ret != ACL_SUCCESS) {
        exitCode = ret;
        goto CLEANUP;
    }

    ret = CreateAclTensor(shape, ACL_FLOAT, input0Device, &inputTensor);
    if (ret != ACL_SUCCESS) {
        exitCode = ret;
        goto CLEANUP;
    }
    ret = CreateAclTensor(shape, ACL_BOOL, outputDevice, &outputTensor);
    if (ret != ACL_SUCCESS) {
        exitCode = ret;
        goto CLEANUP;
    }

    ret = aclnnIsNegInfGetWorkspaceSize(inputTensor, outputTensor, &workspaceSize, &executor);
    if (ret != ACL_SUCCESS) {
        exitCode = ret;
        goto CLEANUP;
    }

    if (workspaceSize > 0) {
        ret = aclrtMalloc(&workspace, workspaceSize, ACL_MEM_MALLOC_HUGE_FIRST);
        if (ret != ACL_SUCCESS) {
            exitCode = ret;
            goto CLEANUP;
        }
    }

    ret = aclnnIsNegInf(workspace, workspaceSize, executor, stream);
    if (ret != ACL_SUCCESS) {
        exitCode = ret;
        goto CLEANUP;
    }

    ret = aclrtSynchronizeStream(stream);
    if (ret != ACL_SUCCESS) {
        exitCode = ret;
        goto CLEANUP;
    }

    ret = aclrtMemcpy(outputHost.data(), outputBytes, outputDevice, outputBytes, ACL_MEMCPY_DEVICE_TO_HOST);
    if (ret != ACL_SUCCESS) {
        exitCode = ret;
        goto CLEANUP;
    }

    if (std::memcmp(outputHost.data(), expected.data(), outputBytes) != 0) {
        std::printf("is_neg_inf output mismatch\n");
        exitCode = 1;
        goto CLEANUP;
    }
    std::printf("is_neg_inf example passed\n");

CLEANUP:
    if (inputTensor != nullptr) {
        aclDestroyTensor(inputTensor);
    }
    if (outputTensor != nullptr) {
        aclDestroyTensor(outputTensor);
    }
    if (workspace != nullptr) {
        aclrtFree(workspace);
    }
    if (input0Device != nullptr) {
        aclrtFree(input0Device);
    }
    if (outputDevice != nullptr) {
        aclrtFree(outputDevice);
    }
    if (stream != nullptr) {
        aclrtDestroyStream(stream);
    }
    if (deviceSet) {
        aclrtResetDevice(kDeviceId);
    }
    if (aclInitialized) {
        aclFinalize();
    }
    return exitCode;
}
