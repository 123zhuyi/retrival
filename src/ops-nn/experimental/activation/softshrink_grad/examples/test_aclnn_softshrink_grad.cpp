/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include <cmath>
#include <cstdio>
#include <vector>

#include "acl/acl.h"
#include "aclnnop/aclnn_softshrink_backward.h"

#define CHECK_RET(condition, action) \
    do {                             \
        if (!(condition)) {          \
            action;                  \
        }                            \
    } while (0)

int64_t GetShapeSize(const std::vector<int64_t>& shape)
{
    int64_t size = 1;
    for (const int64_t dim : shape) {
        size *= dim;
    }
    return size;
}

int InitDevice(int32_t deviceId, aclrtStream* stream)
{
    aclError ret = aclInit(nullptr);
    CHECK_RET(ret == ACL_SUCCESS, std::printf("aclInit failed: %d\n", ret); return ret);

    ret = aclrtSetDevice(deviceId);
    if (ret != ACL_SUCCESS) {
        std::printf("aclrtSetDevice failed: %d\n", ret);
        aclFinalize();
        return ret;
    }

    ret = aclrtCreateStream(stream);
    if (ret != ACL_SUCCESS) {
        std::printf("aclrtCreateStream failed: %d\n", ret);
        aclrtResetDevice(deviceId);
        aclFinalize();
        return ret;
    }
    return ACL_SUCCESS;
}

template <typename T>
int CreateAclTensor(const std::vector<T>& hostData, const std::vector<int64_t>& shape, aclDataType dataType,
                    void** deviceAddress, aclTensor** tensor)
{
    const size_t dataSize = static_cast<size_t>(GetShapeSize(shape)) * sizeof(T);
    aclError ret = aclrtMalloc(deviceAddress, dataSize, ACL_MEM_MALLOC_HUGE_FIRST);
    CHECK_RET(ret == ACL_SUCCESS, std::printf("aclrtMalloc failed: %d\n", ret); return ret);

    ret = aclrtMemcpy(*deviceAddress, dataSize, hostData.data(), dataSize, ACL_MEMCPY_HOST_TO_DEVICE);
    if (ret != ACL_SUCCESS) {
        std::printf("aclrtMemcpy failed: %d\n", ret);
        aclrtFree(*deviceAddress);
        *deviceAddress = nullptr;
        return ret;
    }

    std::vector<int64_t> strides(shape.size(), 1);
    for (int64_t i = static_cast<int64_t>(shape.size()) - 2; i >= 0; --i) {
        strides[static_cast<size_t>(i)] = shape[static_cast<size_t>(i + 1)] * strides[static_cast<size_t>(i + 1)];
    }

    *tensor = aclCreateTensor(shape.data(), shape.size(), dataType, strides.data(), 0, ACL_FORMAT_ND, shape.data(),
                              shape.size(), *deviceAddress);
    if (*tensor == nullptr) {
        std::printf("aclCreateTensor failed\n");
        aclrtFree(*deviceAddress);
        *deviceAddress = nullptr;
        return ACL_ERROR_FAILURE;
    }
    return ACL_SUCCESS;
}

int main()
{
    constexpr int32_t deviceId = 0;
    aclrtStream stream = nullptr;
    int ret = InitDevice(deviceId, &stream);
    CHECK_RET(ret == ACL_SUCCESS, return ret);

    const std::vector<int64_t> shape = {8};
    const std::vector<float> gradOutputHost = {1.0F, 2.0F, 3.0F, 4.0F, 5.0F, 6.0F, 7.0F, 8.0F};
    const std::vector<float> selfHost = {-1.0F, -0.5F, -0.4F, 0.0F, 0.5F, 0.6F, 2.0F, -2.0F};
    const std::vector<float> expected = {1.0F, 0.0F, 0.0F, 0.0F, 0.0F, 6.0F, 7.0F, 8.0F};
    const std::vector<float> outputHost(shape[0], 0.0F);
    float lambdValue = 0.5F;

    void* gradOutputAddress = nullptr;
    void* selfAddress = nullptr;
    void* gradInputAddress = nullptr;
    aclTensor* gradOutput = nullptr;
    aclTensor* self = nullptr;
    aclTensor* gradInput = nullptr;
    aclScalar* lambd = nullptr;
    void* workspace = nullptr;

    auto cleanup = [&]() {
        if (gradOutput != nullptr) {
            aclDestroyTensor(gradOutput);
        }
        if (self != nullptr) {
            aclDestroyTensor(self);
        }
        if (gradInput != nullptr) {
            aclDestroyTensor(gradInput);
        }
        if (lambd != nullptr) {
            aclDestroyScalar(lambd);
        }
        if (gradOutputAddress != nullptr) {
            aclrtFree(gradOutputAddress);
        }
        if (selfAddress != nullptr) {
            aclrtFree(selfAddress);
        }
        if (gradInputAddress != nullptr) {
            aclrtFree(gradInputAddress);
        }
        if (workspace != nullptr) {
            aclrtFree(workspace);
        }
        aclrtDestroyStream(stream);
        aclrtResetDevice(deviceId);
        aclFinalize();
    };

    ret = CreateAclTensor(gradOutputHost, shape, ACL_FLOAT, &gradOutputAddress, &gradOutput);
    CHECK_RET(ret == ACL_SUCCESS, cleanup(); return ret);
    ret = CreateAclTensor(selfHost, shape, ACL_FLOAT, &selfAddress, &self);
    CHECK_RET(ret == ACL_SUCCESS, cleanup(); return ret);
    ret = CreateAclTensor(outputHost, shape, ACL_FLOAT, &gradInputAddress, &gradInput);
    CHECK_RET(ret == ACL_SUCCESS, cleanup(); return ret);

    lambd = aclCreateScalar(&lambdValue, ACL_FLOAT);
    CHECK_RET(lambd != nullptr, std::printf("aclCreateScalar failed\n"); cleanup(); return ACL_ERROR_FAILURE);

    uint64_t workspaceSize = 0;
    aclOpExecutor* executor = nullptr;
    ret = aclnnSoftshrinkBackwardGetWorkspaceSize(gradOutput, self, lambd, gradInput, &workspaceSize, &executor);
    CHECK_RET(ret == ACL_SUCCESS, std::printf("aclnnSoftshrinkBackwardGetWorkspaceSize failed: %d\n", ret); cleanup();
              return ret);

    if (workspaceSize > 0) {
        ret = aclrtMalloc(&workspace, workspaceSize, ACL_MEM_MALLOC_HUGE_FIRST);
        CHECK_RET(ret == ACL_SUCCESS, std::printf("workspace allocation failed: %d\n", ret); cleanup(); return ret);
    }

    ret = aclnnSoftshrinkBackward(workspace, workspaceSize, executor, stream);
    CHECK_RET(ret == ACL_SUCCESS, std::printf("aclnnSoftshrinkBackward failed: %d\n", ret); cleanup(); return ret);
    ret = aclrtSynchronizeStream(stream);
    CHECK_RET(ret == ACL_SUCCESS, std::printf("aclrtSynchronizeStream failed: %d\n", ret); cleanup(); return ret);

    std::vector<float> actual(expected.size(), 0.0F);
    ret = aclrtMemcpy(actual.data(), actual.size() * sizeof(float), gradInputAddress, actual.size() * sizeof(float),
                      ACL_MEMCPY_DEVICE_TO_HOST);
    CHECK_RET(ret == ACL_SUCCESS, std::printf("copy output failed: %d\n", ret); cleanup(); return ret);

    bool passed = true;
    for (size_t i = 0; i < actual.size(); ++i) {
        std::printf("output[%zu]=%.6f, expected=%.6f\n", i, actual[i], expected[i]);
        if (std::fabs(actual[i] - expected[i]) > 1.0e-6F) {
            passed = false;
        }
    }

    cleanup();
    std::printf("SoftShrinkGrad ACLNN example: %s\n", passed ? "PASS" : "FAIL");
    return passed ? 0 : 1;
}
