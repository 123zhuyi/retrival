/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include <cstdint>
#include <cstdio>
#include <vector>

#include "acl/acl.h"
#include "aclnnop/aclnn_right_shift.h"

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

namespace {
constexpr int32_t DEVICE_ID = 0;

int InitAcl(aclrtStream* stream)
{
    auto ret = aclInit(nullptr);
    CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclInit failed. ERROR: %d\n", ret); return ret);
    ret = aclrtSetDevice(DEVICE_ID);
    CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclrtSetDevice failed. ERROR: %d\n", ret); return ret);
    ret = aclrtCreateStream(stream);
    CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclrtCreateStream failed. ERROR: %d\n", ret); return ret);
    return ACL_SUCCESS;
}

aclTensor* CreateInt32Tensor(const std::vector<int32_t>& hostData, const std::vector<int64_t>& shape, void** deviceAddr)
{
    size_t dataBytes = hostData.size() * sizeof(int32_t);
    auto ret = aclrtMalloc(deviceAddr, dataBytes, ACL_MEM_MALLOC_HUGE_FIRST);
    CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclrtMalloc failed. ERROR: %d\n", ret); return nullptr);

    ret = aclrtMemcpy(*deviceAddr, dataBytes, hostData.data(), dataBytes, ACL_MEMCPY_HOST_TO_DEVICE);
    CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclrtMemcpy failed. ERROR: %d\n", ret); return nullptr);

    std::vector<int64_t> strides(shape.size(), 1);
    for (int64_t i = static_cast<int64_t>(shape.size()) - 2; i >= 0; --i) {
        strides[static_cast<size_t>(i)] = shape[static_cast<size_t>(i + 1)] * strides[static_cast<size_t>(i + 1)];
    }

    return aclCreateTensor(shape.data(), shape.size(), ACL_INT32, strides.data(), 0, ACL_FORMAT_ND, shape.data(),
                           shape.size(), *deviceAddr);
}

int RunRightShiftDemo(aclrtStream stream)
{
    const std::vector<int64_t> shape = {2, 4};
    const std::vector<int32_t> input = {-16, -8, -1, 0, 1, 8, 16, 32};
    const std::vector<int32_t> shift = {0, 1, 2, 3, -1, 32, 4, 5};
    const std::vector<int32_t> expect = {-16, -4, -1, 0, 0, 0, 1, 1};
    std::vector<int32_t> output(expect.size(), 0);

    void* inputDeviceAddr = nullptr;
    void* shiftDeviceAddr = nullptr;
    void* outputDeviceAddr = nullptr;
    aclTensor* inputTensor = CreateInt32Tensor(input, shape, &inputDeviceAddr);
    CHECK_RET(inputTensor != nullptr, return ACL_ERROR_FAILURE);
    aclTensor* shiftTensor = CreateInt32Tensor(shift, shape, &shiftDeviceAddr);
    CHECK_RET(shiftTensor != nullptr, return ACL_ERROR_FAILURE);
    aclTensor* outputTensor = CreateInt32Tensor(output, shape, &outputDeviceAddr);
    CHECK_RET(outputTensor != nullptr, return ACL_ERROR_FAILURE);

    uint64_t workspaceSize = 0;
    aclOpExecutor* executor = nullptr;
    auto ret = aclnnRightShiftGetWorkspaceSize(inputTensor, shiftTensor, outputTensor, &workspaceSize, &executor);
    CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclnnRightShiftGetWorkspaceSize failed. ERROR: %d\n", ret); return ret);

    void* workspaceAddr = nullptr;
    if (workspaceSize > 0) {
        ret = aclrtMalloc(&workspaceAddr, workspaceSize, ACL_MEM_MALLOC_HUGE_FIRST);
        CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclrtMalloc workspace failed. ERROR: %d\n", ret); return ret);
    }

    ret = aclnnRightShift(workspaceAddr, workspaceSize, executor, stream);
    CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclnnRightShift failed. ERROR: %d\n", ret); return ret);
    ret = aclrtSynchronizeStream(stream);
    CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclrtSynchronizeStream failed. ERROR: %d\n", ret); return ret);

    size_t outputBytes = output.size() * sizeof(int32_t);
    ret = aclrtMemcpy(output.data(), outputBytes, outputDeviceAddr, outputBytes, ACL_MEMCPY_DEVICE_TO_HOST);
    CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("copy output failed. ERROR: %d\n", ret); return ret);

    bool passed = true;
    LOG_PRINT("input :");
    for (auto value : input) {
        LOG_PRINT(" %d", value);
    }
    LOG_PRINT("\nshift :");
    for (auto value : shift) {
        LOG_PRINT(" %d", value);
    }
    LOG_PRINT("\noutput:");
    for (size_t i = 0; i < output.size(); ++i) {
        LOG_PRINT(" %d", output[i]);
        if (output[i] != expect[i]) {
            passed = false;
        }
    }
    LOG_PRINT("\nexpect:");
    for (auto value : expect) {
        LOG_PRINT(" %d", value);
    }
    LOG_PRINT("\n");

    aclDestroyTensor(inputTensor);
    aclDestroyTensor(shiftTensor);
    aclDestroyTensor(outputTensor);
    aclrtFree(inputDeviceAddr);
    aclrtFree(shiftDeviceAddr);
    aclrtFree(outputDeviceAddr);
    if (workspaceAddr != nullptr) {
        aclrtFree(workspaceAddr);
    }

    return passed ? ACL_SUCCESS : ACL_ERROR_FAILURE;
}
} // namespace

int main()
{
    aclrtStream stream = nullptr;
    auto ret = InitAcl(&stream);
    CHECK_RET(ret == ACL_SUCCESS, return ret);

    ret = RunRightShiftDemo(stream);

    if (stream != nullptr) {
        aclrtDestroyStream(stream);
    }
    aclrtResetDevice(DEVICE_ID);
    aclFinalize();

    LOG_PRINT(ret == ACL_SUCCESS ? "[  PASSED  ] right_shift demo passed.\n" :
                                   "[  FAILED  ] right_shift demo failed.\n");
    return ret;
}
