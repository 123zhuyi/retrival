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
#include <cstdint>
#include <cstdlib>
#include <cstring>
#include <iostream>
#include <vector>

#include "acl/acl.h"
#include "aclnn_sqrt_backward.h"

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

aclError CreateAclTensor(const std::vector<int64_t> &shape, aclDataType dtype, void *deviceAddr, aclTensor **tensor)
{
    std::vector<int64_t> strides = MakeStrides(shape);
    const int64_t *shapePtr = shape.empty() ? nullptr : shape.data();
    const int64_t *stridesPtr = strides.empty() ? nullptr : strides.data();
    *tensor = aclCreateTensor(
        shapePtr, shape.size(), dtype, stridesPtr, 0, ACL_FORMAT_ND, shapePtr, shape.size(), deviceAddr);
    return *tensor == nullptr ? ACL_ERROR_FAILURE : ACL_SUCCESS;
}

int32_t GetDeviceId()
{
    const char *deviceEnv = std::getenv("ACL_DEVICE_ID");
    if (deviceEnv == nullptr || std::strlen(deviceEnv) == 0) {
        return 0;
    }
    return std::atoi(deviceEnv);
}

struct ExampleState {
    aclrtStream stream = nullptr;
    void *yDevice = nullptr;
    void *dyDevice = nullptr;
    void *zDevice = nullptr;
    void *workspace = nullptr;
    aclTensor *yTensor = nullptr;
    aclTensor *dyTensor = nullptr;
    aclTensor *zTensor = nullptr;
    aclOpExecutor *executor = nullptr;
};

void CleanupState(ExampleState &state)
{
    if (state.yTensor != nullptr) {
        aclDestroyTensor(state.yTensor);
    }
    if (state.dyTensor != nullptr) {
        aclDestroyTensor(state.dyTensor);
    }
    if (state.zTensor != nullptr) {
        aclDestroyTensor(state.zTensor);
    }
    if (state.workspace != nullptr) {
        aclrtFree(state.workspace);
    }
    if (state.yDevice != nullptr) {
        aclrtFree(state.yDevice);
    }
    if (state.dyDevice != nullptr) {
        aclrtFree(state.dyDevice);
    }
    if (state.zDevice != nullptr) {
        aclrtFree(state.zDevice);
    }
    if (state.stream != nullptr) {
        aclrtDestroyStream(state.stream);
    }
}

aclError InitRuntime(int32_t deviceId, ExampleState &state)
{
    aclError ret = aclInit(nullptr);
    if (ret != ACL_SUCCESS) {
        return ret;
    }
    ret = aclrtSetDevice(deviceId);
    if (ret != ACL_SUCCESS) {
        return ret;
    }
    return aclrtCreateStream(&state.stream);
}

aclError CreateDeviceTensor(const std::vector<float> &hostData, void **devicePtr, aclTensor **tensor,
                            const std::vector<int64_t> &shape)
{
    size_t bytes = hostData.size() * sizeof(float);
    aclError ret = aclrtMalloc(devicePtr, bytes, ACL_MEM_MALLOC_HUGE_FIRST);
    if (ret != ACL_SUCCESS) {
        return ret;
    }
    ret = aclrtMemcpy(*devicePtr, bytes, hostData.data(), bytes, ACL_MEMCPY_HOST_TO_DEVICE);
    if (ret != ACL_SUCCESS) {
        return ret;
    }
    return CreateAclTensor(shape, ACL_FLOAT, *devicePtr, tensor);
}

aclError CreateOutputTensor(size_t elements, void **devicePtr, aclTensor **tensor, const std::vector<int64_t> &shape)
{
    size_t bytes = elements * sizeof(float);
    aclError ret = aclrtMalloc(devicePtr, bytes, ACL_MEM_MALLOC_HUGE_FIRST);
    if (ret != ACL_SUCCESS) {
        return ret;
    }
    std::vector<float> zeros(elements, 0.0f);
    ret = aclrtMemcpy(*devicePtr, bytes, zeros.data(), bytes, ACL_MEMCPY_HOST_TO_DEVICE);
    if (ret != ACL_SUCCESS) {
        return ret;
    }
    return CreateAclTensor(shape, ACL_FLOAT, *devicePtr, tensor);
}

bool CheckOutput(const std::vector<float> &expected, const std::vector<float> &actual)
{
    constexpr float kTolerance = 1e-4f;
    for (size_t i = 0; i < expected.size(); ++i) {
        if (std::fabs(expected[i] - actual[i]) > kTolerance) {
            std::cerr << "mismatch at " << i << ": expected=" << expected[i] << ", actual=" << actual[i] << '\n';
            return false;
        }
    }
    return true;
}

aclError PrepareExampleTensors(const std::vector<float> &yHost, const std::vector<float> &dyHost,
                               const std::vector<float> &expected, const std::vector<int64_t> &shape,
                               ExampleState &state)
{
    aclError ret = CreateDeviceTensor(yHost, &state.yDevice, &state.yTensor, shape);
    if (ret != ACL_SUCCESS) {
        return ret;
    }
    ret = CreateDeviceTensor(dyHost, &state.dyDevice, &state.dyTensor, shape);
    if (ret != ACL_SUCCESS) {
        return ret;
    }
    return CreateOutputTensor(expected.size(), &state.zDevice, &state.zTensor, shape);
}

aclError ExecuteExampleOp(ExampleState &state, uint64_t &workspaceSize)
{
    aclError ret = aclnnSqrtBackwardGetWorkspaceSize(
        state.yTensor, state.dyTensor, state.zTensor, &workspaceSize, &state.executor);
    if (ret != ACL_SUCCESS) {
        return ret;
    }
    if (workspaceSize > 0) {
        ret = aclrtMalloc(&state.workspace, workspaceSize, ACL_MEM_MALLOC_HUGE_FIRST);
        if (ret != ACL_SUCCESS) {
            return ret;
        }
    }
    ret = aclnnSqrtBackward(state.workspace, workspaceSize, state.executor, state.stream);
    if (ret != ACL_SUCCESS) {
        return ret;
    }
    return aclrtSynchronizeStream(state.stream);
}

aclError CopyOutputToHost(const ExampleState &state, std::vector<float> &actual)
{
    return aclrtMemcpy(actual.data(), actual.size() * sizeof(float), state.zDevice, actual.size() * sizeof(float),
                       ACL_MEMCPY_DEVICE_TO_HOST);
}

int FinishExample(ExampleState &state, const std::vector<float> &expected)
{
    std::vector<float> actual(expected.size(), 0.0f);
    aclError ret = CopyOutputToHost(state, actual);
    CleanupState(state);
    if (ret != ACL_SUCCESS) {
        return static_cast<int>(ret);
    }
    return CheckOutput(expected, actual) ? 0 : 1;
}

int RunExample()
{
    const std::vector<int64_t> shape = {2, 2};
    const std::vector<float> yHost = {1.0f, 2.0f, 4.0f, 8.0f};
    const std::vector<float> dyHost = {2.0f, 4.0f, 8.0f, 0.0f};
    const std::vector<float> expected = {1.0f, 1.0f, 1.0f, 0.0f};
    ExampleState state;
    uint64_t workspaceSize = 0;
    aclError ret = InitRuntime(GetDeviceId(), state);
    if (ret != ACL_SUCCESS) {
        return static_cast<int>(ret);
    }
    ret = PrepareExampleTensors(yHost, dyHost, expected, shape, state);
    if (ret != ACL_SUCCESS) {
        CleanupState(state);
        return static_cast<int>(ret);
    }
    ret = ExecuteExampleOp(state, workspaceSize);
    if (ret != ACL_SUCCESS) {
        CleanupState(state);
        return static_cast<int>(ret);
    }
    return FinishExample(state, expected);
}

}  // namespace

int main()
{
    int result = RunExample();
    aclFinalize();
    return result;
}
