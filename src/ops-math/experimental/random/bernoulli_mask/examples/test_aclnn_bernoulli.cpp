/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <vector>

#include "acl/acl.h"
#include "aclnn_bernoulli.h"

namespace {
constexpr int32_t DEVICE_ID = 0;
constexpr int64_t SEED = 20260725;
constexpr int64_t OFFSET = 4;
constexpr float PROBABILITY = 0.35F;
constexpr int64_t ELEMENTS = 65536;

#define CHECK_ACL(expr)                                                                               \
    do {                                                                                              \
        const aclError checkAclStatus = (expr);                                                       \
        if (checkAclStatus != ACL_SUCCESS) {                                                          \
            std::fprintf(stderr, "%s failed: %d, %s\n", #expr, checkAclStatus, aclGetRecentErrMsg()); \
            return 1;                                                                                 \
        }                                                                                             \
    } while (0)

struct TensorResources {
    void* device = nullptr;
    aclTensor* tensor = nullptr;

    void Release()
    {
        if (tensor != nullptr) {
            (void)aclDestroyTensor(tensor);
            tensor = nullptr;
        }
        if (device != nullptr) {
            (void)aclrtFree(device);
            device = nullptr;
        }
    }

    ~TensorResources() { Release(); }
};

int CreateContiguousFloatTensor(const std::vector<int64_t>& shape, const std::vector<float>& hostData,
                                TensorResources& resources)
{
    const size_t bytes = std::max<size_t>(hostData.size() * sizeof(float), 1);
    CHECK_ACL(aclrtMalloc(&resources.device, bytes, ACL_MEM_MALLOC_HUGE_FIRST));
    if (!hostData.empty()) {
        CHECK_ACL(aclrtMemcpy(resources.device, bytes, hostData.data(), hostData.size() * sizeof(float),
                              ACL_MEMCPY_HOST_TO_DEVICE));
    }

    std::vector<int64_t> strides(shape.size(), 1);
    for (size_t reverse = shape.size(); reverse > 1; --reverse) {
        strides[reverse - 2] = strides[reverse - 1] * shape[reverse - 1];
    }
    resources.tensor = aclCreateTensor(shape.data(), shape.size(), ACL_FLOAT, strides.data(), 0, ACL_FORMAT_ND,
                                       shape.data(), shape.size(), resources.device);
    if (resources.tensor == nullptr) {
        std::fprintf(stderr, "aclCreateTensor failed: %s\n", aclGetRecentErrMsg());
        return 1;
    }
    return 0;
}

int RunOnce(aclrtStream stream, const aclTensor* self, aclTensor* out, const void* outDevice,
            const aclScalar* probability, std::vector<float>& hostOut, uint64_t& workspaceBytes)
{
    aclOpExecutor* executor = nullptr;
    const aclnnStatus status = aclnnBernoulliGetWorkspaceSize(self, probability, SEED, OFFSET, out, &workspaceBytes,
                                                              &executor);
    if (status != ACL_SUCCESS) {
        std::fprintf(stderr, "aclnnBernoulliGetWorkspaceSize failed: %d, %s\n", status, aclGetRecentErrMsg());
        return 1;
    }

    void* workspace = nullptr;
    if (workspaceBytes > 0) {
        CHECK_ACL(aclrtMalloc(&workspace, workspaceBytes, ACL_MEM_MALLOC_HUGE_FIRST));
    }
    const aclnnStatus launchStatus = aclnnBernoulli(workspace, workspaceBytes, executor, stream);
    if (launchStatus != ACL_SUCCESS) {
        std::fprintf(stderr, "aclnnBernoulli failed: %d, %s\n", launchStatus, aclGetRecentErrMsg());
        if (workspace != nullptr) {
            (void)aclrtFree(workspace);
        }
        return 1;
    }
    CHECK_ACL(aclrtSynchronizeStream(stream));
    CHECK_ACL(aclrtMemcpy(hostOut.data(), hostOut.size() * sizeof(float), outDevice, hostOut.size() * sizeof(float),
                          ACL_MEMCPY_DEVICE_TO_HOST));
    if (workspace != nullptr) {
        CHECK_ACL(aclrtFree(workspace));
    }
    return 0;
}
} // namespace

int main()
{
    CHECK_ACL(aclInit(nullptr));
    CHECK_ACL(aclrtSetDevice(DEVICE_ID));
    aclrtStream stream = nullptr;
    CHECK_ACL(aclrtCreateStream(&stream));

    const std::vector<int64_t> shape = {256, 256};
    const std::vector<float> input(ELEMENTS, 0.0F);
    std::vector<float> first(ELEMENTS, -1.0F);
    std::vector<float> second(ELEMENTS, -1.0F);
    TensorResources self;
    TensorResources out;
    if (CreateContiguousFloatTensor(shape, input, self) != 0 || CreateContiguousFloatTensor(shape, first, out) != 0) {
        return 1;
    }

    float probabilityValue = PROBABILITY;
    aclScalar* probability = aclCreateScalar(&probabilityValue, ACL_FLOAT);
    if (probability == nullptr) {
        std::fprintf(stderr, "aclCreateScalar failed: %s\n", aclGetRecentErrMsg());
        return 1;
    }

    uint64_t firstWorkspace = 0;
    uint64_t secondWorkspace = 0;
    if (RunOnce(stream, self.tensor, out.tensor, out.device, probability, first, firstWorkspace) != 0 ||
        RunOnce(stream, self.tensor, out.tensor, out.device, probability, second, secondWorkspace) != 0) {
        return 1;
    }

    const bool binary = std::all_of(first.begin(), first.end(),
                                    [](float value) { return value == 0.0F || value == 1.0F; });
    const bool reproducible = first == second;
    const size_t ones = static_cast<size_t>(std::count(first.begin(), first.end(), 1.0F));
    const double mean = static_cast<double>(ones) / static_cast<double>(first.size());
    const double sigma = std::sqrt(static_cast<double>(PROBABILITY) * (1.0 - static_cast<double>(PROBABILITY)) /
                                   first.size());
    const bool distributionOk = std::abs(mean - static_cast<double>(PROBABILITY)) <= 6.0 * sigma;
    const bool workspaceStable = firstWorkspace == secondWorkspace;

    std::printf("binary=%s reproducible=%s distribution=%s mean=%.8f expected=%.8f workspace=%llu bytes\n",
                binary ? "PASS" : "FAIL", reproducible ? "PASS" : "FAIL", distributionOk ? "PASS" : "FAIL", mean,
                static_cast<double>(PROBABILITY), static_cast<unsigned long long>(firstWorkspace));

    (void)aclDestroyScalar(probability);
    self.Release();
    out.Release();
    (void)aclrtDestroyStream(stream);
    (void)aclrtResetDevice(DEVICE_ID);
    (void)aclFinalize();
    return binary && reproducible && distributionOk && workspaceStable ? 0 : 1;
}
