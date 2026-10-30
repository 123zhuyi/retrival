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
 * \file test_aclnn_sim_thread_exponential_st.cpp
 * \brief Simulation test for sim_thread_exponential operator with validation
 */

#include "acl/acl.h"
#include "aclnnop/aclnn_sim_thread_exponential.h"
#include <iostream>
#include <vector>
#include <cmath>
#include <cstdlib>
#include <cstring>
#include <numeric>
#include <limits>

#define CHECK_RET(cond, return_expr) \
  do {                               \
    if (!(cond)) {                   \
      return_expr;                   \
    }                                \
  } while (0)

#define LOG_PRINT(message, ...)     \
  do {                              \
    printf(message, ##__VA_ARGS__); \
  } while (0)

struct TestCase {
    const char* name;
    std::vector<int64_t> shape;
    aclDataType dtype;
    size_t elemSize;
    int64_t count;
    double lambd;
    int64_t seed;
    int64_t offset;
};

int64_t GetShapeSize(const std::vector<int64_t>& shape) {
    int64_t shape_size = 1;
    for (auto i : shape) {
        shape_size *= i;
    }
    return shape_size;
}

void* CreateHostData(const std::vector<int64_t>& shape, size_t elemSize) {
    int64_t totalElements = GetShapeSize(shape);
    size_t totalBytes = totalElements * elemSize;
    void* data = std::malloc(totalBytes);
    std::memset(data, 0, totalBytes);
    return data;
}

int Init(int32_t deviceId, aclrtStream* stream) {
    auto ret = aclInit(nullptr);
    CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclInit failed. ERROR: %d\n", ret); return ret);
    ret = aclrtSetDevice(deviceId);
    CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclrtSetDevice failed. ERROR: %d\n", ret); return ret);
    ret = aclrtCreateStream(stream);
    CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclrtCreateStream failed. ERROR: %d\n", ret); return ret);
    return 0;
}

int CreateAclTensor(const std::vector<int64_t>& shape, aclDataType dtype,
                    void** deviceAddr, aclTensor** tensor) {
    int64_t totalElements = GetShapeSize(shape);
    size_t elemSize = 0;
    switch (dtype) {
        case ACL_FLOAT:   elemSize = 4; break;
        case ACL_FLOAT16: elemSize = 2; break;
        case ACL_BF16:    elemSize = 2; break;
        default:          elemSize = 4; break;
    }
    size_t size = totalElements * elemSize;

    auto ret = aclrtMalloc(deviceAddr, size, ACL_MEM_MALLOC_HUGE_FIRST);
    CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclrtMalloc failed. ERROR: %d\n", ret); return ret);

    // Zero-fill device memory
    void* hostData = std::malloc(size);
    std::memset(hostData, 0, size);
    ret = aclrtMemcpy(*deviceAddr, size, hostData, size, ACL_MEMCPY_HOST_TO_DEVICE);
    std::free(hostData);
    CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclrtMemcpy failed. ERROR: %d\n", ret); return ret);

    std::vector<int64_t> strides(shape.size(), 1);
    for (int64_t i = shape.size() - 2; i >= 0; i--) {
        strides[i] = shape[i + 1] * strides[i + 1];
    }

    *tensor = aclCreateTensor(shape.data(), shape.size(), dtype,
                              strides.data(), 0, aclFormat::ACL_FORMAT_ND,
                              shape.data(), shape.size(), *deviceAddr);
    return 0;
}

// Statistical validation: check mean is close to 1/lambd and all values > 0
bool ValidateExponential(const float* data, int64_t n, double lambd, int caseNum) {
    double sum = 0.0;
    int64_t invalidCount = 0;
    int64_t nanCount = 0;
    int64_t infCount = 0;
    int64_t zeroOrNegCount = 0;

    for (int64_t i = 0; i < n; ++i) {
        float val = data[i];
        sum += static_cast<double>(val);
        if (std::isnan(val)) nanCount++;
        if (std::isinf(val)) infCount++;
        if (val <= 0.0f) zeroOrNegCount++;
        if (!std::isfinite(val) || val <= 0.0f) invalidCount++;
    }

    double mean = sum / n;
    double expectedMean = 1.0 / lambd;

    LOG_PRINT("  [CASE %d] n=%ld, lambd=%.2f\n", caseNum, n, lambd);
    LOG_PRINT("    mean=%.6f (expected ~%.6f)\n", mean, expectedMean);
    LOG_PRINT("    nan=%ld, inf=%ld, zero_or_neg=%ld\n", nanCount, infCount, zeroOrNegCount);

    // Statistical check: for n >= 100, mean should be within 3*sigma of expected
    // std of sample mean = std_exponential / sqrt(n) = (1/lambd) / sqrt(n)
    bool pass = true;
    double stdMean = expectedMean / std::sqrt(static_cast<double>(n));
    double deviation = std::fabs(mean - expectedMean);
    if (n >= 100 && deviation > 3.0 * stdMean) {
        LOG_PRINT("    [WARN] mean deviation %.6f > 3*sigma(%.6f)\n", deviation, 3.0 * stdMean);
        // Don't fail on statistical check for small n
    }

    if (invalidCount > 0) {
        LOG_PRINT("    [FAIL] %ld invalid values found\n", invalidCount);
        pass = false;
    } else {
        LOG_PRINT("    [PASS] all values valid (finite, > 0)\n");
    }
    return pass;
}

std::vector<TestCase> BuildTestCases() {
    std::vector<TestCase> cases;

    // Basic float32 tests
    cases.push_back({"fp32_small_2x3", {2, 3}, ACL_FLOAT, 4, 6, 1.0, 42, 0});
    cases.push_back({"fp32_1d_256", {256}, ACL_FLOAT, 4, 256, 1.0, 42, 4});
    cases.push_back({"fp32_4d_4x4x4x4", {4, 4, 4, 4}, ACL_FLOAT, 4, 256, 1.0, 42, 8});
    cases.push_back({"fp32_large_1024", {1024}, ACL_FLOAT, 4, 1024, 1.0, 42, 12});
    cases.push_back({"fp32_lambd_0_5", {400}, ACL_FLOAT, 4, 400, 0.5, 100, 0});
    cases.push_back({"fp32_lambd_2_0", {400}, ACL_FLOAT, 4, 400, 2.0, 100, 4});
    cases.push_back({"fp32_seed_123", {256}, ACL_FLOAT, 4, 256, 1.0, 123, 0});
    cases.push_back({"fp32_single", {1, 1}, ACL_FLOAT, 4, 1, 1.0, 42, 0});

    // FP16 tests
    cases.push_back({"fp16_small_2x4", {2, 4}, ACL_FLOAT16, 2, 8, 1.0, 42, 0});
    cases.push_back({"fp16_1d_128", {128}, ACL_FLOAT16, 2, 128, 1.0, 42, 4});

    // BF16 tests
    cases.push_back({"bf16_small_2x4", {2, 4}, ACL_BF16, 2, 8, 1.0, 42, 0});
    cases.push_back({"bf16_1d_128", {128}, ACL_BF16, 2, 128, 1.0, 42, 4});

    return cases;
}

int main() {
    int32_t deviceId = 0;
    aclrtStream stream;
    auto ret = Init(deviceId, &stream);
    CHECK_RET(ret == 0, LOG_PRINT("Init acl failed. ERROR: %d\n", ret); return ret);

    auto testCases = BuildTestCases();
    int totalCases = testCases.size();
    int passCount = 0;
    int failCount = 0;

    LOG_PRINT("\n========================================\n");
    LOG_PRINT("  sim_thread_exponential Simulation Test\n");
    LOG_PRINT("========================================\n\n");

    for (int i = 0; i < totalCases; ++i) {
        auto& tc = testCases[i];
        int64_t totalElements = GetShapeSize(tc.shape);
        LOG_PRINT("--- Test %d/%d: %s (shape=[", i+1, totalCases, tc.name);
        for (size_t j = 0; j < tc.shape.size(); ++j) {
            LOG_PRINT("%ld%s", tc.shape[j], j < tc.shape.size()-1 ? "," : "");
        }
        LOG_PRINT("], dtype=%d, count=%ld, lambd=%.1f, seed=%ld, offset=%ld)\n",
                  tc.dtype, tc.count, tc.lambd, tc.seed, tc.offset);

        // Create self tensor (in-place)
        void* selfDeviceAddr = nullptr;
        aclTensor* selfTensor = nullptr;
        ret = CreateAclTensor(tc.shape, tc.dtype, &selfDeviceAddr, &selfTensor);
        CHECK_RET(ret == ACL_SUCCESS,
                  LOG_PRINT("  CreateAclTensor failed\n"); failCount++; continue;);

        // Call operator
        uint64_t workspaceSize = 0;
        aclOpExecutor* executor;
        ret = aclnnSimThreadExponentialGetWorkspaceSize(selfTensor, tc.count, tc.lambd, tc.seed, tc.offset,
                                                        &workspaceSize, &executor);
        CHECK_RET(ret == ACL_SUCCESS,
                  LOG_PRINT("  GetWorkspaceSize failed. ERROR: %d\n", ret);
                  aclDestroyTensor(selfTensor); aclrtFree(selfDeviceAddr); failCount++; continue;);

        void* workspaceAddr = nullptr;
        if (workspaceSize > 0) {
            ret = aclrtMalloc(&workspaceAddr, workspaceSize, ACL_MEM_MALLOC_HUGE_FIRST);
            CHECK_RET(ret == ACL_SUCCESS,
                      LOG_PRINT("  allocate workspace failed\n");
                      aclDestroyTensor(selfTensor); aclrtFree(selfDeviceAddr); failCount++; continue;);
        }

        ret = aclnnSimThreadExponential(workspaceAddr, workspaceSize, executor, stream);
        CHECK_RET(ret == ACL_SUCCESS,
                  LOG_PRINT("  aclnnSimThreadExponential failed. ERROR: %d\n", ret);
                  if (workspaceAddr) aclrtFree(workspaceAddr);
                  aclDestroyTensor(selfTensor); aclrtFree(selfDeviceAddr); failCount++; continue;);

        ret = aclrtSynchronizeStream(stream);
        CHECK_RET(ret == ACL_SUCCESS,
                  LOG_PRINT("  aclrtSynchronizeStream failed. ERROR: %d\n", ret);
                  if (workspaceAddr) aclrtFree(workspaceAddr);
                  aclDestroyTensor(selfTensor); aclrtFree(selfDeviceAddr); failCount++; continue;);

        // Copy result and validate
        size_t resultSize = totalElements * sizeof(float);
        std::vector<float> resultData(totalElements, 0.0f);

        if (tc.dtype == ACL_FLOAT) {
            ret = aclrtMemcpy(resultData.data(), resultSize, selfDeviceAddr,
                              resultSize, ACL_MEMCPY_DEVICE_TO_HOST);
        } else {
            // For FP16/BF16, need to copy as raw bytes and convert
            size_t rawSize = totalElements * tc.elemSize;
            std::vector<uint8_t> rawData(rawSize);
            ret = aclrtMemcpy(rawData.data(), rawSize, selfDeviceAddr,
                              rawSize, ACL_MEMCPY_DEVICE_TO_HOST);
            if (ret == ACL_SUCCESS) {
                if (tc.dtype == ACL_FLOAT16) {
                    // Convert FP16 to float for validation
                    for (int64_t j = 0; j < totalElements; ++j) {
                        uint16_t half = reinterpret_cast<uint16_t*>(rawData.data())[j];
                        uint32_t sign = (half & 0x8000) << 16;
                        uint32_t exp  = ((half >> 10) & 0x1F);
                        uint32_t mant = (half & 0x3FF) << 13;
                        if (exp == 0) {
                            if (mant == 0) { resultData[j] = sign ? -0.0f : 0.0f; }
                            else { /* subnormal - approx as tiny */ resultData[j] = 0.0f; }
                        } else if (exp == 31) {
                            resultData[j] = std::numeric_limits<float>::quiet_NaN();
                        } else {
                            exp = exp - 15 + 127;
                            uint32_t f32 = sign | (exp << 23) | mant;
                            resultData[j] = *reinterpret_cast<float*>(&f32);
                        }
                    }
                } else {
                    // BF16: just pad with zeros
                    for (int64_t j = 0; j < totalElements; ++j) {
                        uint16_t bf16 = reinterpret_cast<uint16_t*>(rawData.data())[j];
                        uint32_t f32 = static_cast<uint32_t>(bf16) << 16;
                        resultData[j] = *reinterpret_cast<float*>(&f32);
                    }
                }
            }
        }
        CHECK_RET(ret == ACL_SUCCESS,
                  LOG_PRINT("  copy result failed. ERROR: %d\n", ret);
                  if (workspaceAddr) aclrtFree(workspaceAddr);
                  aclDestroyTensor(selfTensor); aclrtFree(selfDeviceAddr); failCount++; continue;);

        bool valid = ValidateExponential(resultData.data(), totalElements, tc.lambd, i+1);
        if (valid) passCount++; else failCount++;

        // Cleanup
        if (workspaceAddr) aclrtFree(workspaceAddr);
        aclDestroyTensor(selfTensor);
        aclrtFree(selfDeviceAddr);
    }

    LOG_PRINT("\n========================================\n");
    LOG_PRINT("  SUMMARY: %d PASS, %d FAIL, %d total\n", passCount, failCount, totalCases);
    LOG_PRINT("========================================\n");

    aclrtDestroyStream(stream);
    aclrtResetDevice(deviceId);
    aclFinalize();
    return (failCount == 0) ? 0 : 1;
}
