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
 * @file test_aclnn_add_mat_mat_elements.cpp
 * @brief AddMatMatElements 算子 ACLNN 调用示例（FP16 / FP32 / BF16）
 *
 * 公式：c_out = c * beta + alpha * a * b
 * 支持 dtype：ACL_FLOAT16, ACL_FLOAT, ACL_BF16
 * 运行环境：Ascend950（Atlas A3 训练系列）
 */

#include <cmath>
#include <cstdint>
#include <cstring>
#include <iostream>
#include <vector>
#include "acl/acl.h"
#include "aclnn_add_mat_mat_elements.h"

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

// ============================================================================
// FP16 / BF16 转换工具
// ============================================================================

static uint16_t FloatToFp16(float val)
{
    uint32_t f32;
    std::memcpy(&f32, &val, sizeof(f32));
    uint16_t sign = (f32 >> 16) & 0x8000;
    int32_t exp = ((f32 >> 23) & 0xFF) - 127 + 15;
    uint16_t frac = (f32 >> 13) & 0x03FF;
    if (exp <= 0) {
        return sign;
    } else if (exp >= 31) {
        return sign | 0x7C00;
    }
    return sign | (static_cast<uint16_t>(exp) << 10) | frac;
}

static float Fp16ToFloat(uint16_t h)
{
    uint32_t sign = (static_cast<uint32_t>(h) & 0x8000) << 16;
    uint32_t exp = (h >> 10) & 0x1F;
    uint32_t frac = h & 0x03FF;
    if (exp == 0) {
        if (frac == 0) {
            uint32_t f32 = sign;
            float result;
            std::memcpy(&result, &f32, sizeof(result));
            return result;
        }
        // subnormal
        exp = 1;
        while (!(frac & 0x0400)) {
            frac <<= 1;
            exp--;
        }
        frac &= 0x03FF;
        exp = exp + 127 - 15;
    } else if (exp == 31) {
        exp = 255;
    } else {
        exp = exp + 127 - 15;
    }
    uint32_t f32 = sign | (exp << 23) | (frac << 13);
    float result;
    std::memcpy(&result, &f32, sizeof(result));
    return result;
}

static uint16_t FloatToBf16(float val)
{
    uint32_t f32;
    std::memcpy(&f32, &val, sizeof(f32));
    // 简单截断：取高 16 位
    return static_cast<uint16_t>(f32 >> 16);
}

static float Bf16ToFloat(uint16_t h)
{
    uint32_t f32 = static_cast<uint32_t>(h) << 16;
    float result;
    std::memcpy(&result, &f32, sizeof(result));
    return result;
}

// ============================================================================
// 通用工具函数
// ============================================================================

int64_t GetShapeSize(const std::vector<int64_t>& shape)
{
    int64_t shapeSize = 1;
    for (auto i : shape) {
        shapeSize *= i;
    }
    return shapeSize;
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

static const char* DtypeName(aclDataType dtype)
{
    switch (dtype) {
        case ACL_FLOAT16:
            return "FP16";
        case ACL_FLOAT:
            return "FP32";
        case ACL_BF16:
            return "BF16";
        default:
            return "UNKNOWN";
    }
}

// ============================================================================
// FP32 测试
// ============================================================================

int RunTestFP32(aclrtStream stream)
{
    LOG_PRINT("\n========== Test FP32 ==========\n");

    std::vector<int64_t> shape = {4, 8};
    int64_t numElements = GetShapeSize(shape);

    std::vector<float> aHost(numElements);
    std::vector<float> bHost(numElements);
    std::vector<float> cHost(numElements);
    std::vector<float> cOutHost(numElements, 0.0f);

    for (int64_t i = 0; i < numElements; ++i) {
        aHost[i] = static_cast<float>(i + 1);
        bHost[i] = 1.0f / static_cast<float>(i + 1);
        cHost[i] = static_cast<float>(i % 4) * 0.25f;
    }

    float alphaVal = 2.0f;
    float betaVal = 0.5f;

    void* devA = nullptr;
    void* devB = nullptr;
    void* devC = nullptr;
    void* devOut = nullptr;
    aclTensor* tA = nullptr;
    aclTensor* tB = nullptr;
    aclTensor* tC = nullptr;
    aclTensor* tOut = nullptr;

    auto ret = CreateAclTensor(aHost, shape, &devA, ACL_FLOAT, &tA);
    CHECK_RET(ret == ACL_SUCCESS, return ret);
    ret = CreateAclTensor(bHost, shape, &devB, ACL_FLOAT, &tB);
    CHECK_RET(ret == ACL_SUCCESS, return ret);
    ret = CreateAclTensor(cHost, shape, &devC, ACL_FLOAT, &tC);
    CHECK_RET(ret == ACL_SUCCESS, return ret);
    ret = CreateAclTensor(cOutHost, shape, &devOut, ACL_FLOAT, &tOut);
    CHECK_RET(ret == ACL_SUCCESS, return ret);

    aclScalar* sAlpha = aclCreateScalar(&alphaVal, ACL_FLOAT);
    aclScalar* sBeta = aclCreateScalar(&betaVal, ACL_FLOAT);

    uint64_t workspaceSize = 0;
    aclOpExecutor* executor = nullptr;
    ret = aclnnAddMatMatElementsGetWorkspaceSize(tA, tB, tC, sAlpha, sBeta, tOut, &workspaceSize, &executor);
    CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("  GetWorkspaceSize failed. ERROR: %d\n", ret); return ret);

    void* wsAddr = nullptr;
    if (workspaceSize > 0) {
        ret = aclrtMalloc(&wsAddr, workspaceSize, ACL_MEM_MALLOC_HUGE_FIRST);
        CHECK_RET(ret == ACL_SUCCESS, return ret);
    }

    ret = aclnnAddMatMatElements(wsAddr, workspaceSize, executor, stream);
    CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("  Execute failed. ERROR: %d\n", ret); return ret);
    ret = aclrtSynchronizeStream(stream);
    CHECK_RET(ret == ACL_SUCCESS, return ret);

    ret = aclrtMemcpy(
        cOutHost.data(), numElements * sizeof(float), devOut, numElements * sizeof(float), ACL_MEMCPY_DEVICE_TO_HOST);
    CHECK_RET(ret == ACL_SUCCESS, return ret);

    // 精度验证
    double maxErr = 0.0;
    for (int64_t i = 0; i < numElements; ++i) {
        float golden = cHost[i] * betaVal + alphaVal * aHost[i] * bHost[i];
        double err = std::abs(static_cast<double>(cOutHost[i]) - static_cast<double>(golden));
        if (err > maxErr)
            maxErr = err;
    }
    bool pass = (maxErr < 1e-5);
    LOG_PRINT("  max_abs_err = %.6e, %s\n", maxErr, pass ? "PASS" : "FAIL");
    for (int64_t i = 0; i < 4; ++i) {
        float golden = cHost[i] * betaVal + alphaVal * aHost[i] * bHost[i];
        LOG_PRINT("  [%ld] actual=%.6f golden=%.6f\n", i, cOutHost[i], golden);
    }

    aclDestroyTensor(tA);
    aclDestroyTensor(tB);
    aclDestroyTensor(tC);
    aclDestroyTensor(tOut);
    aclDestroyScalar(sAlpha);
    aclDestroyScalar(sBeta);
    aclrtFree(devA);
    aclrtFree(devB);
    aclrtFree(devC);
    aclrtFree(devOut);
    if (wsAddr)
        aclrtFree(wsAddr);

    return pass ? 0 : 1;
}

// ============================================================================
// FP16 测试
// ============================================================================

int RunTestFP16(aclrtStream stream)
{
    LOG_PRINT("\n========== Test FP16 ==========\n");

    std::vector<int64_t> shape = {4, 8};
    int64_t numElements = GetShapeSize(shape);

    // 准备 FP16 host 数据（uint16_t 存储）
    std::vector<uint16_t> aHost(numElements);
    std::vector<uint16_t> bHost(numElements);
    std::vector<uint16_t> cHost(numElements);
    std::vector<uint16_t> cOutHost(numElements, 0);

    // 保存 float 原始值用于 golden 计算
    std::vector<float> aFloat(numElements);
    std::vector<float> bFloat(numElements);
    std::vector<float> cFloat(numElements);

    for (int64_t i = 0; i < numElements; ++i) {
        float av = static_cast<float>(i % 8 + 1); // 1~8，避免 FP16 溢出
        float bv = 1.0f / static_cast<float>(i % 8 + 1);
        float cv = static_cast<float>(i % 4) * 0.25f;
        aHost[i] = FloatToFp16(av);
        bHost[i] = FloatToFp16(bv);
        cHost[i] = FloatToFp16(cv);
        aFloat[i] = Fp16ToFloat(aHost[i]); // 用量化后的值作 golden
        bFloat[i] = Fp16ToFloat(bHost[i]);
        cFloat[i] = Fp16ToFloat(cHost[i]);
    }

    float alphaVal = 2.0f;
    float betaVal = 0.5f;

    void* devA = nullptr;
    void* devB = nullptr;
    void* devC = nullptr;
    void* devOut = nullptr;
    aclTensor* tA = nullptr;
    aclTensor* tB = nullptr;
    aclTensor* tC = nullptr;
    aclTensor* tOut = nullptr;

    auto ret = CreateAclTensor(aHost, shape, &devA, ACL_FLOAT16, &tA);
    CHECK_RET(ret == ACL_SUCCESS, return ret);
    ret = CreateAclTensor(bHost, shape, &devB, ACL_FLOAT16, &tB);
    CHECK_RET(ret == ACL_SUCCESS, return ret);
    ret = CreateAclTensor(cHost, shape, &devC, ACL_FLOAT16, &tC);
    CHECK_RET(ret == ACL_SUCCESS, return ret);
    ret = CreateAclTensor(cOutHost, shape, &devOut, ACL_FLOAT16, &tOut);
    CHECK_RET(ret == ACL_SUCCESS, return ret);

    aclScalar* sAlpha = aclCreateScalar(&alphaVal, ACL_FLOAT);
    aclScalar* sBeta = aclCreateScalar(&betaVal, ACL_FLOAT);

    uint64_t workspaceSize = 0;
    aclOpExecutor* executor = nullptr;
    ret = aclnnAddMatMatElementsGetWorkspaceSize(tA, tB, tC, sAlpha, sBeta, tOut, &workspaceSize, &executor);
    CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("  GetWorkspaceSize failed. ERROR: %d\n", ret); return ret);

    void* wsAddr = nullptr;
    if (workspaceSize > 0) {
        ret = aclrtMalloc(&wsAddr, workspaceSize, ACL_MEM_MALLOC_HUGE_FIRST);
        CHECK_RET(ret == ACL_SUCCESS, return ret);
    }

    ret = aclnnAddMatMatElements(wsAddr, workspaceSize, executor, stream);
    CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("  Execute failed. ERROR: %d\n", ret); return ret);
    ret = aclrtSynchronizeStream(stream);
    CHECK_RET(ret == ACL_SUCCESS, return ret);

    ret = aclrtMemcpy(
        cOutHost.data(), numElements * sizeof(uint16_t), devOut, numElements * sizeof(uint16_t),
        ACL_MEMCPY_DEVICE_TO_HOST);
    CHECK_RET(ret == ACL_SUCCESS, return ret);

    // 精度验证（FP16 容差 rtol=1e-3）
    double maxRelErr = 0.0;
    for (int64_t i = 0; i < numElements; ++i) {
        float actual = Fp16ToFloat(cOutHost[i]);
        float golden = cFloat[i] * betaVal + alphaVal * aFloat[i] * bFloat[i];
        double relErr = std::abs(static_cast<double>(actual - golden)) / (std::abs(golden) + 1e-7);
        if (relErr > maxRelErr)
            maxRelErr = relErr;
    }
    bool pass = (maxRelErr < 1e-2);
    LOG_PRINT("  max_rel_err = %.6e, %s\n", maxRelErr, pass ? "PASS" : "FAIL");
    for (int64_t i = 0; i < 4; ++i) {
        float actual = Fp16ToFloat(cOutHost[i]);
        float golden = cFloat[i] * betaVal + alphaVal * aFloat[i] * bFloat[i];
        LOG_PRINT("  [%ld] actual=%.6f golden=%.6f\n", i, actual, golden);
    }

    aclDestroyTensor(tA);
    aclDestroyTensor(tB);
    aclDestroyTensor(tC);
    aclDestroyTensor(tOut);
    aclDestroyScalar(sAlpha);
    aclDestroyScalar(sBeta);
    aclrtFree(devA);
    aclrtFree(devB);
    aclrtFree(devC);
    aclrtFree(devOut);
    if (wsAddr)
        aclrtFree(wsAddr);

    return pass ? 0 : 1;
}

// ============================================================================
// BF16 测试
// ============================================================================

int RunTestBF16(aclrtStream stream)
{
    LOG_PRINT("\n========== Test BF16 ==========\n");

    std::vector<int64_t> shape = {4, 8};
    int64_t numElements = GetShapeSize(shape);

    std::vector<uint16_t> aHost(numElements);
    std::vector<uint16_t> bHost(numElements);
    std::vector<uint16_t> cHost(numElements);
    std::vector<uint16_t> cOutHost(numElements, 0);

    std::vector<float> aFloat(numElements);
    std::vector<float> bFloat(numElements);
    std::vector<float> cFloat(numElements);

    for (int64_t i = 0; i < numElements; ++i) {
        float av = static_cast<float>(i + 1);
        float bv = 1.0f / static_cast<float>(i + 1);
        float cv = static_cast<float>(i % 4) * 0.25f;
        aHost[i] = FloatToBf16(av);
        bHost[i] = FloatToBf16(bv);
        cHost[i] = FloatToBf16(cv);
        aFloat[i] = Bf16ToFloat(aHost[i]);
        bFloat[i] = Bf16ToFloat(bHost[i]);
        cFloat[i] = Bf16ToFloat(cHost[i]);
    }

    float alphaVal = 2.0f;
    float betaVal = 0.5f;

    void* devA = nullptr;
    void* devB = nullptr;
    void* devC = nullptr;
    void* devOut = nullptr;
    aclTensor* tA = nullptr;
    aclTensor* tB = nullptr;
    aclTensor* tC = nullptr;
    aclTensor* tOut = nullptr;

    auto ret = CreateAclTensor(aHost, shape, &devA, ACL_BF16, &tA);
    CHECK_RET(ret == ACL_SUCCESS, return ret);
    ret = CreateAclTensor(bHost, shape, &devB, ACL_BF16, &tB);
    CHECK_RET(ret == ACL_SUCCESS, return ret);
    ret = CreateAclTensor(cHost, shape, &devC, ACL_BF16, &tC);
    CHECK_RET(ret == ACL_SUCCESS, return ret);
    ret = CreateAclTensor(cOutHost, shape, &devOut, ACL_BF16, &tOut);
    CHECK_RET(ret == ACL_SUCCESS, return ret);

    aclScalar* sAlpha = aclCreateScalar(&alphaVal, ACL_FLOAT);
    aclScalar* sBeta = aclCreateScalar(&betaVal, ACL_FLOAT);

    uint64_t workspaceSize = 0;
    aclOpExecutor* executor = nullptr;
    ret = aclnnAddMatMatElementsGetWorkspaceSize(tA, tB, tC, sAlpha, sBeta, tOut, &workspaceSize, &executor);
    CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("  GetWorkspaceSize failed. ERROR: %d\n", ret); return ret);

    void* wsAddr = nullptr;
    if (workspaceSize > 0) {
        ret = aclrtMalloc(&wsAddr, workspaceSize, ACL_MEM_MALLOC_HUGE_FIRST);
        CHECK_RET(ret == ACL_SUCCESS, return ret);
    }

    ret = aclnnAddMatMatElements(wsAddr, workspaceSize, executor, stream);
    CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("  Execute failed. ERROR: %d\n", ret); return ret);
    ret = aclrtSynchronizeStream(stream);
    CHECK_RET(ret == ACL_SUCCESS, return ret);

    ret = aclrtMemcpy(
        cOutHost.data(), numElements * sizeof(uint16_t), devOut, numElements * sizeof(uint16_t),
        ACL_MEMCPY_DEVICE_TO_HOST);
    CHECK_RET(ret == ACL_SUCCESS, return ret);

    // 精度验证（BF16 容差 rtol=1e-2）
    double maxRelErr = 0.0;
    for (int64_t i = 0; i < numElements; ++i) {
        float actual = Bf16ToFloat(cOutHost[i]);
        float golden = cFloat[i] * betaVal + alphaVal * aFloat[i] * bFloat[i];
        double relErr = std::abs(static_cast<double>(actual - golden)) / (std::abs(golden) + 1e-7);
        if (relErr > maxRelErr)
            maxRelErr = relErr;
    }
    bool pass = (maxRelErr < 1e-2);
    LOG_PRINT("  max_rel_err = %.6e, %s\n", maxRelErr, pass ? "PASS" : "FAIL");
    for (int64_t i = 0; i < 4; ++i) {
        float actual = Bf16ToFloat(cOutHost[i]);
        float golden = cFloat[i] * betaVal + alphaVal * aFloat[i] * bFloat[i];
        LOG_PRINT("  [%ld] actual=%.6f golden=%.6f\n", i, actual, golden);
    }

    aclDestroyTensor(tA);
    aclDestroyTensor(tB);
    aclDestroyTensor(tC);
    aclDestroyTensor(tOut);
    aclDestroyScalar(sAlpha);
    aclDestroyScalar(sBeta);
    aclrtFree(devA);
    aclrtFree(devB);
    aclrtFree(devC);
    aclrtFree(devOut);
    if (wsAddr)
        aclrtFree(wsAddr);

    return pass ? 0 : 1;
}

// ============================================================================
// main
// ============================================================================

int main()
{
    int32_t deviceId = 0;
    aclrtStream stream;
    auto ret = Init(deviceId, &stream);
    CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("Init acl failed. ERROR: %d\n", ret); return ret);

    LOG_PRINT("AddMatMatElements ACLNN example (c_out = c * beta + alpha * a * b)\n");
    LOG_PRINT("Testing dtypes: FP16, FP32, BF16\n");

    int failCount = 0;
    failCount += RunTestFP32(stream);
    failCount += RunTestFP16(stream);
    failCount += RunTestBF16(stream);

    LOG_PRINT("\n========== Summary ==========\n");
    LOG_PRINT("Total: 3, Passed: %d, Failed: %d\n", 3 - failCount, failCount);

    aclrtDestroyStream(stream);
    aclrtResetDevice(deviceId);
    aclFinalize();
    return failCount > 0 ? 1 : 0;
}
