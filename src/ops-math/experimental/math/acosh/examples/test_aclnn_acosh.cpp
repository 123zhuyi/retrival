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
 * @file test_aclnn_acosh.cpp
 * @brief aclnnAcosh 调用示例（TilingKey_A: fp16 单缓冲）
 *
 * 用法：
 *   编译后直接运行，验证 fp16 acosh 在 NPU 上计算正确性
 *   输入值域 [1.0, 10.0] 确保 acosh 有效值
 */

#include <iostream>
#include <vector>
#include <cmath>
#include "acl/acl.h"
#include "aclnn_acosh.h"

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

// 将 float 转换为 fp16（简单实现）
uint16_t FloatToFp16(float f)
{
    uint32_t x = *reinterpret_cast<uint32_t*>(&f);
    uint16_t sign = (x >> 31) & 0x1;
    int32_t exp = ((x >> 23) & 0xff) - 127 + 15;
    uint32_t mantissa = x & 0x7fffff;
    if (exp <= 0)
        return sign << 15;
    if (exp >= 31)
        return (sign << 15) | (0x1f << 10);
    return (sign << 15) | (exp << 10) | (mantissa >> 13);
}

float Fp16ToFloat(uint16_t h)
{
    uint32_t sign = (h >> 15) & 0x1;
    uint32_t exp = (h >> 10) & 0x1f;
    uint32_t mantissa = h & 0x3ff;
    if (exp == 0) {
        if (mantissa == 0)
            return sign ? -0.0f : 0.0f;
        // denormal
        float val = mantissa / 1024.0f / 1024.0f;
        return sign ? -val : val;
    }
    if (exp == 31) {
        if (mantissa == 0)
            return sign ? -INFINITY : INFINITY;
        return NAN;
    }
    float val = (1.0f + mantissa / 1024.0f) * std::pow(2.0f, (int)exp - 15);
    return sign ? -val : val;
}

int main()
{
    int32_t deviceId = 0;
    aclrtStream stream;
    auto ret = Init(deviceId, &stream);
    CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("Init acl failed. ERROR: %d\n", ret); return ret);

    // 测试 shape: [8, 8] = 64 元素 (< 1024，触发 TilingKey_A fp16 单缓冲)
    std::vector<int64_t> selfShape = {8, 8};
    int64_t totalNum = GetShapeSize(selfShape);

    // 生成输入数据：范围 [1.0, 10.0]（acosh 有效域）
    std::vector<uint16_t> selfHostDataFp16(totalNum);
    std::vector<float> selfHostDataFloat(totalNum);
    std::vector<float> expectedResult(totalNum);
    for (int64_t i = 0; i < totalNum; i++) {
        float val = 1.0f + (float)i / totalNum * 9.0f; // [1.0, 10.0)
        selfHostDataFloat[i] = val;
        selfHostDataFp16[i] = FloatToFp16(val);
        expectedResult[i] = std::acosh(val);
    }

    // 创建输入 Tensor（fp16）
    aclTensor* selfTensor = nullptr;
    void* selfDeviceAddr = nullptr;
    ret = CreateAclTensor(selfHostDataFp16, selfShape, &selfDeviceAddr, aclDataType::ACL_FLOAT16, &selfTensor);
    CHECK_RET(ret == ACL_SUCCESS, return ret);

    // 创建输出 Tensor（fp16）
    aclTensor* outTensor = nullptr;
    void* outDeviceAddr = nullptr;
    std::vector<uint16_t> outHostDataFp16(totalNum, 0);
    ret = CreateAclTensor(outHostDataFp16, selfShape, &outDeviceAddr, aclDataType::ACL_FLOAT16, &outTensor);
    CHECK_RET(ret == ACL_SUCCESS, return ret);

    // 第一段接口
    uint64_t workspaceSize = 0;
    aclOpExecutor* executor = nullptr;
    ret = aclnnAcoshGetWorkspaceSize(selfTensor, outTensor, &workspaceSize, &executor);
    CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclnnAcoshGetWorkspaceSize failed. ERROR: %d\n", ret); return ret);

    void* workspaceAddr = nullptr;
    if (workspaceSize > 0) {
        ret = aclrtMalloc(&workspaceAddr, workspaceSize, ACL_MEM_MALLOC_HUGE_FIRST);
        CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("allocate workspace failed. ERROR: %d\n", ret); return ret);
    }

    // 第二段接口
    ret = aclnnAcosh(workspaceAddr, workspaceSize, executor, stream);
    CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclnnAcosh failed. ERROR: %d\n", ret); return ret);

    ret = aclrtSynchronizeStream(stream);
    CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclrtSynchronizeStream failed. ERROR: %d\n", ret); return ret);

    // 拷贝结果到 host
    std::vector<uint16_t> resultFp16(totalNum, 0);
    ret = aclrtMemcpy(
        resultFp16.data(), totalNum * sizeof(uint16_t), outDeviceAddr, totalNum * sizeof(uint16_t),
        ACL_MEMCPY_DEVICE_TO_HOST);
    CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("copy result failed. ERROR: %d\n", ret); return ret);

    // 精度验证（atol=1e-3, rtol=1e-3）
    LOG_PRINT("=== Acosh fp16 精度验证 (TilingKey_A: fp16 单缓冲) ===\n");
    int passCount = 0;
    int nanCount = 0;
    for (int64_t i = 0; i < totalNum; i++) {
        float result = Fp16ToFloat(resultFp16[i]);
        float expected = expectedResult[i];
        float absDiff = std::fabs(result - expected);
        float relDiff = (std::fabs(expected) > 1e-6f) ? absDiff / std::fabs(expected) : absDiff;
        if (std::isnan(result) && std::isnan(expected)) {
            nanCount++;
            passCount++;
        } else if (absDiff <= 1e-3f || relDiff <= 1e-3f) {
            passCount++;
        } else {
            LOG_PRINT(
                "FAIL[%ld]: input=%.4f, expected=%.4f, result=%.4f, absDiff=%.6f, relDiff=%.6f\n", i,
                selfHostDataFloat[i], expected, result, absDiff, relDiff);
        }
    }

    LOG_PRINT("总元素数: %ld, 通过: %d, NaN(忽略): %d\n", totalNum, passCount, nanCount);
    if (passCount == totalNum) {
        LOG_PRINT("=== 精度验证通过 ===\n");
    } else {
        LOG_PRINT("=== 精度验证失败 ===\n");
    }

    // 释放资源
    aclDestroyTensor(selfTensor);
    aclDestroyTensor(outTensor);
    aclrtFree(selfDeviceAddr);
    aclrtFree(outDeviceAddr);
    if (workspaceSize > 0) {
        aclrtFree(workspaceAddr);
    }
    aclrtDestroyStream(stream);
    aclrtResetDevice(deviceId);
    aclFinalize();

    return (passCount == totalNum) ? 0 : 1;
}
