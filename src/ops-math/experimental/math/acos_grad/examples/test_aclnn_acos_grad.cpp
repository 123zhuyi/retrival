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
 * @file test_aclnn_acos_grad.cpp
 * @brief aclnnAcosGrad 调用示例（fp16）
 *
 * 用法：
 *   编译后直接运行，验证 fp16 acos_grad 在 NPU 上计算正确性
 *   公式：x_grad = y_grad * (-1 / sqrt(1 - x^2))
 *   输入 x 值域 [-1, 1]，输入 y_grad 为上游梯度
 */

#include <iostream>
#include <vector>
#include <cmath>
#include "acl/acl.h"
#include "../op_api/aclnn_acos_grad.h"

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

    std::vector<int64_t> shape = {4, 8};
    int64_t totalNum = GetShapeSize(shape);

    std::vector<uint16_t> yGradHostFp16(totalNum);
    std::vector<uint16_t> xHostFp16(totalNum);
    std::vector<float> xHostFloat(totalNum);
    std::vector<float> yGradHostFloat(totalNum);
    std::vector<float> expectedResult(totalNum);

    for (int64_t i = 0; i < totalNum; i++) {
        float x_val = -0.9f + (1.8f * i) / totalNum;
        float y_grad_val = 0.5f + (float)i / totalNum;

        xHostFloat[i] = x_val;
        yGradHostFloat[i] = y_grad_val;
        xHostFp16[i] = FloatToFp16(x_val);
        yGradHostFp16[i] = FloatToFp16(y_grad_val);

        double one_minus_x2 = 1.0 - static_cast<double>(x_val) * static_cast<double>(x_val);
        if (one_minus_x2 > 0.0) {
            expectedResult[i] = y_grad_val * (-1.0f / std::sqrt(one_minus_x2));
        } else {
            expectedResult[i] = 0.0f;
        }
    }

    aclTensor* yGradTensor = nullptr;
    void* yGradDeviceAddr = nullptr;
    ret = CreateAclTensor(yGradHostFp16, shape, &yGradDeviceAddr, aclDataType::ACL_FLOAT16, &yGradTensor);
    CHECK_RET(ret == ACL_SUCCESS, return ret);

    aclTensor* xTensor = nullptr;
    void* xDeviceAddr = nullptr;
    ret = CreateAclTensor(xHostFp16, shape, &xDeviceAddr, aclDataType::ACL_FLOAT16, &xTensor);
    CHECK_RET(ret == ACL_SUCCESS, return ret);

    aclTensor* outTensor = nullptr;
    void* outDeviceAddr = nullptr;
    std::vector<uint16_t> outHostFp16(totalNum, 0);
    ret = CreateAclTensor(outHostFp16, shape, &outDeviceAddr, aclDataType::ACL_FLOAT16, &outTensor);
    CHECK_RET(ret == ACL_SUCCESS, return ret);

    uint64_t workspaceSize = 0;
    aclOpExecutor* executor = nullptr;
    ret = aclnnAcosGradGetWorkspaceSize(yGradTensor, xTensor, outTensor, &workspaceSize, &executor);
    CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclnnAcosGradGetWorkspaceSize failed. ERROR: %d\n", ret); return ret);

    void* workspaceAddr = nullptr;
    if (workspaceSize > 0) {
        ret = aclrtMalloc(&workspaceAddr, workspaceSize, ACL_MEM_MALLOC_HUGE_FIRST);
        CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("allocate workspace failed. ERROR: %d\n", ret); return ret);
    }

    ret = aclnnAcosGrad(workspaceAddr, workspaceSize, executor, stream);
    CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclnnAcosGrad failed. ERROR: %d\n", ret); return ret);

    ret = aclrtSynchronizeStream(stream);
    CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclrtSynchronizeStream failed. ERROR: %d\n", ret); return ret);

    std::vector<uint16_t> resultFp16(totalNum, 0);
    ret = aclrtMemcpy(
        resultFp16.data(), totalNum * sizeof(uint16_t), outDeviceAddr, totalNum * sizeof(uint16_t),
        ACL_MEMCPY_DEVICE_TO_HOST);
    CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("copy result failed. ERROR: %d\n", ret); return ret);

    LOG_PRINT("=== AcosGrad fp16 精度验证 ===\n");
    int passCount = 0;
    for (int64_t i = 0; i < totalNum; i++) {
        float result = Fp16ToFloat(resultFp16[i]);
        float expected = expectedResult[i];
        float absDiff = std::fabs(result - expected);
        float tolerance = 1e-3f + 1e-3f * std::fabs(expected);
        if (absDiff <= tolerance) {
            passCount++;
        } else {
            LOG_PRINT(
                "FAIL[%ld]: x=%.4f, y_grad=%.4f, expected=%.4f, result=%.4f, diff=%.6f\n", i, xHostFloat[i],
                yGradHostFloat[i], expected, result, absDiff);
        }
    }

    LOG_PRINT("总元素数: %ld, 通过: %d\n", totalNum, passCount);
    if (passCount == totalNum) {
        LOG_PRINT("=== 精度验证通过 ===\n");
    } else {
        LOG_PRINT("=== 精度验证失败 ===\n");
    }

    aclDestroyTensor(yGradTensor);
    aclDestroyTensor(xTensor);
    aclDestroyTensor(outTensor);
    aclrtFree(yGradDeviceAddr);
    aclrtFree(xDeviceAddr);
    aclrtFree(outDeviceAddr);
    if (workspaceSize > 0) {
        aclrtFree(workspaceAddr);
    }
    aclrtDestroyStream(stream);
    aclrtResetDevice(deviceId);
    aclFinalize();

    return (passCount == totalNum) ? 0 : 1;
}
