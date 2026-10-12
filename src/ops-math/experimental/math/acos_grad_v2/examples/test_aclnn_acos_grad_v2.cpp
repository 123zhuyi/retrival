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
 * @file test_aclnn_acos_grad_v2.cpp
 * @brief aclnnAcosGradV2 调用示例与精度验证 (A2 / Ascend910B)
 *
 * 验证公式: z = -dy / sqrt(1 - y^2)
 * 覆盖 FP32 / FP16 / BF16 三种数据类型，y 取值落在 (-1, 1)。
 */

#include <iostream>
#include <vector>
#include <cmath>
#include <cstring>
#include "acl/acl.h"
#include "../op_api/aclnn_acos_grad_v2.h"

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

enum class TestDtype { FP32, FP16, BF16 };

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

// ---- dtype 编解码 ----
static void EncodeFp32(float v, uint8_t* out) { std::memcpy(out, &v, sizeof(float)); }

static void EncodeFp16(float v, uint8_t* out)
{
    uint32_t x = 0;
    std::memcpy(&x, &v, sizeof(uint32_t));
    uint16_t sign = (x >> 31) & 0x1;
    int32_t exp = ((x >> 23) & 0xff) - 127 + 15;
    uint32_t mantissa = x & 0x7fffff;
    uint16_t h = 0;
    if (exp <= 0) {
        h = sign << 15;
    } else if (exp >= 31) {
        h = (sign << 15) | (0x1f << 10);
    } else {
        h = (sign << 15) | (exp << 10) | (mantissa >> 13);
    }
    std::memcpy(out, &h, sizeof(uint16_t));
}

static float DecodeFp16(const uint8_t* in)
{
    uint16_t h = 0;
    std::memcpy(&h, in, sizeof(uint16_t));
    uint32_t sign = (h >> 15) & 0x1;
    uint32_t exp = (h >> 10) & 0x1f;
    uint32_t mantissa = h & 0x3ff;
    if (exp == 0) {
        if (mantissa == 0) {
            return sign ? -0.0f : 0.0f;
        }
        float val = mantissa / 1024.0f / 1024.0f;
        return sign ? -val : val;
    }
    if (exp == 31) {
        if (mantissa == 0) {
            return sign ? -INFINITY : INFINITY;
        }
        return NAN;
    }
    float val = (1.0f + mantissa / 1024.0f) * std::pow(2.0f, (int)exp - 15);
    return sign ? -val : val;
}

// bfloat16: 截断 float32 低 16 位，round-to-nearest-even
static void EncodeBf16(float v, uint8_t* out)
{
    uint32_t x = 0;
    std::memcpy(&x, &v, sizeof(uint32_t));
    uint32_t lsb = (x >> 16) & 0x1;
    uint32_t rounding_bias = 0x7FFFU + lsb;
    uint16_t bf = static_cast<uint16_t>((x + rounding_bias) >> 16);
    std::memcpy(out, &bf, sizeof(uint16_t));
}

static float DecodeBf16(const uint8_t* in)
{
    uint16_t bf = 0;
    std::memcpy(&bf, in, sizeof(uint16_t));
    uint32_t x = static_cast<uint32_t>(bf) << 16;
    float v = 0;
    std::memcpy(&v, &x, sizeof(float));
    return v;
}

static void Encode(float v, TestDtype dt, uint8_t* out)
{
    switch (dt) {
        case TestDtype::FP32:
            EncodeFp32(v, out);
            break;
        case TestDtype::FP16:
            EncodeFp16(v, out);
            break;
        case TestDtype::BF16:
            EncodeBf16(v, out);
            break;
    }
}

static float Decode(const uint8_t* in, TestDtype dt)
{
    switch (dt) {
        case TestDtype::FP32: {
            float v = 0;
            std::memcpy(&v, in, sizeof(float));
            return v;
        }
        case TestDtype::FP16:
            return DecodeFp16(in);
        case TestDtype::BF16:
            return DecodeBf16(in);
    }
    return 0.0f;
}

static size_t DtypeSize(TestDtype dt) { return (dt == TestDtype::FP32) ? sizeof(float) : sizeof(uint16_t); }

static aclDataType ToAclDtype(TestDtype dt)
{
    switch (dt) {
        case TestDtype::FP32:
            return aclDataType::ACL_FLOAT;
        case TestDtype::FP16:
            return aclDataType::ACL_FLOAT16;
        case TestDtype::BF16:
            return aclDataType::ACL_BF16;
    }
    return aclDataType::ACL_FLOAT;
}

static const char* DtypeName(TestDtype dt)
{
    switch (dt) {
        case TestDtype::FP32:
            return "FP32";
        case TestDtype::FP16:
            return "FP16";
        case TestDtype::BF16:
            return "BF16";
    }
    return "?";
}

// 创建一个 aclTensor：hostData 为已按 dtype 编码的字节流
int CreateAclTensor(const std::vector<uint8_t>& hostBytes, const std::vector<int64_t>& shape, void** deviceAddr,
                    aclDataType dataType, aclTensor** tensor)
{
    auto size = hostBytes.size();
    auto ret = aclrtMalloc(deviceAddr, size, ACL_MEM_MALLOC_HUGE_FIRST);
    CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclrtMalloc failed. ERROR: %d\n", ret); return ret);
    ret = aclrtMemcpy(*deviceAddr, size, hostBytes.data(), size, ACL_MEMCPY_HOST_TO_DEVICE);
    CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclrtMemcpy failed. ERROR: %d\n", ret); return ret);

    std::vector<int64_t> strides(shape.size(), 1);
    for (int64_t i = shape.size() - 2; i >= 0; i--) {
        strides[i] = shape[i + 1] * strides[i + 1];
    }

    *tensor = aclCreateTensor(shape.data(), shape.size(), dataType, strides.data(), 0, aclFormat::ACL_FORMAT_ND,
                              shape.data(), shape.size(), *deviceAddr);
    return 0;
}

// 运行单个 dtype 的精度验证，返回通过元素数
int RunOneDtype(TestDtype dt, aclrtStream stream, const std::vector<int64_t>& shape)
{
    int64_t totalNum = GetShapeSize(shape);
    size_t dsize = DtypeSize(dt);

    std::vector<uint8_t> yBytes(totalNum * dsize, 0);
    std::vector<uint8_t> dyBytes(totalNum * dsize, 0);
    std::vector<float> yFloat(totalNum);
    std::vector<float> dyFloat(totalNum);

    for (int64_t i = 0; i < totalNum; i++) {
        // y 取值 (-0.98, 0.98)，避开定义域边界
        float y_val = -0.98f + (1.96f * i) / totalNum;
        // dy 取值 (0.5, 1.5)
        float dy_val = 0.5f + (float)i / totalNum;

        yFloat[i] = y_val;
        dyFloat[i] = dy_val;

        Encode(y_val, dt, yBytes.data() + i * dsize);
        Encode(dy_val, dt, dyBytes.data() + i * dsize);
    }

    aclTensor* yTensor = nullptr;
    void* yDeviceAddr = nullptr;
    aclTensor* dyTensor = nullptr;
    void* dyDeviceAddr = nullptr;
    aclTensor* zTensor = nullptr;
    void* zDeviceAddr = nullptr;
    void* workspaceAddr = nullptr;
    uint64_t workspaceSize = 0;
    // 提前声明所有带初始化的变量，避免 goto 跨越其初始化（gcc 严格模式）
    std::vector<uint8_t> zBytes(totalNum * dsize, 0);
    std::vector<uint8_t> resultBytes(totalNum * dsize, 0);
    aclOpExecutor* executor = nullptr;
    int passCount = 0;
    float relTol = (dt == TestDtype::FP32) ? 1e-5f : (dt == TestDtype::FP16 ? 1e-3f : 1e-2f);
    float absTol = (dt == TestDtype::FP32) ? 1e-5f : (dt == TestDtype::FP16 ? 1e-3f : 1e-2f);
    int retVal = -1; // 默认失败；成功路径末尾置为 passCount

    auto ret = CreateAclTensor(yBytes, shape, &yDeviceAddr, ToAclDtype(dt), &yTensor);
    CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("CreateAclTensor y failed. ERROR: %d\n", ret); goto cleanup);

    ret = CreateAclTensor(dyBytes, shape, &dyDeviceAddr, ToAclDtype(dt), &dyTensor);
    CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("CreateAclTensor dy failed. ERROR: %d\n", ret); goto cleanup);

    ret = CreateAclTensor(zBytes, shape, &zDeviceAddr, ToAclDtype(dt), &zTensor);
    CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("CreateAclTensor z failed. ERROR: %d\n", ret); goto cleanup);

    ret = aclnnAcosGradV2GetWorkspaceSize(yTensor, dyTensor, zTensor, &workspaceSize, &executor);
    CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclnnAcosGradV2GetWorkspaceSize failed. ERROR: %d\n", ret); goto cleanup);

    if (workspaceSize > 0) {
        ret = aclrtMalloc(&workspaceAddr, workspaceSize, ACL_MEM_MALLOC_HUGE_FIRST);
        CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("allocate workspace failed. ERROR: %d\n", ret); goto cleanup);
    }

    ret = aclnnAcosGradV2(workspaceAddr, workspaceSize, executor, stream);
    CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclnnAcosGradV2 failed. ERROR: %d\n", ret); goto cleanup);

    ret = aclrtSynchronizeStream(stream);
    CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclrtSynchronizeStream failed. ERROR: %d\n", ret); goto cleanup);

    ret = aclrtMemcpy(resultBytes.data(), resultBytes.size(), zDeviceAddr, resultBytes.size(),
                      ACL_MEMCPY_DEVICE_TO_HOST);
    CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("copy result failed. ERROR: %d\n", ret); goto cleanup);

    // 精度比对：以“量化后的 y 和 dy”重算 golden，避免输入量化带入的偏差
    // 算子内部 BF16/FP16 -> FP32 计算 -> 舍回原类型，误差主要来自最后一步舍入(<=0.5 ULP)
    // passCount / relTol / absTol 已在函数开头声明
    for (int64_t i = 0; i < totalNum; i++) {
        float result = Decode(resultBytes.data() + i * dsize, dt);
        float yQuant = Decode(yBytes.data() + i * dsize, dt);
        float dyQuant = Decode(dyBytes.data() + i * dsize, dt);
        double one_minus_y2 = 1.0 - static_cast<double>(yQuant) * static_cast<double>(yQuant);
        float expQuant = (one_minus_y2 > 0.0) ? dyQuant * static_cast<float>(-1.0 / std::sqrt(one_minus_y2)) : NAN;
        float diff = std::fabs(result - expQuant);
        float tol = absTol + relTol * std::fabs(expQuant);
        if (diff <= tol) {
            passCount++;
        } else {
            static int printed = 0;
            if (printed < 8) {
                LOG_PRINT("FAIL[%s][%ld]: y=%.5f, dy=%.5f, expected=%.6f, result=%.6f, diff=%.6f\n", DtypeName(dt), i,
                          yFloat[i], dyFloat[i], expQuant, result, diff);
                printed++;
            }
        }
    }

    LOG_PRINT("[%s] 总元素数: %ld, 通过: %d %s\n", DtypeName(dt), totalNum, passCount,
              (passCount == totalNum) ? "(PASS)" : "(FAIL)");
    retVal = passCount;

cleanup:
    if (yTensor != nullptr) {
        aclDestroyTensor(yTensor);
    }
    if (dyTensor != nullptr) {
        aclDestroyTensor(dyTensor);
    }
    if (zTensor != nullptr) {
        aclDestroyTensor(zTensor);
    }
    if (yDeviceAddr != nullptr) {
        aclrtFree(yDeviceAddr);
    }
    if (dyDeviceAddr != nullptr) {
        aclrtFree(dyDeviceAddr);
    }
    if (zDeviceAddr != nullptr) {
        aclrtFree(zDeviceAddr);
    }
    if (workspaceAddr != nullptr) {
        aclrtFree(workspaceAddr);
    }

    return retVal;
}

int main()
{
    int32_t deviceId = 0;
    aclrtStream stream;
    auto ret = Init(deviceId, &stream);
    CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("Init acl failed. ERROR: %d\n", ret); return ret);

    // 用一个略大的 shape 以触发多核切分 (8192 元素)
    std::vector<int64_t> shape = {32, 256};

    LOG_PRINT("=== AcosGradV2 (A2) 精度验证, shape={32,256} ===\n");
    int allPass = 0;
    int totalDtypes = 0;
    for (auto dt : {TestDtype::FP32, TestDtype::FP16, TestDtype::BF16}) {
        totalDtypes++;
        int passed = RunOneDtype(dt, stream, shape);
        if (passed == GetShapeSize(shape)) {
            allPass++;
        }
    }

    LOG_PRINT("=== 汇总: %d/%d dtype 全部通过 ===\n", allPass, totalDtypes);

    aclrtDestroyStream(stream);
    aclrtResetDevice(deviceId);
    aclFinalize();

    return (allPass == totalDtypes) ? 0 : 1;
}
