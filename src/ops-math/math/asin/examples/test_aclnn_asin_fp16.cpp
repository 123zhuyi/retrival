/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include <iostream>
#include <memory>
#include <type_traits>
#include <vector>
#include <cmath>
#include <stdint.h>
#include "acl/acl.h"
#include "aclnnop/aclnn_asin.h"

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

// Float16 helper functions using standard IEEE 754
static inline float Float16ToFloat(uint16_t value)
{
    // IEEE 754 float16: 1 sign, 5 exponent, 10 mantissa
    unsigned int sign = (value >> 15) & 0x1;
    unsigned int exponent = (value >> 10) & 0x1f;
    unsigned int mantissa = value & 0x3ff;

    float result;
    if (exponent == 0) {
        // Denormalized or zero
        result = mantissa * 0.0000019073486328125f; // 2^-24
    } else if (exponent == 31) {
        // Infinity or NaN
        result = (mantissa == 0) ? 1.0f / 0.0f : 0.0f / 0.0f;
    } else {
        // Normalized number
        result = (1.0f + mantissa * 0.0009765625f) * powf(2.0f, (int)exponent - 15);
    }
    return sign ? -result : result;
}

static inline uint16_t FloatToFloat16(float value)
{
    // Using bit-level conversion for 0.5
    uint32_t bits = *reinterpret_cast<uint32_t*>(&value);
    uint16_t sign = (bits >> 16) & 0x8000;
    int32_t exponent = ((bits >> 23) & 0xff) - 127 + 15;
    uint32_t mantissa = bits & 0x7fffff;

    if (exponent <= 0) {
        // Underflow to denormalized or zero
        return sign;
    } else if (exponent >= 31) {
        // Overflow to infinity
        return sign | 0x7c00; // Infinity
    }

    uint16_t fp16 = sign | (exponent << 10) | (mantissa >> 13);
    return fp16;
}

using StreamPtr = std::unique_ptr<std::remove_pointer<aclrtStream>::type, decltype(&aclrtDestroyStream)>;
using DeviceMemPtr = std::unique_ptr<void, decltype(&aclrtFree)>;
using TensorPtr = std::unique_ptr<aclTensor, decltype(&aclDestroyTensor)>;

void PrintOutResultFp16(std::vector<int64_t>& shape, DeviceMemPtr& deviceAddr,
                        const std::vector<uint16_t>& selfHostData)
{
    auto size = GetShapeSize(shape);
    std::vector<uint16_t> resultData(size, 0);
    auto ret = aclrtMemcpy(resultData.data(), resultData.size() * sizeof(resultData[0]), deviceAddr.get(),
                           size * sizeof(resultData[0]), ACL_MEMCPY_DEVICE_TO_HOST);
    CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("copy result from device to host failed. ERROR: %d\n", ret); return);
    for (int64_t i = 0; i < size; i++) {
        float inputFp32 = Float16ToFloat(selfHostData[i]);
        float resultFp32 = Float16ToFloat(resultData[i]);
        LOG_PRINT("asin input[%ld] is: %f (0x%04x), result[%ld] is: %f (0x%04x)\n", i, inputFp32, selfHostData[i], i,
                  resultFp32, resultData[i]);
    }
}

int Init(int32_t deviceId, StreamPtr& stream, bool& initialized, bool& deviceSet)
{
    auto ret = aclInit(nullptr);
    CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclInit failed. ERROR: %d\n", ret); return ret);
    initialized = true;
    ret = aclrtSetDevice(deviceId);
    CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclrtSetDevice failed. ERROR: %d\n", ret); return ret);
    deviceSet = true;
    aclrtStream rawStream = nullptr;
    ret = aclrtCreateStream(&rawStream);
    CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclrtCreateStream failed. ERROR: %d\n", ret); return ret);
    stream.reset(rawStream);
    return 0;
}

template <typename T>
int CreateAclTensor(const std::vector<T>& hostData, const std::vector<int64_t>& shape, aclDataType dataType,
                    DeviceMemPtr& deviceAddr, TensorPtr& tensor)
{
    auto size = GetShapeSize(shape) * sizeof(T);
    void* rawDeviceAddr = nullptr;
    auto ret = aclrtMalloc(&rawDeviceAddr, size, ACL_MEM_MALLOC_HUGE_FIRST);
    CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclrtMalloc failed. ERROR: %d\n", ret); return ret);
    deviceAddr.reset(rawDeviceAddr);
    ret = aclrtMemcpy(deviceAddr.get(), size, hostData.data(), size, ACL_MEMCPY_HOST_TO_DEVICE);
    CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclrtMemcpy failed. ERROR: %d\n", ret); return ret);

    std::vector<int64_t> strides(shape.size(), 1);
    for (int64_t i = shape.size() - 2; i >= 0; i--) {
        strides[i] = shape[i + 1] * strides[i + 1];
    }

    aclTensor* rawTensor = aclCreateTensor(shape.data(), shape.size(), dataType, strides.data(), 0,
                                           aclFormat::ACL_FORMAT_ND, shape.data(), shape.size(), deviceAddr.get());
    CHECK_RET(rawTensor != nullptr, LOG_PRINT("aclCreateTensor failed.\n"); return ACL_ERROR_FAILURE);
    tensor.reset(rawTensor);
    return 0;
}

int main()
{
    int32_t deviceId = 0;
    bool initialized = false;
    bool deviceSet = false;
    std::shared_ptr<void> aclGuard(nullptr, [&](void*) {
        if (deviceSet) {
            aclrtResetDevice(deviceId);
        }
        if (initialized) {
            aclFinalize();
        }
    });
    StreamPtr stream(nullptr, &aclrtDestroyStream);
    auto ret = Init(deviceId, stream, initialized, deviceSet);
    CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("Init acl failed. ERROR: %d\n", ret); return ret);

    // 8x8 float16 matrix with 0.5
    std::vector<int64_t> selfShape = {8, 8};
    std::vector<uint16_t> selfHostData(64);
    float inputVal = 0.5f;
    uint16_t inputFp16 = FloatToFloat16(inputVal);
    printf("Input value: %f, as float16: 0x%04x\n", inputVal, inputFp16);
    for (int i = 0; i < 64; i++) {
        selfHostData[i] = inputFp16;
    }

    TensorPtr self(nullptr, &aclDestroyTensor);
    DeviceMemPtr selfDeviceAddr(nullptr, &aclrtFree);
    ret = CreateAclTensor(selfHostData, selfShape, aclDataType::ACL_FLOAT16, selfDeviceAddr, self);
    CHECK_RET(ret == ACL_SUCCESS, return ret);

    TensorPtr out(nullptr, &aclDestroyTensor);
    DeviceMemPtr outDeviceAddr(nullptr, &aclrtFree);
    std::vector<uint16_t> outHostData(64, 0);
    ret = CreateAclTensor(outHostData, selfShape, aclDataType::ACL_FLOAT16, outDeviceAddr, out);
    CHECK_RET(ret == ACL_SUCCESS, return ret);

    uint64_t workspaceSize = 0;
    aclOpExecutor* executor;

    ret = aclnnAsinGetWorkspaceSize(self.get(), out.get(), &workspaceSize, &executor);
    CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclnnAsinGetWorkspaceSize failed. ERROR: %d\n", ret); return ret);

    DeviceMemPtr workspaceAddr(nullptr, &aclrtFree);
    if (workspaceSize > static_cast<uint64_t>(0)) {
        void* rawWorkspaceAddr = nullptr;
        ret = aclrtMalloc(&rawWorkspaceAddr, workspaceSize, ACL_MEM_MALLOC_HUGE_FIRST);
        CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("allocate workspace failed. ERROR: %d\n", ret); return ret);
        workspaceAddr.reset(rawWorkspaceAddr);
    }

    ret = aclnnAsin(workspaceAddr.get(), workspaceSize, executor, stream.get());
    CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclnnAsin failed. ERROR: %d\n", ret); return ret);

    ret = aclrtSynchronizeStream(stream.get());
    CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclrtSynchronizeStream failed. ERROR: %d\n", ret); return ret);

    PrintOutResultFp16(selfShape, outDeviceAddr, selfHostData);

    return 0;
}
