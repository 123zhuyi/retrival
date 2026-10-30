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

#include <iostream>
#include <vector>
#include <cmath>
#include <cstdlib>
#include "acl/acl.h"
#include "aclnn_log_add_exp.h"

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

// CPU 端 golden 计算：数值稳定的 log(exp(x) + exp(y))
static float LogAddExpGolden(float x, float y)
{
    float m = std::max(x, y);
    return m + std::log1pf(std::exp(-std::fabs(x - y)));
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

int main()
{
    // 1. 初始化 device/stream
    int32_t deviceId = 0;
    aclrtStream stream;
    auto ret = Init(deviceId, &stream);
    CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("Init acl failed. ERROR: %d\n", ret); return ret);

    // 2. 构造输入与输出（非广播：x、y、out 同 shape）
    std::vector<int64_t> xShape = {2, 4};
    std::vector<int64_t> yShape = {2, 4};
    std::vector<int64_t> outShape = {2, 4};
    int64_t totalSize = GetShapeSize(outShape);

    std::vector<float> xHostData = {-2.0f, -1.0f, 0.0f, 1.0f, 2.0f, 3.0f, 4.0f, 5.0f};
    std::vector<float> yHostData = {1.0f, 2.0f, 3.0f, -1.0f, 0.0f, 1.0f, -2.0f, -3.0f};
    std::vector<float> outHostData(totalSize, 0.0f);

    void* xDeviceAddr = nullptr;
    void* yDeviceAddr = nullptr;
    void* outDeviceAddr = nullptr;
    aclTensor* x = nullptr;
    aclTensor* y = nullptr;
    aclTensor* out = nullptr;

    ret = CreateAclTensor(xHostData, xShape, &xDeviceAddr, aclDataType::ACL_FLOAT, &x);
    CHECK_RET(ret == ACL_SUCCESS, return ret);
    ret = CreateAclTensor(yHostData, yShape, &yDeviceAddr, aclDataType::ACL_FLOAT, &y);
    CHECK_RET(ret == ACL_SUCCESS, return ret);
    ret = CreateAclTensor(outHostData, outShape, &outDeviceAddr, aclDataType::ACL_FLOAT, &out);
    CHECK_RET(ret == ACL_SUCCESS, return ret);

    // 3. 调用 aclnnLogAddExp 第一段接口
    uint64_t workspaceSize = 0;
    aclOpExecutor* executor;
    ret = aclnnLogAddExpGetWorkspaceSize(x, y, out, &workspaceSize, &executor);
    CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclnnLogAddExpGetWorkspaceSize failed. ERROR: %d\n", ret); return ret);

    void* workspaceAddr = nullptr;
    if (workspaceSize > 0) {
        ret = aclrtMalloc(&workspaceAddr, workspaceSize, ACL_MEM_MALLOC_HUGE_FIRST);
        CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("allocate workspace failed. ERROR: %d\n", ret); return ret);
    }

    // 4. 调用 aclnnLogAddExp 第二段接口
    ret = aclnnLogAddExp(workspaceAddr, workspaceSize, executor, stream);
    CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclnnLogAddExp failed. ERROR: %d\n", ret); return ret);

    // 5. 同步等待任务执行结束
    ret = aclrtSynchronizeStream(stream);
    CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclrtSynchronizeStream failed. ERROR: %d\n", ret); return ret);

    // 6. 拷贝输出回 host
    std::vector<float> resultData(totalSize, 0.0f);
    ret = aclrtMemcpy(
        resultData.data(), resultData.size() * sizeof(float), outDeviceAddr, totalSize * sizeof(float),
        ACL_MEMCPY_DEVICE_TO_HOST);
    CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("copy result from device to host failed. ERROR: %d\n", ret); return ret);

    // 7. 与 CPU golden 比对（FP32 atol=1e-4, rtol=1e-4）
    int failCount = 0;
    const float atol = 1e-4f;
    const float rtol = 1e-4f;
    LOG_PRINT("=== aclnnLogAddExp result vs golden ===\n");
    for (int64_t i = 0; i < totalSize; i++) {
        float gold = LogAddExpGolden(xHostData[i], yHostData[i]);
        float diff = std::fabs(resultData[i] - gold);
        bool ok = diff <= (atol + rtol * std::fabs(gold));
        LOG_PRINT(
            "[%ld] x=%8.4f y=%8.4f out=%10.6f gold=%10.6f diff=%.2e %s\n", i, xHostData[i], yHostData[i], resultData[i],
            gold, diff, ok ? "OK" : "FAIL");
        if (!ok) {
            failCount++;
        }
    }

    // 8. 释放 aclTensor
    aclDestroyTensor(x);
    aclDestroyTensor(y);
    aclDestroyTensor(out);

    // 9. 释放 device 资源
    aclrtFree(xDeviceAddr);
    aclrtFree(yDeviceAddr);
    aclrtFree(outDeviceAddr);
    if (workspaceSize > 0) {
        aclrtFree(workspaceAddr);
    }
    aclrtDestroyStream(stream);
    aclrtResetDevice(deviceId);
    aclFinalize();

    if (failCount == 0) {
        LOG_PRINT("=== PASS: all %ld elements match golden ===\n", totalSize);
        return 0;
    } else {
        LOG_PRINT("=== FAIL: %d / %ld elements mismatch ===\n", failCount, totalSize);
        return 1;
    }
}
