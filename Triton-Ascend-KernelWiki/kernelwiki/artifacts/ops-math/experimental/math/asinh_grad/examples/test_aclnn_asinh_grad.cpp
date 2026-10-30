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
 * @file test_aclnn_asinh_grad.cpp
 * @brief AsinhGrad 算子 aclnn 两段式调用示例
 *
 * 功能：计算 asinh 反向梯度 z = dy / cosh(y)
 * 其中 y 为前向 asinh 的输出，dy 为上游梯度，z 为输出梯度。
 *
 * 本示例展示如何通过 aclnn 接口调用自定义 AsinhGrad 算子。
 * 示例使用 FLOAT32 数据类型，shape = [8]。
 */

#include <iostream>
#include <vector>
#include <cmath>
#include <cstring>
#include "acl/acl.h"
#include "aclnn_asinh_grad.h"

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

void PrintResult(
    const std::vector<int64_t>& shape, void* deviceAddr, const std::vector<float>& yData,
    const std::vector<float>& dyData)
{
    auto size = GetShapeSize(shape);
    std::vector<float> resultData(size, 0);
    auto ret = aclrtMemcpy(
        resultData.data(), resultData.size() * sizeof(float), deviceAddr, size * sizeof(float),
        ACL_MEMCPY_DEVICE_TO_HOST);
    CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("copy result from device to host failed. ERROR: %d\n", ret); return);

    LOG_PRINT("AsinhGrad results (z = dy / cosh(y)):\n");
    for (int64_t i = 0; i < size; i++) {
        float expected = dyData[i] / std::cosh(yData[i]);
        LOG_PRINT(
            "  y[%ld]=%.4f, dy[%ld]=%.4f => z[%ld]=%.6f (expected=%.6f)\n", i, yData[i], i, dyData[i], i, resultData[i],
            expected);
    }
}

int main()
{
    // 1. ACL 初始化
    int32_t deviceId = 0;
    aclrtStream stream;
    auto ret = Init(deviceId, &stream);
    CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("Init acl failed. ERROR: %d\n", ret); return ret);

    // 2. 构造输入和输出
    // y: 前向 asinh 的输出, shape=[8], dtype=FLOAT32
    // dy: 上游梯度, shape=[8], dtype=FLOAT32
    std::vector<int64_t> shape = {8};

    std::vector<float> yHostData = {0.0f, 0.5f, -0.5f, 1.0f, -1.0f, 2.0f, -2.0f, 0.1f};
    std::vector<float> dyHostData = {1.0f, 1.0f, 1.0f, 1.0f, 1.0f, 1.0f, 1.0f, 1.0f};

    // 输入 y
    aclTensor* y = nullptr;
    void* yDeviceAddr = nullptr;
    ret = CreateAclTensor(yHostData, shape, &yDeviceAddr, aclDataType::ACL_FLOAT, &y);
    CHECK_RET(ret == ACL_SUCCESS, return ret);

    // 输入 dy
    aclTensor* dy = nullptr;
    void* dyDeviceAddr = nullptr;
    ret = CreateAclTensor(dyHostData, shape, &dyDeviceAddr, aclDataType::ACL_FLOAT, &dy);
    CHECK_RET(ret == ACL_SUCCESS, return ret);

    // 输出 z: 与 y 同 shape 同 dtype
    aclTensor* z = nullptr;
    void* zDeviceAddr = nullptr;
    std::vector<float> zHostData(8, 0.0f);
    ret = CreateAclTensor(zHostData, shape, &zDeviceAddr, aclDataType::ACL_FLOAT, &z);
    CHECK_RET(ret == ACL_SUCCESS, return ret);

    // 3. 调用 aclnnAsinhGrad 第一段接口：获取 workspace 大小和 executor
    uint64_t workspaceSize = 0;
    aclOpExecutor* executor = nullptr;
    ret = aclnnAsinhGradGetWorkspaceSize(y, dy, z, &workspaceSize, &executor);
    CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclnnAsinhGradGetWorkspaceSize failed. ERROR: %d\n", ret); return ret);

    // 4. 申请 workspace（如果需要）
    void* workspaceAddr = nullptr;
    if (workspaceSize > static_cast<uint64_t>(0)) {
        ret = aclrtMalloc(&workspaceAddr, workspaceSize, ACL_MEM_MALLOC_HUGE_FIRST);
        CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("allocate workspace failed. ERROR: %d\n", ret); return ret);
    }

    // 5. 调用 aclnnAsinhGrad 第二段接口：执行计算
    ret = aclnnAsinhGrad(workspaceAddr, workspaceSize, executor, stream);
    CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclnnAsinhGrad failed. ERROR: %d\n", ret); return ret);

    // 6. 同步等待计算完成
    ret = aclrtSynchronizeStream(stream);
    CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclrtSynchronizeStream failed. ERROR: %d\n", ret); return ret);

    // 7. 打印输出结果
    PrintResult(shape, zDeviceAddr, yHostData, dyHostData);

    // 8. 释放资源
    aclDestroyTensor(y);
    aclDestroyTensor(dy);
    aclDestroyTensor(z);
    aclrtFree(yDeviceAddr);
    aclrtFree(dyDeviceAddr);
    aclrtFree(zDeviceAddr);
    if (workspaceSize > static_cast<uint64_t>(0)) {
        aclrtFree(workspaceAddr);
    }
    aclrtDestroyStream(stream);
    aclrtResetDevice(deviceId);
    aclFinalize();

    return 0;
}
