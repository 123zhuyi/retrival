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
#include <vector>
#include "acl/acl.h"
#include "aclnn_diag_part.h"

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
    // 固定写法，资源初始化
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
    // 调用aclrtMalloc申请device侧内存
    auto ret = aclrtMalloc(deviceAddr, size, ACL_MEM_MALLOC_HUGE_FIRST);
    CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclrtMalloc failed. ERROR: %d\n", ret); return ret);
    // 调用aclrtMemcpy将host侧数据拷贝到device侧内存上
    ret = aclrtMemcpy(*deviceAddr, size, hostData.data(), size, ACL_MEMCPY_HOST_TO_DEVICE);
    CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclrtMemcpy failed. ERROR: %d\n", ret); return ret);
    // 计算连续tensor的strides
    std::vector<int64_t> strides(shape.size(), 1);
    for (int64_t i = shape.size() - 2; i >= 0; i--) {
        strides[i] = shape[i + 1] * strides[i + 1];
    }
    // 调用aclCreateTensor接口创建aclTensor
    *tensor = aclCreateTensor(
        shape.data(), shape.size(), dataType, strides.data(), 0, aclFormat::ACL_FORMAT_ND, shape.data(), shape.size(),
        *deviceAddr);
    return 0;
}

int testFp32()
{
    LOG_PRINT("\n========== Test for FP32 ==========\n");
    // 1. device/stream初始化
    int32_t deviceId = 0;
    aclrtStream stream;
    auto ret = Init(deviceId, &stream);
    CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("Init acl failed. ERROR: %d\n", ret); return ret);

    // 2. 构造输入与输出
    // diag_part: [N, N] -> [N] (提取对角线元素)
    int64_t sideLen = 4;
    std::vector<int64_t> xShape = {sideLen, sideLen};
    std::vector<int64_t> yShape = {sideLen};
    void* xDeviceAddr = nullptr;
    void* yDeviceAddr = nullptr;
    aclTensor* x = nullptr;
    aclTensor* y = nullptr;

    // 输入: 4x4 矩阵
    std::vector<float> xHostData = {1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16};
    // 期望输出: 对角线元素 [1, 6, 11, 16]
    std::vector<float> expectedData = {1, 6, 11, 16};
    std::vector<float> yHostData(sideLen, 0);

    // 创建 x aclTensor
    ret = CreateAclTensor(xHostData, xShape, &xDeviceAddr, aclDataType::ACL_FLOAT, &x);
    CHECK_RET(ret == ACL_SUCCESS, return ret);

    // 创建 y aclTensor
    ret = CreateAclTensor(yHostData, yShape, &yDeviceAddr, aclDataType::ACL_FLOAT, &y);
    CHECK_RET(ret == ACL_SUCCESS, return ret);

    // 3. 调用CANN算子库API
    uint64_t workspaceSize = 0;
    aclOpExecutor* executor;
    // 调用 aclnnDiagPart 第一段接口
    ret = aclnnDiagPartGetWorkspaceSize(x, y, &workspaceSize, &executor);
    CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclnnDiagPartGetWorkspaceSize failed. ERROR: %d\n", ret); return ret);
    // 根据第一段接口计算出的workspaceSize申请device内存
    void* workspaceAddr = nullptr;
    if (workspaceSize > 0) {
        ret = aclrtMalloc(&workspaceAddr, workspaceSize, ACL_MEM_MALLOC_HUGE_FIRST);
        CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("allocate workspace failed. ERROR: %d\n", ret); return ret);
    }
    // 调用 aclnnDiagPart 第二段接口
    ret = aclnnDiagPart(workspaceAddr, workspaceSize, executor, stream);
    CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclnnDiagPart failed. ERROR: %d\n", ret); return ret);

    // 4. 同步等待任务执行结束
    ret = aclrtSynchronizeStream(stream);
    CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclrtSynchronizeStream failed. ERROR: %d\n", ret); return ret);

    // 5. 获取输出的值，将device侧内存上的结果拷贝至host侧
    auto size = GetShapeSize(yShape);
    std::vector<float> resultData(size, 0);
    ret = aclrtMemcpy(
        resultData.data(), resultData.size() * sizeof(resultData[0]), yDeviceAddr, size * sizeof(resultData[0]),
        ACL_MEMCPY_DEVICE_TO_HOST);
    CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("copy result from device to host failed. ERROR: %d\n", ret); return ret);

    // 6. 输出结果并验证
    LOG_PRINT("Result:   ");
    for (int64_t i = 0; i < size; i++) {
        LOG_PRINT("%.2f ", resultData[i]);
    }
    LOG_PRINT("\nExpected: ");
    for (int64_t i = 0; i < expectedData.size(); i++) {
        LOG_PRINT("%.2f ", expectedData[i]);
    }
    LOG_PRINT("\n");

    // 验证结果
    bool pass = true;
    for (int64_t i = 0; i < size; i++) {
        if (std::abs(resultData[i] - expectedData[i]) > 0.001) {
            pass = false;
            break;
        }
    }

    if (pass) {
        LOG_PRINT("[PASS] FP32 test PASSED!\n");
    } else {
        LOG_PRINT("[FAIL] FP32 test FAILED!\n");
    }

    // 7. 释放aclTensor
    aclDestroyTensor(x);
    aclDestroyTensor(y);

    // 8. 释放device资源
    aclrtFree(xDeviceAddr);
    aclrtFree(yDeviceAddr);
    if (workspaceSize > 0) {
        aclrtFree(workspaceAddr);
    }
    aclrtDestroyStream(stream);
    aclrtResetDevice(deviceId);
    aclFinalize();
    return pass ? 0 : 1;
}

int testFp16()
{
    LOG_PRINT("\n========== Test for FP16 ==========\n");
    int32_t deviceId = 0;
    aclrtStream stream;
    auto ret = Init(deviceId, &stream);
    CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("Init acl failed. ERROR: %d\n", ret); return ret);

    int64_t sideLen = 8;
    std::vector<int64_t> xShape = {sideLen, sideLen};
    std::vector<int64_t> yShape = {sideLen};
    void* xDeviceAddr = nullptr;
    void* yDeviceAddr = nullptr;
    aclTensor* x = nullptr;
    aclTensor* y = nullptr;

    // 输入: 8x8 矩阵，对角线元素为 1, 10, 19, 28, 37, 46, 55, 64
    std::vector<aclFloat16> xHostData;
    for (int i = 0; i < sideLen * sideLen; i++) {
        xHostData.push_back(aclFloat16(i + 1));
    }
    // 期望输出: 对角线元素 [1, 10, 19, 28, 37, 46, 55, 64]
    std::vector<float> expectedData = {1, 10, 19, 28, 37, 46, 55, 64};
    std::vector<aclFloat16> yHostData(sideLen, 0);

    ret = CreateAclTensor(xHostData, xShape, &xDeviceAddr, aclDataType::ACL_FLOAT16, &x);
    CHECK_RET(ret == ACL_SUCCESS, return ret);
    ret = CreateAclTensor(yHostData, yShape, &yDeviceAddr, aclDataType::ACL_FLOAT16, &y);
    CHECK_RET(ret == ACL_SUCCESS, return ret);

    uint64_t workspaceSize = 0;
    aclOpExecutor* executor;
    ret = aclnnDiagPartGetWorkspaceSize(x, y, &workspaceSize, &executor);
    CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclnnDiagPartGetWorkspaceSize failed. ERROR: %d\n", ret); return ret);
    void* workspaceAddr = nullptr;
    if (workspaceSize > 0) {
        ret = aclrtMalloc(&workspaceAddr, workspaceSize, ACL_MEM_MALLOC_HUGE_FIRST);
        CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("allocate workspace failed. ERROR: %d\n", ret); return ret);
    }
    ret = aclnnDiagPart(workspaceAddr, workspaceSize, executor, stream);
    CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclnnDiagPart failed. ERROR: %d\n", ret); return ret);

    ret = aclrtSynchronizeStream(stream);
    CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclrtSynchronizeStream failed. ERROR: %d\n", ret); return ret);

    auto size = GetShapeSize(yShape);
    std::vector<aclFloat16> resultData(size, 0);
    ret = aclrtMemcpy(
        resultData.data(), resultData.size() * sizeof(resultData[0]), yDeviceAddr, size * sizeof(resultData[0]),
        ACL_MEMCPY_DEVICE_TO_HOST);
    CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("copy result from device to host failed. ERROR: %d\n", ret); return ret);

    // 输出结果并验证
    LOG_PRINT("Result:   ");
    for (int64_t i = 0; i < size; i++) {
        LOG_PRINT("%.2f ", (float)resultData[i]);
    }
    LOG_PRINT("\nExpected: ");
    for (int64_t i = 0; i < expectedData.size(); i++) {
        LOG_PRINT("%.2f ", expectedData[i]);
    }
    LOG_PRINT("\n");

    // 验证结果
    bool pass = true;
    for (int64_t i = 0; i < size; i++) {
        if (std::abs((float)resultData[i] - expectedData[i]) > 0.1) {
            pass = false;
            break;
        }
    }

    if (pass) {
        LOG_PRINT("[PASS] FP16 test PASSED!\n");
    } else {
        LOG_PRINT("[FAIL] FP16 test FAILED!\n");
    }

    aclDestroyTensor(x);
    aclDestroyTensor(y);
    aclrtFree(xDeviceAddr);
    aclrtFree(yDeviceAddr);
    if (workspaceSize > 0) {
        aclrtFree(workspaceAddr);
    }
    aclrtDestroyStream(stream);
    aclrtResetDevice(deviceId);
    aclFinalize();
    return pass ? 0 : 1;
}

int testInt32()
{
    LOG_PRINT("\n========== Test for INT32 ==========\n");
    int32_t deviceId = 0;
    aclrtStream stream;
    auto ret = Init(deviceId, &stream);
    CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("Init acl failed. ERROR: %d\n", ret); return ret);

    int64_t sideLen = 16;
    std::vector<int64_t> xShape = {sideLen, sideLen};
    std::vector<int64_t> yShape = {sideLen};
    void* xDeviceAddr = nullptr;
    void* yDeviceAddr = nullptr;
    aclTensor* x = nullptr;
    aclTensor* y = nullptr;

    // 输入: 16x16 矩阵，对角线元素为 1, 18, 35, ..., 256
    std::vector<int32_t> xHostData;
    for (int i = 0; i < sideLen * sideLen; i++) {
        xHostData.push_back(i + 1);
    }
    // 期望输出: 对角线元素 [1, 18, 35, 52, 69, 86, 103, 120, 137, 154, 171, 188, 205, 222, 239, 256]
    std::vector<int32_t> expectedData;
    for (int i = 0; i < sideLen; i++) {
        expectedData.push_back(i * sideLen + i + 1); // diagonal elements
    }
    std::vector<int32_t> yHostData(sideLen, 0);

    ret = CreateAclTensor(xHostData, xShape, &xDeviceAddr, aclDataType::ACL_INT32, &x);
    CHECK_RET(ret == ACL_SUCCESS, return ret);
    ret = CreateAclTensor(yHostData, yShape, &yDeviceAddr, aclDataType::ACL_INT32, &y);
    CHECK_RET(ret == ACL_SUCCESS, return ret);

    uint64_t workspaceSize = 0;
    aclOpExecutor* executor;
    ret = aclnnDiagPartGetWorkspaceSize(x, y, &workspaceSize, &executor);
    CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclnnDiagPartGetWorkspaceSize failed. ERROR: %d\n", ret); return ret);
    void* workspaceAddr = nullptr;
    if (workspaceSize > 0) {
        ret = aclrtMalloc(&workspaceAddr, workspaceSize, ACL_MEM_MALLOC_HUGE_FIRST);
        CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("allocate workspace failed. ERROR: %d\n", ret); return ret);
    }
    ret = aclnnDiagPart(workspaceAddr, workspaceSize, executor, stream);
    CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclnnDiagPart failed. ERROR: %d\n", ret); return ret);

    ret = aclrtSynchronizeStream(stream);
    CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclrtSynchronizeStream failed. ERROR: %d\n", ret); return ret);

    auto size = GetShapeSize(yShape);
    std::vector<int32_t> resultData(size, 0);
    ret = aclrtMemcpy(
        resultData.data(), resultData.size() * sizeof(resultData[0]), yDeviceAddr, size * sizeof(resultData[0]),
        ACL_MEMCPY_DEVICE_TO_HOST);
    CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("copy result from device to host failed. ERROR: %d\n", ret); return ret);

    // 输出结果并验证
    LOG_PRINT("Result:   ");
    for (int64_t i = 0; i < size; i++) {
        LOG_PRINT("%d ", resultData[i]);
    }
    LOG_PRINT("\nExpected: ");
    for (int64_t i = 0; i < expectedData.size(); i++) {
        LOG_PRINT("%d ", expectedData[i]);
    }
    LOG_PRINT("\n");

    // 验证结果
    bool pass = true;
    for (int64_t i = 0; i < size; i++) {
        if (resultData[i] != expectedData[i]) {
            pass = false;
            break;
        }
    }

    if (pass) {
        LOG_PRINT("[PASS] INT32 test PASSED!\n");
    } else {
        LOG_PRINT("[FAIL] INT32 test FAILED!\n");
    }

    aclDestroyTensor(x);
    aclDestroyTensor(y);
    aclrtFree(xDeviceAddr);
    aclrtFree(yDeviceAddr);
    if (workspaceSize > 0) {
        aclrtFree(workspaceAddr);
    }
    aclrtDestroyStream(stream);
    aclrtResetDevice(deviceId);
    aclFinalize();
    return pass ? 0 : 1;
}

int main()
{
    LOG_PRINT("\n╔════════════════════════════════════════╗\n");
    LOG_PRINT("║  diag_part Operator Test Suite         ║\n");
    LOG_PRINT("╚════════════════════════════════════════╝\n");

    int totalFailed = 0;
    totalFailed += testFp32();
    totalFailed += testFp16();
    totalFailed += testInt32();

    LOG_PRINT("\n╔════════════════════════════════════════╗\n");
    LOG_PRINT("║           Test Summary                 ║\n");
    LOG_PRINT("╚════════════════════════════════════════╝\n");
    if (totalFailed == 0) {
        LOG_PRINT("[PASS] All tests PASSED! (3/3)\n");
    } else {
        LOG_PRINT("[FAIL] %d test(s) FAILED!\n", totalFailed);
    }

    return totalFailed;
}