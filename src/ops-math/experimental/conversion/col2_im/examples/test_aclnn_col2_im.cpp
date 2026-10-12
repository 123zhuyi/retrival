/**
 * This file is part of the OpenBOAT project at Harbin Institute of Technology (HIT)
 * and is contributed to the CANN Open Software.
 *
 * Copyright (c) 2025 AISS Group, Harbin Institute of Technology (HIT).
 * All Rights Reserved.
 *
 * Authors (accounts):
 * - Qiu Zhuang <@qiu-zhuang>
 * - Su Tonghua <@sutonghua>
 *
 * This program is free software: you can redistribute it and/or modify it.
 * Licensed under the CANN Open Software License Agreement Version 2.0 (the "License").
 * You may not use this file except in compliance with the License.
 * See the LICENSE file at the root of the repository for the full text of the License.
 *
 * THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTIES OF ANY KIND, EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 */

#include <iostream>
#include <vector>
#include "acl/acl.h"
#include "aclnn_col2_im.h"

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

void PrintOutResult(std::vector<int64_t>& shape, void** deviceAddr)
{
    auto size = GetShapeSize(shape);
    std::vector<float> resultData(size, 0);
    auto ret = aclrtMemcpy(
        resultData.data(), resultData.size() * sizeof(resultData[0]), *deviceAddr, size * sizeof(resultData[0]),
        ACL_MEMCPY_DEVICE_TO_HOST);
    CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("copy result from device to host failed. ERROR: %d\n", ret); return);
    for (int64_t i = 0; i < size; i++) {
        LOG_PRINT("col2im result[%ld] is: %f\n", i, resultData[i]);
    }
}

int Init(int32_t deviceId, aclrtStream* stream)
{
    // 固定写法，初始化
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
    // 2. 申请device侧内存
    auto ret = aclrtMalloc(deviceAddr, size, ACL_MEM_MALLOC_HUGE_FIRST);
    CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclrtMalloc failed. ERROR: %d\n", ret); return ret);
    // 3. 调用aclrtMemcpy将host侧数据拷贝到device侧内存上
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

int main()
{
    // 1. 调用acl进行device/stream初始化
    int32_t deviceId = 0;
    aclrtStream stream;
    auto ret = Init(deviceId, &stream);
    CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("Init acl failed. ERROR: %d\n", ret); return ret);

    // 2. 构造输入与输出，需要根据API的接口自定义构造
    // Col2Im输入: [N, C*kernel_h*kernel_w, L]
    // 示例: batch=1, channels=3, kernel=2x2, L=9

    // 参数定义
    int64_t N = 1;
    int64_t C = 3;
    int64_t kernel_h = 2;
    int64_t kernel_w = 2;
    int64_t stride = 1;
    int64_t padding = 0;
    int64_t dilation = 1;

    // 输出图像尺寸 (目标)
    int64_t H = 4; // 输出高度
    int64_t W = 4; // 输出宽度

    // 根据输出尺寸计算输入col格式的维度
    int64_t out_H = (H + 2 * padding - dilation * (kernel_h - 1) - 1) / stride + 1;
    int64_t out_W = (W + 2 * padding - dilation * (kernel_w - 1) - 1) / stride + 1;
    int64_t L = out_H * out_W;                        // L = 3 * 3 = 9
    int64_t input_channels = C * kernel_h * kernel_w; // 3 * 2 * 2 = 12

    LOG_PRINT("=== Col2Im Test Parameters ===\n");
    LOG_PRINT("Target output image: [%ld, %ld, %ld, %ld] (N, C, H, W)\n", N, C, H, W);
    LOG_PRINT(
        "Kernel: %ldx%ld, Stride: %ld, Padding: %ld, Dilation: %ld\n", kernel_h, kernel_w, stride, padding, dilation);
    LOG_PRINT("Input col shape: [%ld, %ld, %ld] (N, C*kh*kw, L)\n", N, input_channels, L);
    LOG_PRINT("out_H=%ld, out_W=%ld, L=%ld\n\n", out_H, out_W, L);

    // 创建输入tensor (col格式)
    aclTensor* input = nullptr;
    void* inputDeviceAddr = nullptr;
    std::vector<int64_t> inputShape = {N, input_channels, L}; // [1, 12, 9]
    int64_t inputSize = GetShapeSize(inputShape);
    std::vector<float> inputHostData(inputSize);

    // 为输入数据填充测试值 (使用 Im2Col 的输出作为 Col2Im 的输入)
    for (int64_t i = 0; i < inputSize; i++) {
        inputHostData[i] = static_cast<float>(i % 10);
    }

    ret = CreateAclTensor(inputHostData, inputShape, &inputDeviceAddr, aclDataType::ACL_FLOAT, &input);
    CHECK_RET(ret == ACL_SUCCESS, return ret);

    // 打印输入数据（前20个）
    LOG_PRINT("Input col data (first 20 values):\n");
    for (int64_t i = 0; i < std::min((int64_t)20, inputSize); i++) {
        LOG_PRINT("  input[%ld] = %.1f\n", i, inputHostData[i]);
    }
    LOG_PRINT("\n");

    // 4. 构造输出tensor (图像格式)
    aclTensor* out = nullptr;
    void* outDeviceAddr = nullptr;
    std::vector<int64_t> outShape = {N, C, H, W}; // [1, 3, 4, 4]
    int64_t outSize = GetShapeSize(outShape);
    std::vector<float> outHostData(outSize, 0.0f);
    ret = CreateAclTensor(outHostData, outShape, &outDeviceAddr, aclDataType::ACL_FLOAT, &out);
    CHECK_RET(ret == ACL_SUCCESS, return ret);

    // 5. 调用CANN算子库API
    uint64_t workspaceSize = 0;
    aclOpExecutor* executor;

    // 6. 调用aclnnCol2Im第一段接口
    ret = aclnnCol2ImGetWorkspaceSize(
        input, H, W, kernel_h, kernel_w, stride, padding, dilation, out, &workspaceSize, &executor);
    CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclnnCol2ImGetWorkspaceSize failed. ERROR: %d\n", ret); return ret);

    LOG_PRINT("Workspace size: %lu bytes\n", workspaceSize);

    // 根据第一段接口计算出的workspaceSize申请device内存
    void* workspaceAddr = nullptr;
    if (workspaceSize > static_cast<uint64_t>(0)) {
        ret = aclrtMalloc(&workspaceAddr, workspaceSize, ACL_MEM_MALLOC_HUGE_FIRST);
        CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("allocate workspace failed. ERROR: %d\n", ret); return ret);
    }

    // 7. 调用aclnnCol2Im第二段接口
    ret = aclnnCol2Im(workspaceAddr, workspaceSize, executor, stream);
    CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclnnCol2Im failed. ERROR: %d\n", ret); return ret);

    // 8. （固定写法）同步等待任务执行结束
    ret = aclrtSynchronizeStream(stream);
    CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclrtSynchronizeStream failed. ERROR: %d\n", ret); return ret);

    // 9. 获取输出的值，将device侧内存上的结果拷贝至host侧
    LOG_PRINT("\n=== Col2Im Output Results ===\n");
    LOG_PRINT("Output shape: [%ld, %ld, %ld, %ld] (N, C, H, W)\n", N, C, H, W);
    PrintOutResult(outShape, &outDeviceAddr);

    // 10. 释放aclTensor
    aclDestroyTensor(input);
    aclDestroyTensor(out);

    // 11. 释放device资源
    aclrtFree(inputDeviceAddr);
    aclrtFree(outDeviceAddr);
    if (workspaceSize > static_cast<uint64_t>(0)) {
        aclrtFree(workspaceAddr);
    }
    aclrtDestroyStream(stream);
    aclrtResetDevice(deviceId);

    // 12. acl去初始化
    aclFinalize();

    LOG_PRINT("\n=== Col2Im execution completed successfully ===\n");

    return 0;
}