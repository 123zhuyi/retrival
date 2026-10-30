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
#include "aclnn_im2_col.h"

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

// 常量定义
constexpr int64_t MAX_TEST_VALUE = 10; // 用于测试数据生成的模数

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
        LOG_PRINT("im2col result[%ld] is: %f\n", i, resultData[i]);
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
    // Im2Col输入: [N, C, H, W]
    // 示例: batch=1, channels=3, height=4, width=4
    aclTensor* input = nullptr;
    void* inputDeviceAddr = nullptr;
    std::vector<int64_t> inputShape = {1, 3, 4, 4}; // [N, C, H, W]
    int64_t inputSize = GetShapeSize(inputShape);

    // 为输入数据填充测试值，使用命名常量替代魔法数字
    std::vector<float> inputHostData(inputSize, 1.0f);
    for (int64_t i = 0; i < inputSize; i++) {
        inputHostData[i] = static_cast<float>(i % MAX_TEST_VALUE);
    }

    ret = CreateAclTensor(inputHostData, inputShape, &inputDeviceAddr, aclDataType::ACL_FLOAT, &input);
    CHECK_RET(ret == ACL_SUCCESS, return ret);

    // 3. 设置Im2Col参数
    int32_t kernel_h = 2;
    int32_t kernel_w = 2;
    int32_t stride_h = 1;
    int32_t stride_w = 1;
    int32_t pad_h = 0;
    int32_t pad_w = 0;
    int32_t dilation_h = 1;
    int32_t dilation_w = 1;

    // 计算输出shape
    int64_t N = inputShape[0];
    int64_t C = inputShape[1];
    int64_t H = inputShape[2];
    int64_t W = inputShape[3];

    int64_t out_H = (H + 2 * pad_h - dilation_h * (kernel_h - 1) - 1) / stride_h + 1;
    int64_t out_W = (W + 2 * pad_w - dilation_w * (kernel_w - 1) - 1) / stride_w + 1;
    int64_t output_channels = C * kernel_h * kernel_w;
    int64_t L = out_H * out_W;

    LOG_PRINT("Input shape: [%ld, %ld, %ld, %ld]\n", N, C, H, W);
    LOG_PRINT("Kernel: %ldx%ld, Stride: %ld, Padding: %ld\n", kernel_h, kernel_w, stride, padding);
    LOG_PRINT("Output shape: [%ld, %ld, %ld] (N, C*kh*kw, L)\n", N, output_channels, L);

    // 4. 构造输出tensor
    aclTensor* out = nullptr;
    void* outDeviceAddr = nullptr;
    std::vector<int64_t> outShape = {N, output_channels, L};
    int64_t outSize = GetShapeSize(outShape);

    // 使用计算出的输出大小初始化输出数据
    std::vector<float> outHostData(outSize, 1.0f);
    ret = CreateAclTensor(outHostData, outShape, &outDeviceAddr, aclDataType::ACL_FLOAT, &out);
    CHECK_RET(ret == ACL_SUCCESS, return ret);

    // 5. 调用CANN算子库API
    uint64_t workspaceSize = 0;
    aclOpExecutor* executor;

    LOG_PRINT("Before GetWorkspaceSize: input=%p, out=%p\n", (void*)input, (void*)out);
    LOG_PRINT("Before GetWorkspaceSize: inputDeviceAddr=%p, outDeviceAddr=%p\n", inputDeviceAddr, outDeviceAddr);

    // 6. 调用aclnnIm2Col第一段接口
    ret = aclnnIm2ColGetWorkspaceSize(
        inputX, kernel_h, kernel_w, stride_h, stride_w, pad_h, pad_w, dilation_h, dilation_w, out, &workspaceSize,
        &executor);
    CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclnnIm2ColGetWorkspaceSize failed. ERROR: %d\n", ret); return ret);

    // 根据第一段接口计算出的workspaceSize申请device内存
    void* workspaceAddr = nullptr;
    if (workspaceSize > static_cast<uint64_t>(0)) {
        ret = aclrtMalloc(&workspaceAddr, workspaceSize, ACL_MEM_MALLOC_HUGE_FIRST);
        CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("allocate workspace failed. ERROR: %d\n", ret); return ret);
    }

    // 7. 调用aclnnIm2Col第二段接口
    ret = aclnnIm2Col(workspaceAddr, workspaceSize, executor, stream);
    CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclnnIm2Col failed. ERROR: %d\n", ret); return ret);

    // 8. （固定写法）同步等待任务执行结束
    ret = aclrtSynchronizeStream(stream);
    CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclrtSynchronizeStream failed. ERROR: %d\n", ret); return ret);

    // 9. 获取输出的值，将device侧内存上的结果拷贝至host侧
    LOG_PRINT("\n=== Im2Col Output Results ===\n");
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

    LOG_PRINT("\n=== Im2Col execution completed successfully ===\n");

    return 0;
}
