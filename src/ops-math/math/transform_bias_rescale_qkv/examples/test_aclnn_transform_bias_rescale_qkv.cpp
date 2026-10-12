/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
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
#include "acl/acl.h"
#include "aclnnop/aclnn_transform_bias_rescale_qkv.h"

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

using StreamPtr = std::unique_ptr<std::remove_pointer<aclrtStream>::type, decltype(&aclrtDestroyStream)>;
using DeviceMemPtr = std::unique_ptr<void, decltype(&aclrtFree)>;
using TensorPtr = std::unique_ptr<aclTensor, decltype(&aclDestroyTensor)>;

void PrintOutResult(std::vector<int64_t>& shape, DeviceMemPtr& deviceAddr)
{
    auto size = GetShapeSize(shape);
    std::vector<float> resultData(size, 0);
    auto ret = aclrtMemcpy(resultData.data(), resultData.size() * sizeof(resultData[0]), deviceAddr.get(),
                           size * sizeof(resultData[0]), ACL_MEMCPY_DEVICE_TO_HOST);
    CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("copy result from device to host failed. ERROR: %d\n", ret); return);
    for (int64_t i = 0; i < size; i++) {
        LOG_PRINT("outPut result[%ld] is: %f\n", i, resultData[i]);
    }
}

void PrintInResult(std::vector<int64_t>& shape, DeviceMemPtr& deviceAddr)
{
    auto size = GetShapeSize(shape);
    std::vector<float> resultData(size, 0);
    auto ret = aclrtMemcpy(resultData.data(), resultData.size() * sizeof(resultData[0]), deviceAddr.get(),
                           size * sizeof(resultData[0]), ACL_MEMCPY_DEVICE_TO_HOST);
    CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("copy result from device to host failed. ERROR: %d\n", ret); return);
    for (int64_t i = 0; i < size; i++) {
        LOG_PRINT("input[%ld] is: %f\n", i, resultData[i]);
    }
}

int Init(int32_t deviceId, StreamPtr& stream, bool& initialized, bool& deviceSet)
{
    // (Fixed writing) Initialize.
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
    // Call aclrtMalloc to allocate memory on the device.
    auto ret = aclrtMalloc(&rawDeviceAddr, size, ACL_MEM_MALLOC_HUGE_FIRST);
    CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclrtMalloc failed. ERROR: %d\n", ret); return ret);
    deviceAddr.reset(rawDeviceAddr);
    // Call aclrtMemcpy to copy the data on the host to the memory on the device.
    ret = aclrtMemcpy(deviceAddr.get(), size, hostData.data(), size, ACL_MEMCPY_HOST_TO_DEVICE);
    CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclrtMemcpy failed. ERROR: %d\n", ret); return ret);

    // Compute the strides of the contiguous tensor.
    std::vector<int64_t> strides(shape.size(), 1);
    for (int64_t i = shape.size() - 2; i >= 0; i--) {
        strides[i] = shape[i + 1] * strides[i + 1];
    }

    // Call aclCreateTensor to create an aclTensor.
    aclTensor* rawTensor = aclCreateTensor(shape.data(), shape.size(), dataType, strides.data(), 0,
                                           aclFormat::ACL_FORMAT_ND, shape.data(), shape.size(), deviceAddr.get());
    CHECK_RET(rawTensor != nullptr, LOG_PRINT("aclCreateTensor failed.\n"); return ACL_ERROR_FAILURE);
    tensor.reset(rawTensor);
    return 0;
}

int main()
{
    // 1. （固定写法）device/stream初始化，参考acl API手册
    // 根据自己的实际device填写deviceId
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

    // 2. 构造输入与输出，需要根据API的接口自定义构造
    // qkv
    int64_t B = 3;
    int64_t T = 4;
    int64_t n = 3;
    int64_t d = 16;
    std::vector<int64_t> qkvShape = {B, T, 3 * n * d};
    int qkvCount = B * T * 3 * n * d;
    std::vector<float> qkvHostData(qkvCount, 1);

    for (int i = 0; i < qkvCount; ++i) {
        qkvHostData[i] = i * 1.0;
    }

    DeviceMemPtr qkvDeviceAddr(nullptr, &aclrtFree);
    TensorPtr qkv(nullptr, &aclDestroyTensor);
    // 创建input aclTensor
    ret = CreateAclTensor(qkvHostData, qkvShape, aclDataType::ACL_FLOAT, qkvDeviceAddr, qkv);
    CHECK_RET(ret == ACL_SUCCESS, return ret);

    // qkvBias
    std::vector<int64_t> qkvBiasShape = {3 * n * d};
    std::vector<float> qkvBiasHostData(3 * n * d, 0.5);

    DeviceMemPtr qkvBiasDeviceAddr(nullptr, &aclrtFree);
    TensorPtr qkvBias(nullptr, &aclDestroyTensor);
    // 创建input aclTensor
    ret = CreateAclTensor(qkvBiasHostData, qkvBiasShape, aclDataType::ACL_FLOAT, qkvBiasDeviceAddr, qkvBias);
    CHECK_RET(ret == ACL_SUCCESS, return ret);

    std::vector<int64_t> outShape = {B, n, T, d};
    std::vector<float> outHostData(qkvCount / 3, 1);
    TensorPtr outQ(nullptr, &aclDestroyTensor);
    TensorPtr outK(nullptr, &aclDestroyTensor);
    TensorPtr outV(nullptr, &aclDestroyTensor);
    DeviceMemPtr outQDeviceAddr(nullptr, &aclrtFree);
    DeviceMemPtr outKDeviceAddr(nullptr, &aclrtFree);
    DeviceMemPtr outVDeviceAddr(nullptr, &aclrtFree);

    // 创建out aclTensor
    ret = CreateAclTensor(outHostData, outShape, aclDataType::ACL_FLOAT, outQDeviceAddr, outQ);
    ret = CreateAclTensor(outHostData, outShape, aclDataType::ACL_FLOAT, outKDeviceAddr, outK);
    ret = CreateAclTensor(outHostData, outShape, aclDataType::ACL_FLOAT, outVDeviceAddr, outV);

    CHECK_RET(ret == ACL_SUCCESS, return ret);

    // 3. 调用CANN算子库API，需要修改为具体的Api名称
    uint64_t workspaceSize = 16 * 1024 * 1024;
    aclOpExecutor* executor;

    // LOG_PRINT("qkv input=====");
    // PrintInResult(qkvShape, &qkvDeviceAddr);

    // LOG_PRINT("qkvBias input=====");
    // PrintInResult(qkvBiasShape, &qkvBiasDeviceAddr);

    // 调用aclnnTransformBiasRescaleQkv第一段接口
    ret = aclnnTransformBiasRescaleQkvGetWorkspaceSize(qkv.get(), qkvBias.get(), n, outQ.get(), outK.get(), outV.get(),
                                                       &workspaceSize, &executor);
    CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclnnTransformBiasRescaleQkvGetWorkspaceSize failed. ERROR: %d\n", ret);
              return ret);

    // 根据第一段接口计算出的workspaceSize申请device内存
    DeviceMemPtr workspaceAddr(nullptr, &aclrtFree);
    if (workspaceSize > static_cast<uint64_t>(0)) {
        void* rawWorkspaceAddr = nullptr;
        ret = aclrtMalloc(&rawWorkspaceAddr, workspaceSize, ACL_MEM_MALLOC_HUGE_FIRST);
        CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("allocate workspace failed. ERROR: %d\n", ret); return ret);
        workspaceAddr.reset(rawWorkspaceAddr);
    }

    // 调用aclnnTransformBiasRescaleQkv第二段接口
    ret = aclnnTransformBiasRescaleQkv(workspaceAddr.get(), workspaceSize, executor, stream.get());
    CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclnnTransformBiasRescaleQkv failed. ERROR: %d\n", ret); return ret);

    // 4. （固定写法）同步等待任务执行结束
    ret = aclrtSynchronizeStream(stream.get());
    CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclrtSynchronizeStream failed. ERROR: %d\n", ret); return ret);

    // 5. 获取输出的值，将device侧内存上的结果复制至host侧，需要根据具体API的接口定义修改
    LOG_PRINT("q output=====");
    PrintOutResult(outShape, outQDeviceAddr);

    LOG_PRINT("k output=====");
    PrintOutResult(outShape, outKDeviceAddr);

    LOG_PRINT("v output=====");
    PrintOutResult(outShape, outVDeviceAddr);

    return 0;
}
