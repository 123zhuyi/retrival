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
#include "aclnnop/aclnn_ceil.h"

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

int Init(int32_t deviceId, StreamPtr& stream, bool& initialized, bool& deviceSet)
{
    // 固定写法，资源初始化
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
    // 调用aclrtMalloc申请device侧内存
    void* rawDeviceAddr = nullptr;
    auto ret = aclrtMalloc(&rawDeviceAddr, size, ACL_MEM_MALLOC_HUGE_FIRST);
    CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclrtMalloc failed. ERROR: %d\n", ret); return ret);
    deviceAddr.reset(rawDeviceAddr);
    // 调用aclrtMemcpy将host侧数据拷贝到device侧内存上
    ret = aclrtMemcpy(deviceAddr.get(), size, hostData.data(), size, ACL_MEMCPY_HOST_TO_DEVICE);
    CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclrtMemcpy failed. ERROR: %d\n", ret); return ret);

    // 计算连续tensor的strides
    std::vector<int64_t> strides(shape.size(), 1);
    for (int64_t i = shape.size() - 2; i >= 0; i--) {
        strides[i] = shape[i + 1] * strides[i + 1];
    }

    // 调用aclCreateTensor接口创建aclTensor
    aclTensor* rawTensor = aclCreateTensor(shape.data(), shape.size(), dataType, strides.data(), 0,
                                           aclFormat::ACL_FORMAT_ND, shape.data(), shape.size(), deviceAddr.get());
    CHECK_RET(rawTensor != nullptr, LOG_PRINT("aclCreateTensor failed.\n"); return ACL_ERROR_FAILURE);
    tensor.reset(rawTensor);
    return 0;
}

aclError InitAcl(int32_t deviceId, StreamPtr& stream, bool& initialized, bool& deviceSet)
{
    auto ret = Init(deviceId, stream, initialized, deviceSet);
    CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("Init acl failed. ERROR: %d\n", ret); return ret);
    return ACL_SUCCESS;
}

aclError CreateInputs(std::vector<int64_t>& inputShape, std::vector<int64_t>& outShape, DeviceMemPtr& selfDeviceAddr,
                      DeviceMemPtr& outDeviceAddr, TensorPtr& self, TensorPtr& out)
{
    std::vector<float> selfHostData = {0, 1.1, 2.2, 3.3, 4.4, 5.5, 6.6, 7.7};
    std::vector<float> outHostData(8, 0);

    // 创建 self aclTensor
    auto ret = CreateAclTensor(selfHostData, inputShape, aclDataType::ACL_FLOAT, selfDeviceAddr, self);
    CHECK_RET(ret == ACL_SUCCESS, return ret);

    // 创建 out aclTensor
    ret = CreateAclTensor(outHostData, outShape, aclDataType::ACL_FLOAT, outDeviceAddr, out);
    CHECK_RET(ret == ACL_SUCCESS, return ret);

    return ACL_SUCCESS;
}

aclError ExecOpApi(TensorPtr& self, TensorPtr& out, DeviceMemPtr& selfDeviceAddr, DeviceMemPtr& outDeviceAddr,
                   std::vector<int64_t>& outShape, StreamPtr& stream)
{
    aclOpExecutor* executor;
    uint64_t workspaceSize = 0;
    auto size = GetShapeSize(outShape);
    std::vector<float> resultData(size, 0);

    // aclnnCeil 接口调用示例
    LOG_PRINT("test aclnnCeil\n");

    // 调用 aclnnCeil 第一段接口
    auto ret = aclnnCeilGetWorkspaceSize(self.get(), out.get(), &workspaceSize, &executor);
    CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclnnCeilGetWorkspaceSize failed. ERROR: %d\n", ret); return ret);

    // 根据 workspaceSize 申请 device 内存
    DeviceMemPtr workspaceAddr(nullptr, &aclrtFree);
    if (workspaceSize > static_cast<uint64_t>(0)) {
        void* rawWorkspaceAddr = nullptr;
        ret = aclrtMalloc(&rawWorkspaceAddr, workspaceSize, ACL_MEM_MALLOC_HUGE_FIRST);
        CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("allocate workspace failed. ERROR: %d\n", ret); return ret);
        workspaceAddr.reset(rawWorkspaceAddr);
    }

    // 调用 aclnnCeil 第二段接口
    ret = aclnnCeil(workspaceAddr.get(), workspaceSize, executor, stream.get());
    CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclnnCeil failed. ERROR: %d\n", ret); return ret);

    // 同步
    ret = aclrtSynchronizeStream(stream.get());
    CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclrtSynchronizeStream failed. ERROR: %d\n", ret); return ret);

    // 拷贝 out 结果
    ret = aclrtMemcpy(resultData.data(), resultData.size() * sizeof(resultData[0]), outDeviceAddr.get(),
                      size * sizeof(resultData[0]), ACL_MEMCPY_DEVICE_TO_HOST);
    CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("copy result from device to host failed. ERROR: %d\n", ret); return ret);

    for (int64_t i = 0; i < size; i++) {
        LOG_PRINT("result[%ld] is: %lf\n", i, resultData[i]);
    }

    // aclnnInplaceCeil 调用示例
    LOG_PRINT("\ntest aclnnInplaceCeil\n");

    // 调用 aclnnInplaceCeil 第一段接口
    ret = aclnnInplaceCeilGetWorkspaceSize(self.get(), &workspaceSize, &executor);
    CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclnnInplaceCeilGetWorkspaceSize failed. ERROR: %d\n", ret); return ret);

    // 根据 workspaceSize 再次申请 device 内存（与原逻辑一致）
    if (workspaceSize > static_cast<uint64_t>(0)) {
        void* rawWorkspaceAddr = nullptr;
        ret = aclrtMalloc(&rawWorkspaceAddr, workspaceSize, ACL_MEM_MALLOC_HUGE_FIRST);
        CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("allocate workspace failed. ERROR: %d\n", ret); return ret);
        workspaceAddr.reset(rawWorkspaceAddr);
    }

    // 调用 aclnnInplaceCeil 第二段接口
    ret = aclnnInplaceCeil(workspaceAddr.get(), workspaceSize, executor, stream.get());
    CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclnnInplaceCeil failed. ERROR: %d\n", ret); return ret);

    // 同步
    ret = aclrtSynchronizeStream(stream.get());
    CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclrtSynchronizeStream failed. ERROR: %d\n", ret); return ret);

    // 拷贝 self 结果
    ret = aclrtMemcpy(resultData.data(), resultData.size() * sizeof(resultData[0]), selfDeviceAddr.get(),
                      size * sizeof(resultData[0]), ACL_MEMCPY_DEVICE_TO_HOST);
    CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("copy result from device to host failed. ERROR: %d\n", ret); return ret);

    for (int64_t i = 0; i < size; i++) {
        LOG_PRINT("result[%ld] is: %f\n", i, resultData[i]);
    }

    return ACL_SUCCESS;
}

int main()
{
    // 1. device/stream 初始化
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
    auto ret = InitAcl(deviceId, stream, initialized, deviceSet);
    CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("InitAcl failed. ERROR: %d\n", ret); return ret);

    // 2. 构造输入与输出
    std::vector<int64_t> inputShape = {4, 2};
    std::vector<int64_t> outShape = {4, 2};

    DeviceMemPtr selfDeviceAddr(nullptr, &aclrtFree);
    DeviceMemPtr outDeviceAddr(nullptr, &aclrtFree);
    TensorPtr self(nullptr, &aclDestroyTensor);
    TensorPtr out(nullptr, &aclDestroyTensor);

    ret = CreateInputs(inputShape, outShape, selfDeviceAddr, outDeviceAddr, self, out);
    CHECK_RET(ret == ACL_SUCCESS, return ret);

    // 3. 调用 CANN 算子 API（ceil + inplace ceil）
    ret = ExecOpApi(self, out, selfDeviceAddr, outDeviceAddr, outShape, stream);
    CHECK_RET(ret == ACL_SUCCESS, return ret);

    return 0;
}
