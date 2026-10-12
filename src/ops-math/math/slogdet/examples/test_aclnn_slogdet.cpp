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
#include "aclnnop/aclnn_slogdet.h"

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
    void* rawDeviceAddr = nullptr;
    // 调用aclrtMalloc申请device侧内存
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

aclError CreateInputs(std::vector<int64_t>& selfShape, std::vector<int64_t>& signOutShape,
                      std::vector<int64_t>& logOutShape, DeviceMemPtr& selfDeviceAddr, DeviceMemPtr& signOutDeviceAddr,
                      DeviceMemPtr& logOutDeviceAddr, TensorPtr& self, TensorPtr& signOut, TensorPtr& logOut)
{
    std::vector<float> selfHostData = {0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11};
    std::vector<float> signOutHostData = {0, 0, 0};
    std::vector<float> logOutHostData = {0, 0, 0};

    // 创建 self aclTensor
    auto ret = CreateAclTensor(selfHostData, selfShape, aclDataType::ACL_FLOAT, selfDeviceAddr, self);
    CHECK_RET(ret == ACL_SUCCESS, return ret);

    // 创建 signOut aclTensor
    ret = CreateAclTensor(signOutHostData, signOutShape, aclDataType::ACL_FLOAT, signOutDeviceAddr, signOut);
    CHECK_RET(ret == ACL_SUCCESS, return ret);

    // 创建 logOut aclTensor
    ret = CreateAclTensor(logOutHostData, logOutShape, aclDataType::ACL_FLOAT, logOutDeviceAddr, logOut);
    CHECK_RET(ret == ACL_SUCCESS, return ret);

    return ACL_SUCCESS;
}

aclError ExecOpApi(TensorPtr& self, TensorPtr& signOut, TensorPtr& logOut, DeviceMemPtr& workspaceAddrOut,
                   uint64_t& workspaceSize, DeviceMemPtr& signOutDeviceAddr, DeviceMemPtr& logOutDeviceAddr,
                   std::vector<int64_t>& signOutShape, std::vector<int64_t>& logOutShape, StreamPtr& stream)
{
    aclOpExecutor* executor;

    // 调用 aclnnSlogdet 第一段接口
    auto ret = aclnnSlogdetGetWorkspaceSize(self.get(), signOut.get(), logOut.get(), &workspaceSize, &executor);
    CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclnnSlogdetGetWorkspaceSize failed. ERROR: %d\n", ret); return ret);

    // 根据 workspaceSize 申请 device 内存
    if (workspaceSize > static_cast<uint64_t>(0)) {
        void* rawWorkspaceAddr = nullptr;
        ret = aclrtMalloc(&rawWorkspaceAddr, workspaceSize, ACL_MEM_MALLOC_HUGE_FIRST);
        CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("allocate workspace failed. ERROR: %d\n", ret); return ret);
        workspaceAddrOut.reset(rawWorkspaceAddr);
    }

    // 调用 aclnnSlogdet 第二段接口
    ret = aclnnSlogdet(workspaceAddrOut.get(), workspaceSize, executor, stream.get());
    CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclnnSlogdet failed. ERROR: %d\n", ret); return ret);

    // 同步
    ret = aclrtSynchronizeStream(stream.get());
    CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclrtSynchronizeStream failed. ERROR: %d\n", ret); return ret);

    // 拷贝 signOut
    auto sizeSign = GetShapeSize(signOutShape);
    std::vector<float> resultData(sizeSign, 0);
    ret = aclrtMemcpy(resultData.data(), resultData.size() * sizeof(resultData[0]), signOutDeviceAddr.get(),
                      sizeSign * sizeof(resultData[0]), ACL_MEMCPY_DEVICE_TO_HOST);
    CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("copy result from device to host failed. ERROR: %d\n", ret); return ret);
    for (int64_t i = 0; i < sizeSign; i++) {
        LOG_PRINT("signout result[%ld] is: %f\n", i, resultData[i]);
    }

    // 拷贝 logOut
    auto sizeLog = GetShapeSize(logOutShape);
    std::vector<float> logResultData(sizeLog, 0);
    ret = aclrtMemcpy(logResultData.data(), logResultData.size() * sizeof(logResultData[0]), logOutDeviceAddr.get(),
                      sizeLog * sizeof(logResultData[0]), ACL_MEMCPY_DEVICE_TO_HOST);
    CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("copy result from device to host failed. ERROR: %d\n", ret); return ret);
    for (int64_t i = 0; i < sizeLog; i++) {
        LOG_PRINT("logout result[%ld] is: %f\n", i, logResultData[i]);
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
    CHECK_RET(ret == ACL_SUCCESS, return ret);

    // 2. 构造输入与输出
    std::vector<int64_t> selfShape = {3, 2, 2};
    std::vector<int64_t> signOutShape = {3};
    std::vector<int64_t> logOutShape = {3};

    DeviceMemPtr selfDeviceAddr(nullptr, &aclrtFree);
    DeviceMemPtr signOutDeviceAddr(nullptr, &aclrtFree);
    DeviceMemPtr logOutDeviceAddr(nullptr, &aclrtFree);
    TensorPtr self(nullptr, &aclDestroyTensor);
    TensorPtr signOut(nullptr, &aclDestroyTensor);
    TensorPtr logOut(nullptr, &aclDestroyTensor);

    ret = CreateInputs(selfShape, signOutShape, logOutShape, selfDeviceAddr, signOutDeviceAddr, logOutDeviceAddr, self,
                       signOut, logOut);
    CHECK_RET(ret == ACL_SUCCESS, return ret);

    // 3. 调用 CANN 算子 API
    uint64_t workspaceSize = 0;
    DeviceMemPtr workspaceAddr(nullptr, &aclrtFree);

    ret = ExecOpApi(self, signOut, logOut, workspaceAddr, workspaceSize, signOutDeviceAddr, logOutDeviceAddr,
                    signOutShape, logOutShape, stream);
    CHECK_RET(ret == ACL_SUCCESS, return ret);

    return 0;
}
