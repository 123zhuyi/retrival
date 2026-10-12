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
#include "aclnnop/aclnn_amp_update_scale.h"

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
int CreateAclTensor(const std::vector<T>& hostData, const std::vector<int64_t>& shape, void** deviceAddr,
                    aclDataType dataType, aclTensor** tensor)
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
    *tensor = aclCreateTensor(shape.data(), shape.size(), dataType, strides.data(), 0, aclFormat::ACL_FORMAT_ND,
                              shape.data(), shape.size(), *deviceAddr);
    return 0;
}

int main()
{
    // 1.（固定写法）device/stream初始化，参考acl API手册
    // 根据自己的实际device填写deviceId
    int32_t deviceId = 0;
    aclrtStream stream;
    auto ret = Init(deviceId, &stream);
    CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("Init acl failed. ERROR: %d\n", ret); return ret);

    // 2. 构造输入与输出，需要根据API的接口自定义构造
    std::vector<int64_t> scalarShape = {1};
    void* currentScaleDeviceAddr = nullptr;
    void* growthTrackerDeviceAddr = nullptr;
    void* foundInfDeviceAddr = nullptr;
    void* updatedScaleDeviceAddr = nullptr;
    void* updatedGrowthTrackerDeviceAddr = nullptr;
    aclTensor* currentScale = nullptr;
    aclTensor* growthTracker = nullptr;
    aclTensor* foundInf = nullptr;
    aclTensor* updatedScale = nullptr;
    aclTensor* updatedGrowthTracker = nullptr;

    // 创建currentScale
    std::vector<float> currentScaleHost = {65536.0f};
    ret = CreateAclTensor(currentScaleHost, scalarShape, &currentScaleDeviceAddr, ACL_FLOAT, &currentScale);
    CHECK_RET(ret == ACL_SUCCESS, return ret);

    // 创建growthTracker
    std::vector<int32_t> growthTrackerHost = {900};
    ret = CreateAclTensor(growthTrackerHost, scalarShape, &growthTrackerDeviceAddr, ACL_INT32, &growthTracker);
    CHECK_RET(ret == ACL_SUCCESS, return ret);

    // 创建foundInf
    std::vector<float> foundInfHost = {0.0f};
    ret = CreateAclTensor(foundInfHost, scalarShape, &foundInfDeviceAddr, ACL_FLOAT, &foundInf);
    CHECK_RET(ret == ACL_SUCCESS, return ret);

    // 创建输出updatedScale
    std::vector<float> updatedScaleHost = {0.0f};
    ret = CreateAclTensor(updatedScaleHost, scalarShape, &updatedScaleDeviceAddr, ACL_FLOAT, &updatedScale);
    CHECK_RET(ret == ACL_SUCCESS, return ret);

    // 创建输出updatedGrowthTracker
    std::vector<int32_t> updatedGrowthTrackerHost = {0};
    ret = CreateAclTensor(updatedGrowthTrackerHost, scalarShape, &updatedGrowthTrackerDeviceAddr, ACL_INT32,
                          &updatedGrowthTracker);
    CHECK_RET(ret == ACL_SUCCESS, return ret);

    // 3. 调用第一段接口，获取workspace大小和执行器
    float growthFactor = 2.0f;
    float backoffFactor = 0.5f;
    int64_t growthInterval = 1000;

    uint64_t workspaceSize = 0;
    aclOpExecutor* executor = nullptr;
    ret = aclnnAmpUpdateScaleGetWorkspaceSize(currentScale, growthTracker, foundInf, growthFactor, backoffFactor,
                                              growthInterval, updatedScale, updatedGrowthTracker, &workspaceSize,
                                              &executor);
    CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclnnAmpUpdateScaleGetWorkspaceSize failed. ERROR: %d\n", ret);
              return ret);

    // 4. 根据workspaceSize申请workspace内存
    void* workspace = nullptr;
    if (workspaceSize > 0) {
        ret = aclrtMalloc(&workspace, workspaceSize, ACL_MEM_MALLOC_HUGE_FIRST);
        CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("allocate workspace failed. ERROR: %d\n", ret); return ret);
    }

    // 5. 调用第二段接口，执行计算
    ret = aclnnAmpUpdateScale(workspace, workspaceSize, executor, stream);
    CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclnnAmpUpdateScale failed. ERROR: %d\n", ret); return ret);

    // 6. 同步等待任务执行结束
    ret = aclrtSynchronizeStream(stream);
    CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclrtSynchronizeStream failed. ERROR: %d\n", ret); return ret);

    // 7. 将输出数据从device拷贝到host，并打印结果
    float updatedScaleVal = 0.0f;
    int32_t updatedGrowthTrackerVal = 0;
    ret = aclrtMemcpy(&updatedScaleVal, sizeof(float), updatedScaleDeviceAddr, sizeof(float),
                      ACL_MEMCPY_DEVICE_TO_HOST);
    CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("copy updatedScale failed. ERROR: %d\n", ret); return ret);
    ret = aclrtMemcpy(&updatedGrowthTrackerVal, sizeof(int32_t), updatedGrowthTrackerDeviceAddr, sizeof(int32_t),
                      ACL_MEMCPY_DEVICE_TO_HOST);
    CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("copy updatedGrowthTracker failed. ERROR: %d\n", ret); return ret);
    LOG_PRINT("aclnnAmpUpdateScale result: updatedScale = %f, updatedGrowthTracker = %d\n", updatedScaleVal,
              updatedGrowthTrackerVal);

    // 8.（固定写法）释放资源
    aclDestroyTensor(currentScale);
    aclDestroyTensor(growthTracker);
    aclDestroyTensor(foundInf);
    aclDestroyTensor(updatedScale);
    aclDestroyTensor(updatedGrowthTracker);
    aclrtFree(currentScaleDeviceAddr);
    aclrtFree(growthTrackerDeviceAddr);
    aclrtFree(foundInfDeviceAddr);
    aclrtFree(updatedScaleDeviceAddr);
    aclrtFree(updatedGrowthTrackerDeviceAddr);
    if (workspaceSize > 0) {
        aclrtFree(workspace);
    }
    aclrtDestroyStream(stream);
    auto aclRet = aclrtResetDevice(deviceId);
    CHECK_RET(aclRet == ACL_SUCCESS, LOG_PRINT("reset device failed. ERROR: %d\n", aclRet); return aclRet);
    aclRet = aclFinalize();
    CHECK_RET(aclRet == ACL_SUCCESS, LOG_PRINT("finalize acl failed. ERROR: %d\n", aclRet); return aclRet);
    return 0;
}
