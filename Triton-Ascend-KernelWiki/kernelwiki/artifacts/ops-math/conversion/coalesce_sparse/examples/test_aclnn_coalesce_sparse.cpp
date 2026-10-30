/**
 * Copyright (c) Huawei Technologies Co., Ltd.2025. All rights reserved.
 * This file is a part of the CANN Open Software.
 * Licensed under CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include <iostream>
#include <vector>
#include "acl/acl.h"
#include "aclnnop/aclnn_coalesce_sparse.h"

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

int main()
{
    // 1. （固定写法）device/stream初始化，参考acl API文档
    // 根据自己的实际device填写deviceId
    int32_t deviceId = 0;
    aclrtStream stream;
    auto ret = Init(deviceId, &stream);
    CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("Init acl failed. ERROR: %d\n", ret); return ret);

    // 2. 构造输入与输出，需要根据API的接口自定义构造
    std::vector<int64_t> uniqueLenShape = {1};
    std::vector<int64_t> uniqueIndicesShape = {8};
    std::vector<int64_t> indexShape = {8,4};
    std::vector<int64_t> valueShape = {8};
    std::vector<int64_t> newIndexShape = {8,4};
    std::vector<int64_t> newValueShape = {8};
    void* uniqueLenDeviceAddr = nullptr;
    void* uniqueIndicesDeviceAddr = nullptr;
    void* indexDeviceAddr = nullptr;
    void* valueDeviceAddr = nullptr;
    void* newIndexDeviceAddr = nullptr;
    void* newValueDeviceAddr = nullptr;
    aclTensor* uniqueLen = nullptr;
    aclTensor* uniqueIndices = nullptr;
    aclTensor* index = nullptr;
    aclTensor* value = nullptr;
    aclTensor* newIndex = nullptr;
    aclTensor* newValue = nullptr;
    std::vector<int32_t> uniqueLenData = {8};
    std::vector<int32_t> uniqueIndicesData = {5, 1, 6, 4, 7, 0, 3, 2};
    std::vector<int32_t> indexData = {
        17, 17,  1,  0,	 
         3, 17,  1,  8,
        19, 10, 13,  6,
        13, 16,  0, 15,
        19, 14,  5, 15,
         2,  8, 18,  5,
         5,  2,  0, 13,
         4,  5,  9,  1
    };
    std::vector<float> valueData = {148.2500, -706.5000,  178.6250,  399.5000, -795.5000, -171.8750, 
        388.7500, -171.6250};
    std::vector<int32_t> newIndexData = {
        0, 0, 0, 0,
        0, 0, 0, 0, 
        0, 0, 0, 0, 
        0, 0, 0, 0, 
        0, 0, 0, 0, 
        0, 0, 0, 0, 
        0, 0, 0, 0, 
        0, 0, 0, 0
    };
    std::vector<float> newValueData = {0, 0, 0, 0, 0, 0, 0, 0};

    // 创建in aclTensor
    ret = CreateAclTensor(uniqueLenData, uniqueLenShape, &uniqueLenDeviceAddr, aclDataType::ACL_INT32, &uniqueLen);
    CHECK_RET(ret == ACL_SUCCESS, return ret);
    // 创建in aclTensor
    ret = CreateAclTensor(uniqueIndicesData, uniqueIndicesShape, &uniqueIndicesDeviceAddr, aclDataType::ACL_INT32, &uniqueIndices);
    CHECK_RET(ret == ACL_SUCCESS, return ret);
    // 创建in aclTensor
    ret = CreateAclTensor(indexData, indexShape, &indexDeviceAddr, aclDataType::ACL_INT32, &index);
    CHECK_RET(ret == ACL_SUCCESS, return ret);
    // 创建in aclTensor
    ret = CreateAclTensor(valueData, valueShape, &valueDeviceAddr, aclDataType::ACL_FLOAT, &value);
    CHECK_RET(ret == ACL_SUCCESS, return ret);
    // 创建out aclTensor
    ret = CreateAclTensor(newIndexData, newIndexShape, &newIndexDeviceAddr, aclDataType::ACL_INT32, &newIndex);
    CHECK_RET(ret == ACL_SUCCESS, return ret);
    // 创建out aclTensor
    ret = CreateAclTensor(newValueData, newValueShape, &newValueDeviceAddr, aclDataType::ACL_FLOAT, &newValue);
    CHECK_RET(ret == ACL_SUCCESS, return ret);

    // 3. 调用CANN算子库API，需要修改为具体的Api名称
    uint64_t workspaceSize = 0;
    aclOpExecutor* executor;
    // 调用aclnnCoalesceSparse第一段接口
    ret = aclnnCoalesceSparseGetWorkspaceSize(uniqueLen, uniqueIndices, index, value, newIndex, newValue, &workspaceSize, &executor);
    CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclnnCoalesceSparseGetWorkspaceSize failed. ERROR: %d\n", ret); return ret);
    // 根据第一段接口计算出的workspaceSize申请device内存
    void* workspaceAddr = nullptr;
    if (workspaceSize > static_cast<uint64_t>(0)) {
        ret = aclrtMalloc(&workspaceAddr, workspaceSize, ACL_MEM_MALLOC_HUGE_FIRST);
        CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("allocate workspace failed. ERROR: %d\n", ret); return ret);
    }
    // 调用aclnnCoalesceSparse第二段接口
    ret = aclnnCoalesceSparse(workspaceAddr, workspaceSize, executor, stream);
    CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclnnCoalesceSparse failed. ERROR: %d\n", ret); return ret);

    // 4. （固定写法）同步等待任务执行结束
    ret = aclrtSynchronizeStream(stream);
    CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclrtSynchronizeStream failed. ERROR: %d\n", ret); return ret);

    // 5. 获取输出的值，将device侧内存上的结果拷贝至host侧，需要根据具体API的接口定义修改
    auto indexSize = GetShapeSize(newIndexShape);
    std::vector<int32_t> resultIndexData(indexSize, 0);
    ret = aclrtMemcpy(
        resultIndexData.data(), resultIndexData.size() * sizeof(int32_t), newIndexDeviceAddr, indexSize * sizeof(int32_t),
        ACL_MEMCPY_DEVICE_TO_HOST);
    CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("copy result from device to host failed. ERROR: %d\n", ret); return ret);
    for (int64_t i = 0; i < indexSize; i++) {
        LOG_PRINT("result[%ld] is: %d\n", i, resultIndexData[i]);
    }

    auto valueSize = GetShapeSize(newValueShape);
    std::vector<float> resultValueData(valueSize, 0);
    ret = aclrtMemcpy(
        resultValueData.data(), resultValueData.size() * sizeof(resultValueData[0]), newValueDeviceAddr, valueSize * sizeof(resultValueData[0]),
        ACL_MEMCPY_DEVICE_TO_HOST);
    CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("copy result from device to host failed. ERROR: %d\n", ret); return ret);
    for (int64_t i = 0; i < valueSize; i++) {
        LOG_PRINT("result[%ld] is: %f\n", i, resultValueData[i]);
    }

    // 6. 释放aclTensor，需要根据具体API的接口定义修改
    aclDestroyTensor(uniqueLen);
    aclDestroyTensor(uniqueIndices);
    aclDestroyTensor(index);
    aclDestroyTensor(value);
    aclDestroyTensor(newIndex);
    aclDestroyTensor(newValue);

    // 7. 释放device资源
    aclrtFree(uniqueLenDeviceAddr);
    aclrtFree(uniqueIndicesDeviceAddr);
    aclrtFree(indexDeviceAddr);
    aclrtFree(valueDeviceAddr);
    aclrtFree(newIndexDeviceAddr);
    aclrtFree(newValueDeviceAddr);
    if (workspaceSize > static_cast<uint64_t>(0)) {
        aclrtFree(workspaceAddr);
    }
    aclrtDestroyStream(stream);
    aclrtResetDevice(deviceId);
    aclFinalize();
    return 0;
}