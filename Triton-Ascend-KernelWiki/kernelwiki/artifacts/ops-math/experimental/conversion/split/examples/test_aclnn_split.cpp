/**
 * This program is free software, you can redistribute it and/or modify it.
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This file is a part of the CANN Open Software.
 * Licensed under CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED, INCLUDING
 * BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include <algorithm>
#include <cstdint>
#include <iostream>
#include <vector>
#include <sys/types.h>
#include <sys/stat.h>
#include <unistd.h>
#include <fstream>
#include <fcntl.h>
#include <string>
#include <cstdio>
#include <cstdlib>

#include "acl/acl.h"
#include "aclnn_split.h"

#define SUCCESS 0
#define FAILED 1

// 保留用户要求的日志宏
#define INFO_LOG(fmt, args...) fprintf(stdout, "[INFO]  " fmt "\n", ##args)
#define WARN_LOG(fmt, args...) fprintf(stdout, "[WARN]  " fmt "\n", ##args)
#define ERROR_LOG(fmt, args...) fprintf(stderr, "[ERROR]  " fmt "\n", ##args)
#define DEBUG_LOG(fmt, args...) fprintf(stdout, "[DEBUG] " fmt "\n", ##args)

// 可变参数CHECK_RET宏
#define CHECK_RET(cond, fmt, ...)          \
    do {                                   \
        if (!(cond)) {                     \
            ERROR_LOG(fmt, ##__VA_ARGS__); \
            return FAILED;                 \
        }                                  \
    } while (0)

#define LOG_PRINT(message, ...)         \
    do {                                \
        printf(message, ##__VA_ARGS__); \
    } while (0)

constexpr int32_t DIM_LIMIT = 10;
// 计算shape总元素数
int64_t GetShapeSize(const std::vector<int64_t>& shape)
{
    int64_t shapeSize = 1;
    for (size_t i = 0; i < shape.size(); ++i) {
        CHECK_RET(shape[i] > 0, "shape dimension %zu must be positive", i);
        shapeSize *= shape[i];
    }
    return shapeSize;
}

// ACL初始化
int Init(int32_t deviceId, aclrtStream* stream)
{
    aclError ret = aclInit(nullptr);
    CHECK_RET(ret == ACL_SUCCESS, "aclInit failed, error=%d", ret);

    ret = aclrtSetDevice(deviceId);
    CHECK_RET(ret == ACL_SUCCESS, "aclrtSetDevice failed, deviceId=%d, error=%d", deviceId, ret);

    ret = aclrtCreateStream(stream);
    CHECK_RET(ret == ACL_SUCCESS, "aclrtCreateStream failed, error=%d", ret);

    INFO_LOG("ACL init success, deviceId=%d", deviceId);
    return SUCCESS;
}

// 调试工具：验证Host→Device拷贝是否成功
template <typename T>
bool VerifyH2DCopy(void* deviceAddr, const std::vector<T>& hostData, size_t memSize)
{
    std::vector<T> verifyData(hostData.size());
    aclError ret = aclrtMemcpy(verifyData.data(), memSize, deviceAddr, memSize, ACL_MEMCPY_DEVICE_TO_HOST);
    if (ret != ACL_SUCCESS) {
        ERROR_LOG("Verify H2D copy failed, aclrtMemcpy error=%d", ret);
        return false;
    }

    // 对比前10个元素
    bool match = true;
    for (size_t i = 0; i < std::min((size_t)10, hostData.size()); ++i) {
        if (std::abs(verifyData[i] - hostData[i]) > 1e-6) {
            ERROR_LOG("H2D copy mismatch: host[%zu] = %.2f, device[%zu] = %.2f", i, hostData[i], i, verifyData[i]);
            match = false;
            break;
        }
    }
    if (match) {
        DEBUG_LOG("H2D copy verification success (first 10 elements match)");
    }
    return match;
}

// 重载1：输入Tensor（从Host数据初始化，带H2D验证）
template <typename T>
int CreateAclTensor(
    const std::vector<T>& hostData, const std::vector<int64_t>& shape, void** deviceAddr, aclDataType dataType,
    aclTensor** tensor)
{
    // 校验输入合法性
    int64_t expectedSize = GetShapeSize(shape);
    CHECK_RET(
        static_cast<int64_t>(hostData.size()) == expectedSize, "host data size mismatch: expected=%lld, actual=%zu",
        (long long)expectedSize, hostData.size());

    size_t memSize = expectedSize * sizeof(T);
    DEBUG_LOG(
        "Create input Tensor: shape=[%s], memSize=%zu bytes",
        [&]() {
            std::string s;
            for (auto d : shape)
                s += std::to_string(d) + ",";
            return s.substr(0, s.size() - 1);
        }()
            .c_str(),
        memSize);

    // 申请Device内存
    aclError ret = aclrtMalloc(deviceAddr, memSize, ACL_MEM_MALLOC_HUGE_FIRST);
    CHECK_RET(ret == ACL_SUCCESS, "aclrtMalloc failed, size=%zu, error=%d", memSize, ret);
    DEBUG_LOG("Device memory allocated: addr=%p, size=%zu", *deviceAddr, memSize);

    // Host -> Device 数据拷贝
    ret = aclrtMemcpy(*deviceAddr, memSize, hostData.data(), memSize, ACL_MEMCPY_HOST_TO_DEVICE);
    CHECK_RET(ret == ACL_SUCCESS, "aclrtMemcpy(H2D) failed, error=%d", ret);

    // 验证拷贝结果（关键！）
    VerifyH2DCopy(*deviceAddr, hostData, memSize);

    // 计算连续Tensor的strides
    std::vector<int64_t> strides(shape.size(), 1);
    for (int64_t i = shape.size() - 2; i >= 0; --i) {
        strides[i] = shape[i + 1] * strides[i + 1];
    }

    // 创建ACL Tensor
    *tensor = aclCreateTensor(
        shape.data(), shape.size(), dataType, strides.data(), 0, ACL_FORMAT_ND, shape.data(), shape.size(),
        *deviceAddr);
    CHECK_RET(*tensor != nullptr, "aclCreateTensor failed");
    DEBUG_LOG("Input Tensor created: tensor=%p, deviceAddr=%p", *tensor, *deviceAddr);

    return SUCCESS;
}

// 重载2：输出Tensor（全零初始化）
template <typename T>
int CreateAclTensor(const std::vector<int64_t>& shape, void** deviceAddr, aclDataType dataType, aclTensor** tensor)
{
    int64_t tensorSize = GetShapeSize(shape);
    std::vector<T> hostData(tensorSize, 0); // 全零初始化
    return CreateAclTensor(hostData, shape, deviceAddr, dataType, tensor);
}

// 创建TensorList（带输出地址日志），分配地址不连续
template <typename T>
int CreateAclTensorList(
    const std::vector<std::vector<int64_t>>& shapes, std::vector<void*>& deviceAddrs, aclDataType dataType,
    aclTensorList** tensorList)
{
    size_t listSize = shapes.size();
    CHECK_RET(listSize > 0, "tensor list size must be positive");

    deviceAddrs.resize(listSize, nullptr);
    std::vector<aclTensor*> tensors(listSize, nullptr);

    for (size_t i = 0; i < listSize; ++i) {
        DEBUG_LOG(
            "Create TensorList[%zu]: shape=[%s]", i,
            [&]() {
                std::string s;
                for (auto d : shapes[i])
                    s += std::to_string(d) + ",";
                return s.substr(0, s.size() - 1);
            }()
                .c_str());
        int ret = CreateAclTensor<T>(shapes[i], &deviceAddrs[i], dataType, &tensors[i]);
        CHECK_RET(ret == SUCCESS, "create tensor %zu in list failed", i);
        DEBUG_LOG("TensorList[%zu] created: tensor=%p, deviceAddr=%p", i, tensors[i], deviceAddrs[i]);
    }

    *tensorList = aclCreateTensorList(tensors.data(), listSize);
    CHECK_RET(*tensorList != nullptr, "aclCreateTensorList failed");
    DEBUG_LOG("ACL TensorList created: list=%p, size=%zu", *tensorList, listSize);

    return SUCCESS;
}

int main(int argc, char** argv)
{
    // 1. ACL初始化
    int32_t deviceId = 0;
    aclrtStream stream = nullptr;
    int ret = Init(deviceId, &stream);
    CHECK_RET(ret == SUCCESS, "ACL init failed");

    // 2. 算子参数配置（明确拆分逻辑）
    const std::vector<int64_t> inputXShape = {4, 5};                                  // 输入：[4,5] → 20个元素
    const std::vector<std::vector<int64_t>> outputZShapes = {{4, 1}, {4, 1}, {4, 3}}; // 输出
    const int64_t splitAxis = 1;                                                      // 沿第1维拆分（5→1+1+3）
    // const std::vector<std::vector<int64_t>> outputZShapes = {{1,5}, {1,5},{2,5}}; // 输出
    // const int64_t splitAxis = 0;                        // 沿第0维拆分（4→2+2）
    const std::vector<int64_t> splitSections = {1, 2}; // 拆分规则
    const aclDataType dataType = ACL_FLOAT;            // 数据类型：float
    using DataT = float;                               // C++对应类型

    // 打印参数摘要
    DEBUG_LOG("=");
    DEBUG_LOG("Split operator config:");
    DEBUG_LOG("  input shape: [%lld, %lld]", (long long)inputXShape[0], (long long)inputXShape[1]);
    DEBUG_LOG(
        "  output shapes: [%lld, %lld], [%lld, %lld]", (long long)outputZShapes[0][0], (long long)outputZShapes[0][1],
        (long long)outputZShapes[1][0], (long long)outputZShapes[1][1]);
    DEBUG_LOG("  split axis: %lld", (long long)splitAxis);
    DEBUG_LOG("  split sections: [%lld]", (long long)splitSections[0]);
    DEBUG_LOG("=");

    // 3. 申请Host内存
    const int64_t inputXSize = GetShapeSize(inputXShape);
    const size_t inputXHostMemSize = inputXSize * sizeof(DataT);
    std::vector<DataT> inputXHostData(inputXSize);
    DEBUG_LOG("Host input buffer: size=%zu bytes, element count=%lld", inputXHostMemSize, (long long)inputXSize);

    // 计算输出总大小
    size_t outputZTotalSize = 0;
    for (size_t i = 0; i < outputZShapes.size(); ++i) {
        outputZTotalSize += GetShapeSize(outputZShapes[i]);
    }
    const size_t outputZHostMemSize = outputZTotalSize * sizeof(DataT);
    std::vector<DataT> outputZHostData(outputZTotalSize, 0);
    DEBUG_LOG("Host output buffer: total size=%zu bytes, element count=%zu", outputZHostMemSize, outputZTotalSize);

    // 生成测试数据
    for (int64_t i = 0; i < inputXSize; ++i) {
        inputXHostData[i] = static_cast<DataT>(i);
    }

    // 5. 创建ACL Tensor/TensorList
    aclTensor* inputXTensor = nullptr;
    void* inputXDeviceAddr = nullptr;
    ret = CreateAclTensor(inputXHostData, inputXShape, &inputXDeviceAddr, dataType, &inputXTensor);
    CHECK_RET(ret == SUCCESS, "create input XTensor failed");

    aclTensorList* outputZTensorList = nullptr;
    std::vector<void*> outputZDeviceAddrs;
    ret = CreateAclTensorList<DataT>(outputZShapes, outputZDeviceAddrs, dataType, &outputZTensorList);
    CHECK_RET(ret == SUCCESS, "create output ZTensorList failed");

    // 6. 创建split参数的ACL IntArray
    aclIntArray* aclSplitSections = aclCreateIntArray(splitSections.data(), splitSections.size());
    CHECK_RET(aclSplitSections != nullptr, "aclCreateIntArray for split sections failed");
    for (size_t i = 0; i <= splitSections.size(); ++i) {
        int64_t start = (i == 0) ? 0 : splitSections[i - 1];
        int64_t end = (i < splitSections.size()) ? splitSections[i] : inputXSize;
        DEBUG_LOG("Segment %zu: start=%lld, end=%lld, length=%lld", i, start, end, end - start);
    }

    // 7. 算子执行
    uint64_t workspaceSize = 0;
    aclOpExecutor* executor = nullptr;
    // 7.1 计算工作空间大小
    ret = aclnnSplitGetWorkspaceSize(
        inputXTensor, aclSplitSections, splitAxis, outputZTensorList, &workspaceSize, &executor);
    CHECK_RET(ret == ACL_SUCCESS, "aclnnSplitGetWorkspaceSize failed, error=%d", ret);
    DEBUG_LOG("Workspace size: %zu bytes", workspaceSize);

    // 7.2 申请工作空间
    void* workspaceAddr = nullptr;
    if (workspaceSize > 0) {
        ret = aclrtMalloc(&workspaceAddr, workspaceSize, ACL_MEM_MALLOC_HUGE_FIRST);
        CHECK_RET(ret == ACL_SUCCESS, "allocate workspace failed, error=%d", ret);
        DEBUG_LOG("Workspace allocated: addr=%p, size=%zu", workspaceAddr, workspaceSize);
    }

    // 7.3 执行算子
    DEBUG_LOG("Start execute aclnnSplit...");
    ret = aclnnSplit(workspaceAddr, workspaceSize, executor, stream);
    CHECK_RET(ret == ACL_SUCCESS, "aclnnSplit failed, error=%d", ret);

    // 8. 同步等待
    ret = aclrtSynchronizeStream(stream);
    CHECK_RET(ret == ACL_SUCCESS, "aclrtSynchronizeStream failed, error=%d", ret);
    INFO_LOG("split operator execute success");

    // 9. Device→Host拷贝输出（关键：逐个Tensor拷贝，验证地址）
    size_t copyOffset = 0;
    for (size_t i = 0; i < outputZShapes.size(); ++i) {
        const int64_t tensorSize = GetShapeSize(outputZShapes[i]);
        const size_t tensorMemSize = tensorSize * sizeof(DataT);
        DEBUG_LOG(
            "Copy output Tensor[%zu]: deviceAddr=%p, size=%zu bytes, offset=%zu", i, outputZDeviceAddrs[i],
            tensorMemSize, copyOffset);

        // 拷贝单个Tensor
        ret = aclrtMemcpy(
            outputZHostData.data() + copyOffset, tensorMemSize, outputZDeviceAddrs[i], tensorMemSize,
            ACL_MEMCPY_DEVICE_TO_HOST);
        CHECK_RET(ret == ACL_SUCCESS, "copy output tensor %zu failed, error=%d", i, ret);

        // 打印单个Tensor的结果
        DEBUG_LOG("Output Tensor[%zu] data:", i);
        for (int64_t j = 0; j < tensorSize; ++j) {
            size_t globalIdx = copyOffset + j;
            DEBUG_LOG("  outputZHostData[%zu] = %.2f", globalIdx, outputZHostData[globalIdx]);
        }

        copyOffset += tensorSize;
    }

    // 验证总拷贝大小
    CHECK_RET(
        copyOffset == outputZTotalSize, "output copy size mismatch: expected=%zu, actual=%zu", outputZTotalSize,
        copyOffset);

    // 11. 资源释放
    if (outputZTensorList != nullptr) {
        aclDestroyTensorList(outputZTensorList);
    }
    if (inputXTensor != nullptr) {
        aclDestroyTensor(inputXTensor);
    }
    for (size_t i = 0; i < outputZDeviceAddrs.size(); ++i) {
        if (outputZDeviceAddrs[i] != nullptr) {
            aclrtFree(outputZDeviceAddrs[i]);
        }
    }
    if (inputXDeviceAddr != nullptr) {
        aclrtFree(inputXDeviceAddr);
    }
    if (workspaceAddr != nullptr) {
        aclrtFree(workspaceAddr);
    }
    if (aclSplitSections != nullptr) {
        aclDestroyIntArray(aclSplitSections);
    }
    if (executor != nullptr) {
        aclDestroyAclOpExecutor(executor);
    }
    if (stream != nullptr) {
        aclrtDestroyStream(stream);
    }
    aclrtResetDevice(deviceId);
    aclFinalize();
    return SUCCESS;
}