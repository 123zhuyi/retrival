/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This file is a part of the CANN Open Software.
 * Licensed under CANN Open Software License Agreement Version 2.0 (the "License").
 */

#include <iostream>
#include <vector>
#include "acl/acl.h"
#include "aclnn_div_v3.h"

using DataType = float;

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
    std::vector<DataType> resultData(size, 0);
    auto ret = aclrtMemcpy(
        resultData.data(), resultData.size() * sizeof(resultData[0]), *deviceAddr, size * sizeof(resultData[0]),
        ACL_MEMCPY_DEVICE_TO_HOST);
    CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("copy result from device to host failed. ERROR: %d\n", ret); return);
    for (int64_t i = 0; i < size && i < 32; i++) {
        LOG_PRINT("result[%ld] is: %f\n", i, resultData[i]);
    }
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

int main()
{
    int32_t deviceId = 0;
    aclrtStream stream;
    auto ret = Init(deviceId, &stream);
    CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("Init failed. ERROR: %d\n", ret); return ret);

    // construct input tensors
    std::vector<int64_t> shape = {4, 256};
    int64_t totalSize = GetShapeSize(shape);
    std::vector<DataType> selfHostData(totalSize, 7.0f);
    std::vector<DataType> otherHostData(totalSize, 2.0f);
    std::vector<DataType> outHostData(totalSize, 0.0f);

    aclTensor* selfTensor = nullptr;
    void* selfDeviceAddr = nullptr;
    ret = CreateAclTensor(selfHostData, shape, &selfDeviceAddr, aclDataType::ACL_FLOAT, &selfTensor);
    CHECK_RET(ret == ACL_SUCCESS, return ret);

    aclTensor* otherTensor = nullptr;
    void* otherDeviceAddr = nullptr;
    ret = CreateAclTensor(otherHostData, shape, &otherDeviceAddr, aclDataType::ACL_FLOAT, &otherTensor);
    CHECK_RET(ret == ACL_SUCCESS, return ret);

    aclTensor* outTensor = nullptr;
    void* outDeviceAddr = nullptr;
    ret = CreateAclTensor(outHostData, shape, &outDeviceAddr, aclDataType::ACL_FLOAT, &outTensor);
    CHECK_RET(ret == ACL_SUCCESS, return ret);

    // test all three modes
    const char* modeNames[] = {"RealDiv", "TruncDiv", "FloorDiv"};
    for (int64_t mode = 0; mode <= 2; mode++) {
        LOG_PRINT("\n=== Testing mode=%ld (%s): 7.0 / 2.0 ===\n", mode, modeNames[mode]);

        uint64_t workspaceSize = 0;
        aclOpExecutor* executor = nullptr;

        ret = aclnnDivV3GetWorkspaceSize(selfTensor, otherTensor, mode, outTensor, &workspaceSize, &executor);
        CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclnnDivV3GetWorkspaceSize failed. ERROR: %d\n", ret); return ret);

        void* workspaceAddr = nullptr;
        if (workspaceSize > 0) {
            ret = aclrtMalloc(&workspaceAddr, workspaceSize, ACL_MEM_MALLOC_HUGE_FIRST);
            CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("allocate workspace failed. ERROR: %d\n", ret); return ret);
        }

        ret = aclnnDivV3(workspaceAddr, workspaceSize, executor, stream);
        CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclnnDivV3 failed. ERROR: %d\n", ret); return ret);

        ret = aclrtSynchronizeStream(stream);
        CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclrtSynchronizeStream failed. ERROR: %d\n", ret); return ret);

        PrintOutResult(shape, &outDeviceAddr);

        if (workspaceSize > 0) {
            aclrtFree(workspaceAddr);
        }
    }

    // cleanup
    aclDestroyTensor(selfTensor);
    aclDestroyTensor(otherTensor);
    aclDestroyTensor(outTensor);
    aclrtFree(selfDeviceAddr);
    aclrtFree(otherDeviceAddr);
    aclrtFree(outDeviceAddr);
    aclrtDestroyStream(stream);
    aclrtResetDevice(deviceId);
    aclFinalize();

    return 0;
}
