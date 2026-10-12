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
#include <memory>
#include <vector>
#include "acl/acl.h"
#include "aclnnop/aclnn_weight_quant_preprocess.h"

#define CHECK_RET(cond, return_expr) \
    do {                             \
        if (!(cond)) {               \
            return_expr;             \
        }                            \
    } while (0)

#define CEIL_DIV(x, y) (((x) + (y) - 1) / (y))

int64_t GetShapeSize(const std::vector<int64_t>& shape)
{
    int64_t size = 1;
    for (auto d : shape)
        size *= d;
    return size;
}

class AclRuntimeGuard {
public:
    explicit AclRuntimeGuard(int32_t deviceId) : deviceId_(deviceId) {}

    ~AclRuntimeGuard()
    {
        if (stream_ != nullptr) {
            aclrtDestroyStream(stream_);
            stream_ = nullptr;
        }
        if (deviceSet_) {
            aclrtResetDevice(deviceId_);
            deviceSet_ = false;
        }
        if (aclInited_) {
            aclFinalize();
            aclInited_ = false;
        }
    }

    int Init(aclrtStream* stream)
    {
        auto ret = aclInit(nullptr);
        CHECK_RET(ret == ACL_SUCCESS, return ret);
        aclInited_ = true;

        ret = aclrtSetDevice(deviceId_);
        CHECK_RET(ret == ACL_SUCCESS, return ret);
        deviceSet_ = true;

        ret = aclrtCreateStream(stream);
        CHECK_RET(ret == ACL_SUCCESS, return ret);
        stream_ = *stream;
        return ACL_SUCCESS;
    }

private:
    int32_t deviceId_;
    aclrtStream stream_ = nullptr;
    bool aclInited_ = false;
    bool deviceSet_ = false;
};

// WeightQuantPreprocess currently supports Ascend 950 only.
int main()
{
    int32_t deviceId = 0;
    aclrtStream stream = nullptr;
    AclRuntimeGuard aclGuard(deviceId);
    auto ret = aclGuard.Init(&stream);
    CHECK_RET(ret == ACL_SUCCESS, std::cout << "Init failed" << std::endl; return ret);

    // weight: FLOAT4_E2M1, transposed (MM_MX_A8W4)
    int64_t k = 64;
    int64_t n = 128;
    int64_t C0 = 32; // FLOAT4_E2M1 对应 C0=32

    std::vector<int64_t> weightViewShape = {k, n};
    std::vector<int64_t> weightStorageShape = {n, k};
    std::vector<int64_t> weightStrides = {1, k};
    int64_t weightStorageSize = GetShapeSize(weightStorageShape);
    int64_t weightBytes = weightStorageSize / 2; // FP4: 4 bits = 0.5 bytes per element

    std::vector<int8_t> weightHostData(weightBytes, 0);
    void* weightDeviceAddr = nullptr;
    ret = aclrtMalloc(&weightDeviceAddr, weightBytes, ACL_MEM_MALLOC_HUGE_FIRST);
    CHECK_RET(ret == ACL_SUCCESS, std::cout << "Malloc weight failed" << std::endl; return ret);
    std::unique_ptr<void, aclError (*)(void*)> weightDeviceAddrPtr(weightDeviceAddr, aclrtFree);
    ret = aclrtMemcpy(weightDeviceAddr, weightBytes, weightHostData.data(), weightBytes, ACL_MEMCPY_HOST_TO_DEVICE);
    CHECK_RET(ret == ACL_SUCCESS, std::cout << "Memcpy weight failed" << std::endl; return ret);
    aclTensor* weight = aclCreateTensor(weightViewShape.data(), weightViewShape.size(), ACL_FLOAT4_E2M1,
                                        weightStrides.data(), 0, ACL_FORMAT_ND, weightStorageShape.data(),
                                        weightStorageShape.size(), weightDeviceAddr);
    std::unique_ptr<aclTensor, aclnnStatus (*)(const aclTensor*)> weightPtr(weight, aclDestroyTensor);
    CHECK_RET(weight != nullptr, std::cout << "Create weight tensor failed" << std::endl; return ACL_ERROR_FAILURE);

    // weightScale: FLOAT8_E8M0, 3-D transposed (MM_MX_A8W4)
    // viewShape: {ceildiv(K,64), N, 2} = {1, 128, 2}
    // storageShape: {N, ceildiv(K,64), 2} = {128, 1, 2}
    // transposed stride: {2, 2, 1} (dim0 <-> dim1)
    std::vector<int64_t> scaleViewShape = {k / 64, n, 2};
    std::vector<int64_t> scaleStorageShape = {n, k / 64, 2};
    std::vector<int64_t> scaleStrides = {2, 2, 1};
    int64_t scaleStorageSize = GetShapeSize(scaleStorageShape);
    int64_t scaleBytes = scaleStorageSize; // FP8: 1 byte per element

    std::vector<int8_t> scaleHostData(scaleBytes, 0);
    void* scaleDeviceAddr = nullptr;
    ret = aclrtMalloc(&scaleDeviceAddr, scaleBytes, ACL_MEM_MALLOC_HUGE_FIRST);
    CHECK_RET(ret == ACL_SUCCESS, std::cout << "Malloc weightScale failed" << std::endl; return ret);
    std::unique_ptr<void, aclError (*)(void*)> scaleDeviceAddrPtr(scaleDeviceAddr, aclrtFree);
    ret = aclrtMemcpy(scaleDeviceAddr, scaleBytes, scaleHostData.data(), scaleBytes, ACL_MEMCPY_HOST_TO_DEVICE);
    CHECK_RET(ret == ACL_SUCCESS, std::cout << "Memcpy weightScale failed" << std::endl; return ret);
    aclTensor* weightScale = aclCreateTensor(scaleViewShape.data(), scaleViewShape.size(), ACL_FLOAT8_E8M0,
                                             scaleStrides.data(), 0, ACL_FORMAT_ND, scaleStorageShape.data(),
                                             scaleStorageShape.size(), scaleDeviceAddr);
    std::unique_ptr<aclTensor, aclnnStatus (*)(const aclTensor*)> weightScalePtr(weightScale, aclDestroyTensor);
    CHECK_RET(weightScale != nullptr, std::cout << "Create weightScale tensor failed" << std::endl;
              return ACL_ERROR_FAILURE);

    // 用户自行构造 outWeight (FRACTAL_NZ_C0_32)
    // viewShape 与 weight viewShape 相同，storageShape 按公式计算
    std::vector<int64_t> outWeightViewShape = {k, n};
    std::vector<int64_t> outWeightStorageShape = {CEIL_DIV(k, C0), CEIL_DIV(n, 16), 16, C0};
    int64_t outWeightStorageSize = GetShapeSize(outWeightStorageShape);
    int64_t outWeightBytes = outWeightStorageSize / 2; // FP4

    void* outWeightDeviceAddr = nullptr;
    ret = aclrtMalloc(&outWeightDeviceAddr, outWeightBytes, ACL_MEM_MALLOC_HUGE_FIRST);
    CHECK_RET(ret == ACL_SUCCESS, std::cout << "Malloc outWeight failed" << std::endl; return ret);
    std::unique_ptr<void, aclError (*)(void*)> outWeightDeviceAddrPtr(outWeightDeviceAddr, aclrtFree);
    aclTensor* outWeight = aclCreateTensor(outWeightViewShape.data(), outWeightViewShape.size(), ACL_FLOAT4_E2M1,
                                           nullptr, 0, ACL_FORMAT_FRACTAL_NZ_C0_32, outWeightStorageShape.data(),
                                           outWeightStorageShape.size(), outWeightDeviceAddr);
    std::unique_ptr<aclTensor, aclnnStatus (*)(const aclTensor*)> outWeightPtr(outWeight, aclDestroyTensor);
    CHECK_RET(outWeight != nullptr, std::cout << "Create outWeight tensor failed" << std::endl;
              return ACL_ERROR_FAILURE);

    // 构造 outWeightScale (viewShape 和 storageShape 都与 weightScale 相同)
    // 根据实现要求：outWeightScale 的 viewShape 和 storageShape 必须都与 weightScale 相同
    void* outScaleDeviceAddr = nullptr;
    ret = aclrtMalloc(&outScaleDeviceAddr, scaleBytes, ACL_MEM_MALLOC_HUGE_FIRST);
    CHECK_RET(ret == ACL_SUCCESS, std::cout << "Malloc outWeightScale failed" << std::endl; return ret);
    std::unique_ptr<void, aclError (*)(void*)> outScaleDeviceAddrPtr(outScaleDeviceAddr, aclrtFree);
    aclTensor* outWeightScale = aclCreateTensor(scaleViewShape.data(), scaleViewShape.size(), ACL_FLOAT8_E8M0,
                                                scaleStrides.data(), 0, ACL_FORMAT_ND, scaleStorageShape.data(),
                                                scaleStorageShape.size(), outScaleDeviceAddr);
    std::unique_ptr<aclTensor, aclnnStatus (*)(const aclTensor*)> outWeightScalePtr(outWeightScale, aclDestroyTensor);
    CHECK_RET(outWeightScale != nullptr, std::cout << "Create outWeightScale tensor failed" << std::endl;
              return ACL_ERROR_FAILURE);

    aclDataType xDtype = ACL_FLOAT8_E4M3FN;
    aclDataType xScaleDtype = ACL_FLOAT8_E8M0;
    int64_t kGroupSize = 32;

    // 1. 获取 workspace 与执行器
    uint64_t workspaceSize = 0;
    aclOpExecutor* executor = nullptr;
    ret = aclnnWeightQuantPreprocessGetWorkspaceSize(
        weight, weightScale, nullptr, nullptr, // weightOffsetOptional, biasOptional
        xDtype, xScaleDtype, kGroupSize, outWeight, outWeightScale, nullptr, nullptr, &workspaceSize, &executor);
    CHECK_RET(ret == ACL_SUCCESS, std::cout << "GetWorkspaceSize failed" << std::endl; return ret);

    void* workspaceAddr = nullptr;
    std::unique_ptr<void, aclError (*)(void*)> workspaceAddrPtr(nullptr, aclrtFree);
    if (workspaceSize > 0) {
        ret = aclrtMalloc(&workspaceAddr, workspaceSize, ACL_MEM_MALLOC_HUGE_FIRST);
        CHECK_RET(ret == ACL_SUCCESS, std::cout << "Malloc workspace failed" << std::endl; return ret);
        workspaceAddrPtr.reset(workspaceAddr);
    }

    // 2. 执行计算
    ret = aclnnWeightQuantPreprocess(workspaceAddr, workspaceSize, executor, stream);
    CHECK_RET(ret == ACL_SUCCESS, std::cout << "Preprocess failed" << std::endl; return ret);

    ret = aclrtSynchronizeStream(stream);
    CHECK_RET(ret == ACL_SUCCESS, std::cout << "Synchronize failed" << std::endl; return ret);

    // 3. 释放资源
    workspaceAddrPtr.reset();
    outWeightScalePtr.reset();
    outWeightPtr.reset();
    weightScalePtr.reset();
    weightPtr.reset();
    outScaleDeviceAddrPtr.reset();
    outWeightDeviceAddrPtr.reset();
    scaleDeviceAddrPtr.reset();
    weightDeviceAddrPtr.reset();
    return 0;
}
