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

// 设备侧 Tensor 持有者：析构时按创建逆序自动释放 tensor 与 device 内存
class DeviceTensor {
public:
    // bytesPerElem=0.5 表示 4-bit 紧凑打包（INT4/FP4，每字节 2 个值），内存按 numel/2 字节申请
    DeviceTensor(const std::vector<int64_t>& viewShape, const std::vector<int64_t>& storageShape,
                 const std::vector<int64_t>& strides, aclDataType dtype, aclFormat format, double bytesPerElem)
    {
        int64_t storageSize = GetShapeSize(storageShape);
        bytes_ = static_cast<int64_t>(storageSize * bytesPerElem);

        std::vector<int8_t> hostData(bytes_, 0);
        auto ret = aclrtMalloc(&deviceAddr_, bytes_, ACL_MEM_MALLOC_HUGE_FIRST);
        CHECK_RET(ret == ACL_SUCCESS, std::cout << "Malloc device memory failed" << std::endl; return);
        ret = aclrtMemcpy(deviceAddr_, bytes_, hostData.data(), bytes_, ACL_MEMCPY_HOST_TO_DEVICE);
        CHECK_RET(ret == ACL_SUCCESS, std::cout << "Memcpy H2D failed" << std::endl; return);
        const int64_t* stridesData = strides.empty() ? nullptr : strides.data();
        tensor_ = aclCreateTensor(viewShape.data(), viewShape.size(), dtype, stridesData, 0, format,
                                  storageShape.data(), storageShape.size(), deviceAddr_);
        CHECK_RET(tensor_ != nullptr, std::cout << "Create tensor failed" << std::endl; return);
    }

    ~DeviceTensor()
    {
        if (tensor_ != nullptr) {
            aclDestroyTensor(tensor_);
            tensor_ = nullptr;
        }
        if (deviceAddr_ != nullptr) {
            aclrtFree(deviceAddr_);
            deviceAddr_ = nullptr;
        }
    }

    aclTensor* Get() const { return tensor_; }
    bool IsValid() const { return tensor_ != nullptr && deviceAddr_ != nullptr; }

private:
    aclTensor* tensor_ = nullptr;
    void* deviceAddr_ = nullptr;
    int64_t bytes_ = 0;
};

int RunPreprocess(aclTensor* weight, aclTensor* weightScale, aclTensor* weightOffset, aclDataType xDtype,
                  int64_t kGroupSize, aclTensor* outWeight, aclTensor* outWeightScale, aclTensor* outWeightOffset,
                  aclrtStream stream)
{
    // A16W4 数据流无 xScale 语义，xScaleDtype 固定传 ACL_DT_UNDEFINED
    aclDataType xScaleDtype = ACL_DT_UNDEFINED;

    // 1. 获取 workspace 与执行器
    uint64_t workspaceSize = 0;
    aclOpExecutor* executor = nullptr;
    auto ret = aclnnWeightQuantPreprocessGetWorkspaceSize(weight, weightScale, weightOffset, nullptr, // biasOptional
                                                          xDtype, xScaleDtype, kGroupSize, outWeight, outWeightScale,
                                                          outWeightOffset, nullptr, // outBiasOptional
                                                          &workspaceSize, &executor);
    CHECK_RET(ret == ACL_SUCCESS, std::cout << "GetWorkspaceSize failed, ret=" << ret << std::endl; return ret);

    void* workspaceAddr = nullptr;
    std::unique_ptr<void, aclError (*)(void*)> workspaceAddrPtr(nullptr, aclrtFree);
    if (workspaceSize > 0) {
        ret = aclrtMalloc(&workspaceAddr, workspaceSize, ACL_MEM_MALLOC_HUGE_FIRST);
        CHECK_RET(ret == ACL_SUCCESS, std::cout << "Malloc workspace failed" << std::endl; return ret);
        workspaceAddrPtr.reset(workspaceAddr);
    }

    // 2. 执行计算
    ret = aclnnWeightQuantPreprocess(workspaceAddr, workspaceSize, executor, stream);
    CHECK_RET(ret == ACL_SUCCESS, std::cout << "Preprocess failed, ret=" << ret << std::endl; return ret);

    ret = aclrtSynchronizeStream(stream);
    CHECK_RET(ret == ACL_SUCCESS, std::cout << "Synchronize failed" << std::endl; return ret);
    return ACL_SUCCESS;
}

// 场景一：MM_A16S4_PERCHANNEL（INT4 perchannel，非转置 weight -> FRACTAL_NZ_C0_16 输出，offset 直拷透传）
int TestA16S4PerChannelNz(aclrtStream stream)
{
    int64_t k = 256;
    int64_t n = 256;

    // weight: INT4，非转置连续 [K, N]；紧凑 4-bit 打包维（N）须为偶数
    DeviceTensor weight({k, n}, {k, n}, {n, 1}, ACL_INT4, ACL_FORMAT_ND, 0.5);
    CHECK_RET(weight.IsValid(), return ACL_ERROR_FAILURE);

    // weightScale: perchannel {N}，FP16；perchannel 场景 kGroupSize 必须为 0
    DeviceTensor weightScale({n}, {n}, {1}, ACL_FLOAT16, ACL_FORMAT_ND, 2);
    CHECK_RET(weightScale.IsValid(), return ACL_ERROR_FAILURE);

    // weightOffsetOptional: A16S4 各数据流支持透传，要求与 weightScale 同形同 dtype
    DeviceTensor weightOffset({n}, {n}, {1}, ACL_FLOAT16, ACL_FORMAT_ND, 2);
    CHECK_RET(weightOffset.IsValid(), return ACL_ERROR_FAILURE);

    // outWeight: 用户自行构造，viewShape 与 weight 相同；NZ_C0_16 的 storageShape 为
    // {ceildiv(N, 16), ceildiv(K, 16), 16, 16}（N 块在前，区别于 A8W4 NZ_C0_32 的 K 块在前）
    std::vector<int64_t> outStorageShape = {CEIL_DIV(n, 16), CEIL_DIV(k, 16), 16, 16};
    DeviceTensor outWeight({k, n}, outStorageShape, {}, ACL_INT4, ACL_FORMAT_FRACTAL_NZ_C0_16, 0.5);
    CHECK_RET(outWeight.IsValid(), return ACL_ERROR_FAILURE);

    // outWeightScale/outWeightOffset: viewShape、storageShape、dtype 均与输入相同（直拷）
    DeviceTensor outWeightScale({n}, {n}, {1}, ACL_FLOAT16, ACL_FORMAT_ND, 2);
    CHECK_RET(outWeightScale.IsValid(), return ACL_ERROR_FAILURE);
    DeviceTensor outWeightOffset({n}, {n}, {1}, ACL_FLOAT16, ACL_FORMAT_ND, 2);
    CHECK_RET(outWeightOffset.IsValid(), return ACL_ERROR_FAILURE);

    auto ret = RunPreprocess(weight.Get(), weightScale.Get(), weightOffset.Get(), ACL_FLOAT16, 0, outWeight.Get(),
                             outWeightScale.Get(), outWeightOffset.Get(), stream);
    CHECK_RET(ret == ACL_SUCCESS, std::cout << "TestA16S4PerChannelNz failed" << std::endl; return ret);
    std::cout << "TestA16S4PerChannelNz success" << std::endl;
    return ACL_SUCCESS;
}

// 场景二：MM_A16S4_PERTENSOR（INT4 pertensor，一律 ND 直拷，与转置无关）
int TestA16S4PerTensorNd(aclrtStream stream)
{
    int64_t k = 256;
    int64_t n = 256;

    // weight: INT4，连续 [K, N]（pertensor 对转置/非转置均按字节直拷）
    DeviceTensor weight({k, n}, {k, n}, {n, 1}, ACL_INT4, ACL_FORMAT_ND, 0.5);
    CHECK_RET(weight.IsValid(), return ACL_ERROR_FAILURE);

    // weightScale: pertensor 单元素 {1}，FP16；kGroupSize 必须为 0
    DeviceTensor weightScale({1}, {1}, {1}, ACL_FLOAT16, ACL_FORMAT_ND, 2);
    CHECK_RET(weightScale.IsValid(), return ACL_ERROR_FAILURE);
    DeviceTensor weightOffset({1}, {1}, {1}, ACL_FLOAT16, ACL_FORMAT_ND, 2);
    CHECK_RET(weightOffset.IsValid(), return ACL_ERROR_FAILURE);

    // outWeight: ND 直拷，format 为 ND，storageShape 与 viewShape 相同
    DeviceTensor outWeight({k, n}, {k, n}, {}, ACL_INT4, ACL_FORMAT_ND, 0.5);
    CHECK_RET(outWeight.IsValid(), return ACL_ERROR_FAILURE);

    DeviceTensor outWeightScale({1}, {1}, {1}, ACL_FLOAT16, ACL_FORMAT_ND, 2);
    CHECK_RET(outWeightScale.IsValid(), return ACL_ERROR_FAILURE);
    DeviceTensor outWeightOffset({1}, {1}, {1}, ACL_FLOAT16, ACL_FORMAT_ND, 2);
    CHECK_RET(outWeightOffset.IsValid(), return ACL_ERROR_FAILURE);

    auto ret = RunPreprocess(weight.Get(), weightScale.Get(), weightOffset.Get(), ACL_FLOAT16, 0, outWeight.Get(),
                             outWeightScale.Get(), outWeightOffset.Get(), stream);
    CHECK_RET(ret == ACL_SUCCESS, std::cout << "TestA16S4PerTensorNd failed" << std::endl; return ret);
    std::cout << "TestA16S4PerTensorNd success" << std::endl;
    return ACL_SUCCESS;
}

// WeightQuantPreprocess currently supports Ascend 950 only.
int main()
{
    int32_t deviceId = 0;
    aclrtStream stream = nullptr;
    AclRuntimeGuard aclGuard(deviceId);
    auto ret = aclGuard.Init(&stream);
    CHECK_RET(ret == ACL_SUCCESS, std::cout << "Init failed" << std::endl; return ret);

    ret = TestA16S4PerChannelNz(stream);
    CHECK_RET(ret == ACL_SUCCESS, return ret);

    ret = TestA16S4PerTensorNd(stream);
    CHECK_RET(ret == ACL_SUCCESS, return ret);

    std::cout << "All A16S4 examples run success" << std::endl;
    return 0;
}
