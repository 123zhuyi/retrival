/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

/**
 * @file test_aclnn_xlog1py.cpp
 * @brief aclnn xlog1py 算子 NPU 调用示例
 *
 * 算子: z = x * log1p(y), x==0 -> z=0
 * 支持 broadcast, FLOAT16/FLOAT/BF16
 */

#include <iostream>
#include <memory>
#include <type_traits>
#include <vector>
#include <cmath>
#include "acl/acl.h"
#include "aclnnop/aclnn_xlog1py.h"

#define CHECK_RET(cond, msg)            \
    do {                                \
        if (!(cond)) {                  \
            printf("[FAIL] " msg "\n"); \
            return -1;                  \
        }                               \
    } while (0)

#define LOG_PRINT(msg, ...) printf(msg "\n", ##__VA_ARGS__)

int64_t GetShapeSize(const std::vector<int64_t>& shape)
{
    int64_t size = 1;
    for (auto i : shape)
        size *= i;
    return size;
}

using StreamPtr = std::unique_ptr<std::remove_pointer<aclrtStream>::type, decltype(&aclrtDestroyStream)>;
using DeviceMemPtr = std::unique_ptr<void, decltype(&aclrtFree)>;
using TensorPtr = std::unique_ptr<aclTensor, decltype(&aclDestroyTensor)>;

// Broadcast index: map flat index in output to flat index in input
static int64_t BroadcastIdx(int64_t flat, const std::vector<int64_t>& inShape, const std::vector<int64_t>& outShape)
{
    int inRank = (int)inShape.size();
    int outRank = (int)outShape.size();
    int64_t outIdx = 0, outStride = 1;
    for (int d = 0; d < outRank; d++) {
        int dimIdx = outRank - 1 - d;
        int64_t dim = outShape[dimIdx];
        int64_t coord = (flat / outStride) % dim;
        int inDimIdx = dimIdx - (outRank - inRank);
        int64_t inDim = (inDimIdx >= 0) ? inShape[inDimIdx] : 1;
        int64_t inCoord = (inDim == 1) ? 0 : coord;
        int inStride = 1;
        for (int dd = inRank - 1; dd > inDimIdx; dd--)
            inStride *= inShape[dd];
        outIdx += inCoord * inStride;
        outStride *= dim;
    }
    return outIdx;
}

std::vector<float> ComputeGolden(const std::vector<float>& x, const std::vector<int64_t>& shapeX,
                                 const std::vector<float>& y, const std::vector<int64_t>& shapeY,
                                 const std::vector<int64_t>& outShape)
{
    int64_t n = GetShapeSize(outShape);
    std::vector<float> result(n);
    for (int64_t i = 0; i < n; i++) {
        int64_t ix = BroadcastIdx(i, shapeX, outShape);
        int64_t iy = BroadcastIdx(i, shapeY, outShape);
        float fx = x[ix], fy = y[iy];
        if (fx == 0.0f) {
            result[i] = 0.0f;
        } else {
            result[i] = fx * std::log1p(fy);
        }
    }
    return result;
}

template <typename T>
bool CompareResult(const std::vector<float>& golden, const std::vector<T>& npuResult, const std::vector<int64_t>& shape,
                   const std::string& tag)
{
    int64_t n = GetShapeSize(shape);
    bool allPass = true;
    double maxMere = 0.0;
    for (int64_t i = 0; i < n; i++) {
        float g = golden[i];
        float r = static_cast<float>(npuResult[i]);
        double mere = 0.0;
        if (std::fabs(g) > 1e-6) {
            mere = std::fabs(static_cast<double>(r - g)) / std::fabs(static_cast<double>(g));
        } else {
            mere = std::fabs(static_cast<double>(r - g));
        }
        if (mere > maxMere)
            maxMere = mere;
        if (mere > 0.001) {
            LOG_PRINT("  [FAIL][%s][%ld] golden=%.6f npu=%.6f mere=%.6e", tag.c_str(), i, g, r, mere);
            allPass = false;
        }
    }
    if (allPass) {
        LOG_PRINT("  [PASS][%s] all %ld elems OK, max_mere=%.6e", tag.c_str(), n, maxMere);
    }
    return allPass;
}

int Init(int32_t deviceId, StreamPtr& stream, bool& initialized, bool& deviceSet)
{
    auto ret = aclInit(nullptr);
    CHECK_RET(ret == ACL_SUCCESS, "aclInit failed");
    initialized = true;
    ret = aclrtSetDevice(deviceId);
    CHECK_RET(ret == ACL_SUCCESS, "aclrtSetDevice failed");
    deviceSet = true;
    aclrtStream rawStream = nullptr;
    ret = aclrtCreateStream(&rawStream);
    CHECK_RET(ret == ACL_SUCCESS, "aclrtCreateStream failed");
    stream.reset(rawStream);
    return 0;
}

template <typename T>
int CreateAclTensor(const std::vector<T>& hostData, const std::vector<int64_t>& shape, aclDataType dataType,
                    DeviceMemPtr& deviceAddr, TensorPtr& tensor)
{
    auto size = GetShapeSize(shape) * sizeof(T);
    void* rawDeviceAddr = nullptr;
    auto ret = aclrtMalloc(&rawDeviceAddr, size, ACL_MEM_MALLOC_HUGE_FIRST);
    CHECK_RET(ret == ACL_SUCCESS, "aclrtMalloc failed");
    deviceAddr.reset(rawDeviceAddr);
    ret = aclrtMemcpy(deviceAddr.get(), size, hostData.data(), size, ACL_MEMCPY_HOST_TO_DEVICE);
    CHECK_RET(ret == ACL_SUCCESS, "aclrtMemcpy H2D failed");

    std::vector<int64_t> strides(shape.size(), 1);
    for (int64_t i = shape.size() - 2; i >= 0; i--) {
        strides[i] = shape[i + 1] * strides[i + 1];
    }
    aclTensor* rawTensor = aclCreateTensor(shape.data(), shape.size(), dataType, strides.data(), 0,
                                           aclFormat::ACL_FORMAT_ND, shape.data(), shape.size(), deviceAddr.get());
    CHECK_RET(rawTensor != nullptr, "aclCreateTensor failed");
    tensor.reset(rawTensor);
    return 0;
}

int RunXlog1py(const std::vector<int64_t>& shapeX, const std::vector<float>& dataX, const std::vector<int64_t>& shapeY,
               const std::vector<float>& dataY, const std::string& tag, StreamPtr& stream)
{
    LOG_PRINT("--- Test %s ---", tag.c_str());

    // Compute broadcast output shape
    int rank = std::max(shapeX.size(), shapeY.size());
    std::vector<int64_t> outShape(rank);
    for (int d = 0; d < rank; d++) {
        int dx = d - (rank - shapeX.size());
        int dy = d - (rank - shapeY.size());
        auto sx = (dx >= 0) ? shapeX[dx] : 1;
        auto sy = (dy >= 0) ? shapeY[dy] : 1;
        outShape[d] = std::max(sx, sy);
    }
    int64_t outSize = GetShapeSize(outShape);

    LOG_PRINT("  shapeX in=%ld outShape=[%ld,%ld,%ld,%ld]", dataX.size(), outShape[0], outShape[1], outShape[2],
              outShape[3]);

    // Compute golden
    auto golden = ComputeGolden(dataX, shapeX, dataY, shapeY, outShape);

    // Allocate device tensors
    TensorPtr aclX(nullptr, &aclDestroyTensor);
    DeviceMemPtr devX(nullptr, &aclrtFree);
    auto ret = CreateAclTensor(dataX, shapeX, aclDataType::ACL_FLOAT, devX, aclX);
    CHECK_RET(ret == 0, "create tensor X failed");

    TensorPtr aclY(nullptr, &aclDestroyTensor);
    DeviceMemPtr devY(nullptr, &aclrtFree);
    ret = CreateAclTensor(dataY, shapeY, aclDataType::ACL_FLOAT, devY, aclY);
    CHECK_RET(ret == 0, "create tensor Y failed");

    std::vector<float> outHostData(outSize, 0);
    TensorPtr aclOut(nullptr, &aclDestroyTensor);
    DeviceMemPtr devOut(nullptr, &aclrtFree);
    ret = CreateAclTensor(outHostData, outShape, aclDataType::ACL_FLOAT, devOut, aclOut);
    CHECK_RET(ret == 0, "create tensor Out failed");

    // Phase 1: GetWorkspaceSize
    uint64_t workspaceSize = 0;
    aclOpExecutor* executor = nullptr;
    ret = aclnnXlog1pyGetWorkspaceSize(aclX.get(), aclY.get(), aclOut.get(), &workspaceSize, &executor);
    CHECK_RET(ret == ACL_SUCCESS, "aclnnXlog1pyGetWorkspaceSize failed");

    DeviceMemPtr workspaceAddr(nullptr, &aclrtFree);
    if (workspaceSize > static_cast<uint64_t>(0)) {
        void* rawWorkspaceAddr = nullptr;
        ret = aclrtMalloc(&rawWorkspaceAddr, workspaceSize, ACL_MEM_MALLOC_HUGE_FIRST);
        CHECK_RET(ret == ACL_SUCCESS, "allocate workspace failed");
        workspaceAddr.reset(rawWorkspaceAddr);
    }

    // Phase 2: Execute
    ret = aclnnXlog1py(workspaceAddr.get(), workspaceSize, executor, stream.get());
    CHECK_RET(ret == ACL_SUCCESS, "aclnnXlog1py execute failed");

    ret = aclrtSynchronizeStream(stream.get());
    CHECK_RET(ret == ACL_SUCCESS, "aclrtSynchronizeStream failed");

    // Copy result back
    std::vector<float> npuResult(outSize, 0);
    ret = aclrtMemcpy(npuResult.data(), outSize * sizeof(float), devOut.get(), outSize * sizeof(float),
                      ACL_MEMCPY_DEVICE_TO_HOST);
    CHECK_RET(ret == ACL_SUCCESS, "copy result D2H failed");

    // Compare
    bool pass = CompareResult(golden, npuResult, outShape, tag);

    return pass ? 0 : -1;
}

int main()
{
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
    CHECK_RET(ret == ACL_SUCCESS, "Init failed");

    int numPass = 0, numFail = 0;

    // Test 1: same shape [1,2,4,4]
    {
        std::vector<int64_t> shape = {1, 2, 4, 4};
        std::vector<float> x(32), y(32);
        for (int i = 0; i < 32; i++) {
            x[i] = 2.0f;
            y[i] = 1.0f;
        }
        if (RunXlog1py(shape, x, shape, y, "same_shape", stream) == 0)
            numPass++;
        else
            numFail++;
    }

    // Test 2: broadcast x=[1,2,1,4]  y=[1,2,4,4]
    {
        std::vector<int64_t> shapeX = {1, 2, 1, 4};
        std::vector<int64_t> shapeY = {1, 2, 4, 4};
        std::vector<float> x(8), y(32);
        for (int i = 0; i < 8; i++)
            x[i] = 3.0f;
        for (int i = 0; i < 32; i++)
            y[i] = 2.0f;
        if (RunXlog1py(shapeX, x, shapeY, y, "broadcast", stream) == 0)
            numPass++;
        else
            numFail++;
    }

    // Test 3: x == 0 boundary case
    {
        std::vector<int64_t> shape = {1, 1, 8, 8};
        std::vector<float> x(64, 0.0f);
        std::vector<float> y(64, 100.0f);
        if (RunXlog1py(shape, x, shape, y, "x_eq_0", stream) == 0)
            numPass++;
        else
            numFail++;
    }

    // Test 4: scalar broadcast x=scalar, y=[4,8,16,16]
    {
        std::vector<int64_t> shapeX = {1};
        std::vector<int64_t> shapeY = {4, 8, 16, 16};
        std::vector<float> x(1, 2.5f);
        int64_t n = 8192;
        std::vector<float> y(n);
        for (int64_t i = 0; i < n; i++)
            y[i] = 1.0f + 0.1f * (i % 5);
        if (RunXlog1py(shapeX, x, shapeY, y, "scalar_broadcast", stream) == 0)
            numPass++;
        else
            numFail++;
    }

    LOG_PRINT("========================================");
    LOG_PRINT("ACLNN Xlog1py NPU results: PASS=%d  FAIL=%d", numPass, numFail);
    LOG_PRINT("========================================");

    return (numFail == 0) ? 0 : -1;
}
