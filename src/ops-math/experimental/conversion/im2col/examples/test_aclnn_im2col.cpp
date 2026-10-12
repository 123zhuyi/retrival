/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include <cstdint>
#include <cstdio>
#include <limits>
#include <string>
#include <vector>
#include "acl/acl.h"
#include "aclnnop/aclnn_im2col.h"

namespace {

constexpr size_t CHW_RANK = 3U;
constexpr size_t NCHW_RANK = 4U;
constexpr size_t SPATIAL_PAIR_SIZE = 2U;
constexpr size_t CHW_CHANNEL_DIM = 0U;
constexpr size_t CHW_HEIGHT_DIM = 1U;
constexpr size_t CHW_WIDTH_DIM = 2U;
constexpr size_t NCHW_BATCH_DIM = 0U;
constexpr size_t NCHW_CHANNEL_DIM = 1U;
constexpr size_t NCHW_HEIGHT_DIM = 2U;
constexpr size_t NCHW_WIDTH_DIM = 3U;
constexpr size_t PAIR_HEIGHT_INDEX = 0U;
constexpr size_t PAIR_WIDTH_INDEX = 1U;
constexpr int64_t SINGLE_BATCH_COUNT = 1;
constexpr int64_t INITIAL_ELEMENT_COUNT = 1;
constexpr int64_t CONTIGUOUS_INNER_STRIDE = 1;
constexpr int64_t CONTIGUOUS_STRIDE_START_OFFSET = 2;
constexpr int64_t NEXT_DIMENSION_OFFSET = 1;
constexpr int64_t SYMMETRIC_PADDING_SIDE_COUNT = 2;
constexpr size_t BOOL_VALUE_MASK = 1U;
constexpr size_t TEST_DATA_MULTIPLIER = 17U;
constexpr size_t TEST_DATA_OFFSET = 3U;
constexpr size_t BYTE_VALUE_MASK = std::numeric_limits<uint8_t>::max();
constexpr uint64_t CHECKSUM_MULTIPLIER = 131U;

struct TensorResource {
    void* deviceAddress = nullptr;
    aclTensor* tensor = nullptr;
};

bool SafeMul(int64_t lhs, int64_t rhs, int64_t& result)
{
    if (lhs < 0 || rhs < 0 || (lhs != 0 && rhs > std::numeric_limits<int64_t>::max() / lhs)) {
        return false;
    }
    result = lhs * rhs;
    return true;
}

bool Numel(const std::vector<int64_t>& shape, int64_t& elementCount)
{
    elementCount = INITIAL_ELEMENT_COUNT;
    for (int64_t dim : shape) {
        if (dim <= 0 || !SafeMul(elementCount, dim, elementCount)) {
            return false;
        }
    }
    return true;
}

bool ContiguousStrides(const std::vector<int64_t>& shape, std::vector<int64_t>& strides)
{
    strides.assign(shape.size(), CONTIGUOUS_INNER_STRIDE);
    for (int64_t i = static_cast<int64_t>(shape.size()) - CONTIGUOUS_STRIDE_START_OFFSET; i >= 0; --i) {
        if (!SafeMul(strides[i + NEXT_DIMENSION_OFFSET], shape[i + NEXT_DIMENSION_OFFSET], strides[i])) {
            return false;
        }
    }
    return true;
}

bool CalculateOutputDim(int64_t input, int64_t kernel, int64_t dilation, int64_t padding, int64_t stride,
                        int64_t& output)
{
    if (input <= 0 || kernel <= 0 || dilation <= 0 || padding < 0 || stride <= 0) {
        return false;
    }
    const __int128 effectiveKernel = static_cast<__int128>(dilation) * (kernel - 1) + 1;
    const __int128 numerator = static_cast<__int128>(input) +
                               SYMMETRIC_PADDING_SIDE_COUNT * static_cast<__int128>(padding) - effectiveKernel;
    if (numerator < 0) {
        return false;
    }
    const __int128 result = numerator / stride + 1;
    if (result <= 0 || result > std::numeric_limits<int64_t>::max()) {
        return false;
    }
    output = static_cast<int64_t>(result);
    return true;
}

bool BuildOutputShape(const std::vector<int64_t>& inputShape, const std::vector<int64_t>& kernel,
                      const std::vector<int64_t>& dilation, const std::vector<int64_t>& padding,
                      const std::vector<int64_t>& stride, std::vector<int64_t>& outputShape)
{
    if ((inputShape.size() != CHW_RANK && inputShape.size() != NCHW_RANK) || kernel.size() != SPATIAL_PAIR_SIZE ||
        dilation.size() != SPATIAL_PAIR_SIZE || padding.size() != SPATIAL_PAIR_SIZE ||
        stride.size() != SPATIAL_PAIR_SIZE) {
        return false;
    }
    const bool rank3 = inputShape.size() == CHW_RANK;
    const int64_t n = rank3 ? SINGLE_BATCH_COUNT : inputShape[NCHW_BATCH_DIM];
    const int64_t c = rank3 ? inputShape[CHW_CHANNEL_DIM] : inputShape[NCHW_CHANNEL_DIM];
    const int64_t h = rank3 ? inputShape[CHW_HEIGHT_DIM] : inputShape[NCHW_HEIGHT_DIM];
    const int64_t w = rank3 ? inputShape[CHW_WIDTH_DIM] : inputShape[NCHW_WIDTH_DIM];
    int64_t outH = 0;
    int64_t outW = 0;
    int64_t outChannels = 0;
    int64_t outSpatial = 0;
    if (n <= 0 || c <= 0 ||
        !CalculateOutputDim(h, kernel[PAIR_HEIGHT_INDEX], dilation[PAIR_HEIGHT_INDEX], padding[PAIR_HEIGHT_INDEX],
                            stride[PAIR_HEIGHT_INDEX], outH) ||
        !CalculateOutputDim(w, kernel[PAIR_WIDTH_INDEX], dilation[PAIR_WIDTH_INDEX], padding[PAIR_WIDTH_INDEX],
                            stride[PAIR_WIDTH_INDEX], outW) ||
        !SafeMul(c, kernel[PAIR_HEIGHT_INDEX], outChannels) ||
        !SafeMul(outChannels, kernel[PAIR_WIDTH_INDEX], outChannels) || !SafeMul(outH, outW, outSpatial)) {
        return false;
    }
    outputShape = rank3 ? std::vector<int64_t>{outChannels, outSpatial} :
                          std::vector<int64_t>{n, outChannels, outSpatial};
    return true;
}

bool CreateTensor(const std::vector<int64_t>& shape, aclDataType dtype, size_t elementBytes, bool boolValues,
                  TensorResource& resource)
{
    int64_t elementCount = 0;
    if (elementBytes == 0U || !Numel(shape, elementCount) ||
        static_cast<uint64_t>(elementCount) > std::numeric_limits<size_t>::max() / elementBytes) {
        return false;
    }
    const size_t bytes = static_cast<size_t>(elementCount) * elementBytes;
    std::vector<uint8_t> host(bytes, 0);
    for (size_t i = 0; i < host.size(); ++i) {
        host[i] = boolValues ? static_cast<uint8_t>((i / elementBytes) & BOOL_VALUE_MASK) :
                               static_cast<uint8_t>((TEST_DATA_MULTIPLIER * i + TEST_DATA_OFFSET) & BYTE_VALUE_MASK);
    }
    if (aclrtMalloc(&resource.deviceAddress, bytes, ACL_MEM_MALLOC_HUGE_FIRST) != ACL_SUCCESS) {
        return false;
    }
    if (aclrtMemcpy(resource.deviceAddress, bytes, host.data(), bytes, ACL_MEMCPY_HOST_TO_DEVICE) != ACL_SUCCESS) {
        aclrtFree(resource.deviceAddress);
        resource.deviceAddress = nullptr;
        return false;
    }
    std::vector<int64_t> strides;
    if (!ContiguousStrides(shape, strides)) {
        aclrtFree(resource.deviceAddress);
        resource.deviceAddress = nullptr;
        return false;
    }
    resource.tensor = aclCreateTensor(shape.data(), shape.size(), dtype, strides.data(), 0, ACL_FORMAT_ND, shape.data(),
                                      shape.size(), resource.deviceAddress);
    if (resource.tensor == nullptr) {
        aclrtFree(resource.deviceAddress);
        resource.deviceAddress = nullptr;
        return false;
    }
    return true;
}

void DestroyTensor(TensorResource& resource)
{
    if (resource.tensor != nullptr) {
        aclDestroyTensor(resource.tensor);
        resource.tensor = nullptr;
    }
    if (resource.deviceAddress != nullptr) {
        aclrtFree(resource.deviceAddress);
        resource.deviceAddress = nullptr;
    }
}

int RunCase(const std::string& name, aclrtStream stream, aclDataType dtype, size_t elementBytes,
            const std::vector<int64_t>& inputShape, const std::vector<int64_t>& kernel,
            const std::vector<int64_t>& dilation, const std::vector<int64_t>& padding,
            const std::vector<int64_t>& stride)
{
    if (elementBytes == 0U) {
        std::printf("%s: element size must be positive\n", name.c_str());
        return 1;
    }

    std::vector<int64_t> outputShape;
    TensorResource input;
    TensorResource output;
    aclIntArray* kernelArray = nullptr;
    aclIntArray* dilationArray = nullptr;
    aclIntArray* paddingArray = nullptr;
    aclIntArray* strideArray = nullptr;
    void* workspace = nullptr;
    int result = 1;

    do {
        if (!BuildOutputShape(inputShape, kernel, dilation, padding, stride, outputShape) ||
            !CreateTensor(inputShape, dtype, elementBytes, dtype == ACL_BOOL, input) ||
            !CreateTensor(outputShape, dtype, elementBytes, false, output)) {
            std::printf("%s: tensor creation failed\n", name.c_str());
            break;
        }
        kernelArray = aclCreateIntArray(kernel.data(), kernel.size());
        dilationArray = aclCreateIntArray(dilation.data(), dilation.size());
        paddingArray = aclCreateIntArray(padding.data(), padding.size());
        strideArray = aclCreateIntArray(stride.data(), stride.size());
        if (kernelArray == nullptr || dilationArray == nullptr || paddingArray == nullptr || strideArray == nullptr) {
            std::printf("%s: attribute creation failed\n", name.c_str());
            break;
        }

        uint64_t workspaceSize = 0;
        aclOpExecutor* executor = nullptr;
        aclnnStatus ret = aclnnIm2colGetWorkspaceSize(input.tensor, kernelArray, dilationArray, paddingArray,
                                                      strideArray, output.tensor, &workspaceSize, &executor);
        if (ret == ACL_SUCCESS && workspaceSize > 0) {
            ret = aclrtMalloc(&workspace, workspaceSize, ACL_MEM_MALLOC_HUGE_FIRST);
        }
        if (ret == ACL_SUCCESS) {
            ret = aclnnIm2col(workspace, workspaceSize, executor, stream);
        }
        if (ret == ACL_SUCCESS) {
            ret = aclrtSynchronizeStream(stream);
        }

        int64_t outputElements = 0;
        if (ret != ACL_SUCCESS || !Numel(outputShape, outputElements) ||
            static_cast<uint64_t>(outputElements) > std::numeric_limits<size_t>::max() / elementBytes) {
            std::printf("%s: execution failed, status=%d\n", name.c_str(), static_cast<int>(ret));
            break;
        }
        const size_t outputBytes = static_cast<size_t>(outputElements) * elementBytes;
        std::vector<uint8_t> hostOutput(outputBytes, 0);
        ret = aclrtMemcpy(hostOutput.data(), outputBytes, output.deviceAddress, outputBytes, ACL_MEMCPY_DEVICE_TO_HOST);
        if (ret != ACL_SUCCESS) {
            std::printf("%s: output copy failed, status=%d\n", name.c_str(), static_cast<int>(ret));
            break;
        }
        uint64_t checksum = 0;
        for (uint8_t byte : hostOutput) {
            checksum = checksum * CHECKSUM_MULTIPLIER + byte;
        }
        std::printf("%s: output_elements=%ld checksum=%lu\n", name.c_str(), outputElements, checksum);
        result = 0;
    } while (false);

    if (workspace != nullptr) {
        aclrtFree(workspace);
    }
    if (kernelArray != nullptr) {
        aclDestroyIntArray(kernelArray);
    }
    if (dilationArray != nullptr) {
        aclDestroyIntArray(dilationArray);
    }
    if (paddingArray != nullptr) {
        aclDestroyIntArray(paddingArray);
    }
    if (strideArray != nullptr) {
        aclDestroyIntArray(strideArray);
    }
    DestroyTensor(input);
    DestroyTensor(output);
    return result;
}

} // namespace

int main()
{
    constexpr int32_t deviceId = 0;
    aclrtStream stream = nullptr;
    aclError ret = aclInit(nullptr);
    if (ret != ACL_SUCCESS) {
        std::printf("aclInit failed, status=%d\n", static_cast<int>(ret));
        return 1;
    }
    ret = aclrtSetDevice(deviceId);
    if (ret != ACL_SUCCESS) {
        std::printf("aclrtSetDevice failed, status=%d\n", static_cast<int>(ret));
        aclFinalize();
        return 1;
    }
    ret = aclrtCreateStream(&stream);
    if (ret != ACL_SUCCESS) {
        std::printf("aclrtCreateStream failed, status=%d\n", static_cast<int>(ret));
        aclrtResetDevice(deviceId);
        aclFinalize();
        return 1;
    }

    int result = 0;
    result |= RunCase("fp32_rank3", stream, ACL_FLOAT, sizeof(float), {2, 5, 6}, {3, 2}, {1, 1}, {1, 0}, {2, 1});
    result |= RunCase("fp16_contiguous", stream, ACL_FLOAT16, sizeof(uint16_t), {2, 3, 7, 8}, {3, 3}, {1, 1}, {1, 1},
                      {1, 1});
    result |= RunCase("bf16_dilation", stream, ACL_BF16, sizeof(uint16_t), {1, 2, 8, 9}, {3, 2}, {2, 3}, {2, 1},
                      {2, 2});
    result |= RunCase("bool_gather", stream, ACL_BOOL, sizeof(uint8_t), {1, 2, 33, 35}, {3, 5}, {1, 2}, {2, 4}, {2, 1});

    aclrtDestroyStream(stream);
    aclrtResetDevice(deviceId);
    aclFinalize();
    return result;
}
