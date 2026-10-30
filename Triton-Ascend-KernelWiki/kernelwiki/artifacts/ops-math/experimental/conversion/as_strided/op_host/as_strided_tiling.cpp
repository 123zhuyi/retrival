/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

/*!
 * \file as_strided_tiling.cpp
 * \brief
 */

#include <algorithm>
#include <cstdint>
#include "graph/types.h"
#include "log/log.h"
#include "register/op_impl_registry.h"
#include "platform/platform_ascendc.h"
#include "../op_kernel/as_strided_tiling_data.h"
#include "../op_kernel/as_strided_tiling_key.h"

namespace optiling {

namespace {

constexpr size_t IN_X = 0;
constexpr size_t IN_SIZE = 1;
constexpr size_t IN_STRIDE = 2;
constexpr size_t IN_STORAGE_OFFSET = 3;
constexpr int64_t DIM_LIMIT = AS_STRIDED_MAX_DIMS;

// UB bytes reserved for kernel stack / tiling overhead
constexpr int64_t RESERVED_UB_BYTES = 1024;

// DataCopy blockCount hardware limit (uint16_t max for DataCopyExtParams)
constexpr int64_t MAX_BLOCK_COUNT = 4095;

// Empirical core-sizing targets: small copies should not pay 40-core launch overhead.
constexpr int64_t TARGET_BYTES_PER_CORE = 6144;
constexpr int64_t RANK1_STRIDE_TARGET_BYTES_PER_CORE = 512;
constexpr int64_t RANK1_SPAN_ELEMENT_LIMIT = 16384;
constexpr int64_t SMALL_SPAN_MAX_OUTPUT_ELEMENTS = 1024;
constexpr int64_t SMALL_SPAN_MAX_LAST_DIM = 8;
constexpr int64_t ROW_BATCH_MAX_OUTPUT_ELEMENTS = 512;
constexpr int64_t ROW_BATCH_MAX_LAST_DIM = 4;
constexpr int64_t SMALL_ROW_MC_MIN_OUTPUT_ELEMENTS = 128;
constexpr int64_t SMALL_ROW_MC_MIN_AXIS0 = 32;
constexpr int64_t SMALL_ROW_TARGET_ROWS_PER_CORE = 16;
constexpr int64_t RANK1_SPAN_FP16_SMALL_STRIDE_TARGET_BYTES = 128;
constexpr int64_t SMALL_CONTIGUOUS_TARGET_BYTES_PER_CORE = TARGET_BYTES_PER_CORE;
constexpr int64_t SMALL_CONTIGUOUS_MIN_BYTES = 512;
constexpr int64_t SMALL_CONTIGUOUS_MAX_BYTES = 8192;
constexpr int64_t COMPACT_SPAN_TARGET_BYTES_PER_CORE = 2048;
constexpr int64_t COMPACT_SPAN_ELEMENT_LIMIT = 4096;
constexpr int64_t COMPACT_SUFFIX_SMALL_OUTPUT_ELEMENTS = 50000;
constexpr int64_t COMPACT_SUFFIX_SMALL_TARGET_BYTES_PER_CORE = 2048;
constexpr int64_t COMPACT_SUFFIX_STRIDE1_TARGET_BYTES_PER_CORE = 2048;
constexpr int64_t COMPACT_SUFFIX_RANK2_STRIDED_TARGET_BYTES_PER_CORE = TARGET_BYTES_PER_CORE;
constexpr int64_t COMPACT_SUFFIX_RANK2_GROUP_ROWS = 8;
constexpr int64_t COMPACT_SUFFIX_MAX_GATHER_ELEMENTS_B32 = 8192;
constexpr int64_t BLOCK_BYTES = 32;

struct AsStridedCompileInfo {
    uint32_t ubSizePlatform = 0;
    uint32_t maxCoreNum = 0;
};

// =========================================================================
// Utility functions
// =========================================================================

static int64_t CeilDiv(int64_t lhs, int64_t rhs) { return (lhs + rhs - 1) / rhs; }

static int64_t AlignUp(int64_t value, int64_t alignment) { return CeilDiv(value, alignment) * alignment; }

static gert::Shape EnsureNotScalarShape(const gert::Shape& shape)
{
    if (shape.GetDimNum() != 0) {
        return shape;
    }
    gert::Shape scalarShape;
    scalarShape.SetDimNum(1);
    scalarShape.SetDim(0, 1);
    return scalarShape;
}

static ge::graphStatus GetPlatformInfo(gert::TilingContext* context, uint64_t& ubSize, int64_t& coreNum)
{
    fe::PlatFormInfos* platformInfoPtr = context->GetPlatformInfo();
    OP_CHECK_NULL_WITH_CONTEXT(context, platformInfoPtr);
    auto ascendcPlatform = platform_ascendc::PlatformAscendC(platformInfoPtr);
    coreNum = ascendcPlatform.GetCoreNumAiv();
    OP_CHECK_IF(coreNum == 0, OP_LOGE(context, "coreNum is 0"), return ge::GRAPH_FAILED);
    ascendcPlatform.GetCoreMemSize(platform_ascendc::CoreMemType::UB, ubSize);
    OP_CHECK_IF(ubSize == 0, OP_LOGE(context, "ubSize is 0"), return ge::GRAPH_FAILED);
    return ge::GRAPH_SUCCESS;
}

static ge::graphStatus GetWorkspaceSize(gert::TilingContext* context)
{
    auto ascendcPlatform = platform_ascendc::PlatformAscendC(context->GetPlatformInfo());
    size_t* currentWorkspace = context->GetWorkspaceSizes(1);
    OP_CHECK_NULL_WITH_CONTEXT(context, currentWorkspace);
    currentWorkspace[0] = ascendcPlatform.GetLibApiWorkSpaceSize();
    return ge::GRAPH_SUCCESS;
}

template <typename T>
static bool ReadTensorToShape(gert::TilingContext* context, const gert::Tensor* tensor, const char* inputName,
                              gert::Shape& shape)
{
    const T* data = tensor->GetData<T>();
    OP_CHECK_IF(data == nullptr, OP_LOGE(context, "%s data is null", inputName), return false);

    const int64_t elementNum = tensor->GetShapeSize();
    OP_CHECK_IF(elementNum < 0, OP_LOGE(context, "%s shape size is invalid", inputName), return false);
    OP_CHECK_IF(elementNum > DIM_LIMIT, OP_LOGE(context, "%s rank must be <= %ld", inputName, DIM_LIMIT), return false);

    shape.SetDimNum(elementNum);
    for (int64_t i = 0; i < elementNum; ++i) {
        shape.SetDim(i, static_cast<int64_t>(data[i]));
    }
    return true;
}

static bool ReadIntTensorToShape(gert::TilingContext* context, size_t inputIndex, const char* inputName,
                                 gert::Shape& shape)
{
    const gert::Tensor* tensor = context->GetInputTensor(inputIndex);
    OP_CHECK_IF(tensor == nullptr, OP_LOGE(context, "%s tensor is null", inputName), return false);

    const ge::DataType dtype = tensor->GetDataType();
    if (dtype == ge::DT_INT32) {
        return ReadTensorToShape<int32_t>(context, tensor, inputName, shape);
    }
    if (dtype == ge::DT_INT64) {
        return ReadTensorToShape<int64_t>(context, tensor, inputName, shape);
    }

    OP_LOGE(context, "%s only supports int32/int64", inputName);
    return false;
}

template <typename T>
static bool ReadScalarTensor(gert::TilingContext* context, const gert::Tensor* tensor, const char* inputName,
                             int64_t& value)
{
    const T* data = tensor->GetData<T>();
    OP_CHECK_IF(data == nullptr, OP_LOGE(context, "%s data is null", inputName), return false);
    value = static_cast<int64_t>(data[0]);
    return true;
}

static bool ReadOptionalIntScalar(gert::TilingContext* context, size_t inputIndex, const char* inputName,
                                  int64_t& value)
{
    value = 0;
    const gert::Tensor* tensor = context->GetInputTensor(inputIndex);
    if (tensor == nullptr) {
        return true;
    }

    OP_CHECK_IF(tensor->GetShapeSize() != 1, OP_LOGE(context, "%s must be a scalar tensor", inputName), return false);

    const ge::DataType dtype = tensor->GetDataType();
    if (dtype == ge::DT_INT32) {
        return ReadScalarTensor<int32_t>(context, tensor, inputName, value);
    }
    if (dtype == ge::DT_INT64) {
        return ReadScalarTensor<int64_t>(context, tensor, inputName, value);
    }

    OP_LOGE(context, "%s only supports int32/int64", inputName);
    return false;
}

static bool GetRunInfo(gert::TilingContext* context, gert::Shape& outSize, gert::Shape& outStride,
                       int64_t& storageOffset)
{
    OP_CHECK_IF(!ReadIntTensorToShape(context, IN_SIZE, "size", outSize), OP_LOGE(context, "read size failed"),
                return false);
    OP_CHECK_IF(!ReadIntTensorToShape(context, IN_STRIDE, "stride", outStride), OP_LOGE(context, "read stride failed"),
                return false);
    OP_CHECK_IF(!ReadOptionalIntScalar(context, IN_STORAGE_OFFSET, "storage_offset", storageOffset),
                OP_LOGE(context, "read storage_offset failed"), return false);

    OP_CHECK_IF(outSize.GetDimNum() != outStride.GetDimNum(), OP_LOGE(context, "size rank and stride rank must match"),
                return false);
    OP_CHECK_IF(outSize.GetDimNum() == 0, OP_LOGE(context, "size rank must be greater than 0"), return false);
    return true;
}

static bool ValidateInputs(gert::TilingContext* context, const gert::Shape& xShape, const gert::Shape& outSize,
                           const gert::Shape& outStride, int64_t storageOffset, int64_t& totalOutputElements,
                           int64_t& inputElementCount)
{
    OP_CHECK_IF(outSize.GetDimNum() > DIM_LIMIT, OP_LOGE(context, "only rank <= %ld is supported", DIM_LIMIT),
                return false);
    OP_CHECK_IF(storageOffset < 0, OP_LOGE(context, "storage_offset must be non-negative"), return false);

    totalOutputElements = 1;
    for (size_t i = 0; i < static_cast<size_t>(outSize.GetDimNum()); ++i) {
        OP_CHECK_IF(outSize.GetDim(i) < 0, OP_LOGE(context, "size[%zu] must be >= 0", i), return false);
        OP_CHECK_IF(outStride.GetDim(i) < 0, OP_LOGE(context, "stride[%zu] must be >= 0", i), return false);
        totalOutputElements *= outSize.GetDim(i);
    }

    inputElementCount = xShape.GetShapeSize();
    OP_CHECK_IF(inputElementCount < 0, OP_LOGE(context, "invalid input shape"), return false);

    if (totalOutputElements == 0) {
        return true;
    }

    int64_t maxInputOffset = storageOffset;
    for (size_t i = 0; i < static_cast<size_t>(outSize.GetDimNum()); ++i) {
        if (outSize.GetDim(i) == 0) {
            return true;
        }
        maxInputOffset += (outSize.GetDim(i) - 1) * outStride.GetDim(i);
    }
    OP_CHECK_IF(inputElementCount <= 0, OP_LOGE(context, "input must contain at least one element"), return false);
    OP_CHECK_IF(maxInputOffset >= inputElementCount, OP_LOGE(context, "as_strided view exceeds input storage range"),
                return false);
    return true;
}

static uint32_t GetTypeSize(gert::TilingContext* context, ge::DataType dataType)
{
    const int typeSize = ge::GetSizeByDataType(dataType);
    OP_CHECK_IF(typeSize <= 0, OP_LOGE(context, "unsupported dtype size"), return 0);
    return static_cast<uint32_t>(typeSize);
}

// =========================================================================
// IsContiguousView: checks if strides form a C-contiguous compact layout
// =========================================================================
static bool IsContiguousView(const gert::Shape& outSize, const gert::Shape& outStride)
{
    int64_t expectedStride = 1;
    for (int64_t i = outSize.GetDimNum() - 1; i >= 0; --i) {
        if (outSize.GetDim(i) == 1) {
            continue;
        }
        if (outStride.GetDim(i) != expectedStride) {
            return false;
        }
        expectedStride *= outSize.GetDim(i);
    }
    return true;
}

static bool IsSmallAlignedContiguous(int64_t totalOutputElements, int64_t typeSize, int64_t storageOffset)
{
    int64_t totalBytes = totalOutputElements * typeSize;
    int64_t offsetBytes = storageOffset * typeSize;
    return totalBytes >= SMALL_CONTIGUOUS_MIN_BYTES && totalBytes <= SMALL_CONTIGUOUS_MAX_BYTES &&
           totalBytes % BLOCK_BYTES == 0 && offsetBytes % BLOCK_BYTES == 0;
}

static int64_t ComputeInputSpanElements(const gert::Shape& outSize, const gert::Shape& outStride)
{
    int64_t spanElements = 1;
    for (size_t i = 0; i < static_cast<size_t>(outSize.GetDimNum()); ++i) {
        spanElements += (outSize.GetDim(i) - 1) * outStride.GetDim(i);
    }
    return spanElements;
}

static int64_t CompactRawBytes(int64_t rawElements, int64_t typeSize)
{
    const int64_t rawBytes = rawElements * typeSize;
    if (typeSize != 1) {
        return rawBytes;
    }
    return AlignUp(rawBytes, BLOCK_BYTES) + rawElements * static_cast<int64_t>(sizeof(uint16_t));
}

static int64_t CompactPackedBytes(int64_t elements, int64_t typeSize)
{
    const int64_t packedBytes = elements * typeSize;
    if (typeSize != 1) {
        return packedBytes;
    }
    return AlignUp(packedBytes, BLOCK_BYTES) + elements * static_cast<int64_t>(sizeof(uint16_t));
}

static int64_t CompactTileBytes(int64_t elements, int64_t typeSize)
{
    return CompactPackedBytes(elements, typeSize) + elements * static_cast<int64_t>(sizeof(uint32_t));
}

static int64_t MaxCompactTileElements(int64_t availableBytes, int64_t typeSize, int64_t blockElements)
{
    if (availableBytes <= 0) {
        return blockElements;
    }

    int64_t divisor = typeSize + static_cast<int64_t>(sizeof(uint32_t));
    if (typeSize == 1) {
        divisor += static_cast<int64_t>(sizeof(uint16_t));
    }
    int64_t candidate = std::max<int64_t>(blockElements, availableBytes / divisor);
    while (candidate > blockElements && CompactTileBytes(candidate, typeSize) > availableBytes) {
        --candidate;
    }
    return candidate;
}

static bool CanUseCompactSpan(int64_t inputSpanElements, int64_t typeSize, int64_t usableUbBytes, int64_t blockElements)
{
    if (inputSpanElements <= 0 || inputSpanElements > COMPACT_SPAN_ELEMENT_LIMIT) {
        return false;
    }
    const int64_t rawElements = CeilDiv(inputSpanElements, blockElements) * blockElements;
    const int64_t minPackedElements = blockElements;
    return CompactRawBytes(rawElements, typeSize) + CompactTileBytes(minPackedElements, typeSize) <= usableUbBytes;
}

static bool FindCompactSuffix(const gert::Shape& outSize, int64_t inputSpanElements, int64_t typeSize,
                              int64_t usableUbBytes, int64_t blockElements, int64_t minOuterElements,
                              int64_t& suffixStartDim, int64_t& suffixElements, int64_t& suffixOuterElements)
{
    suffixStartDim = outSize.GetDimNum();
    suffixElements = 0;
    suffixOuterElements = 0;

    const int64_t rawElements = CeilDiv(inputSpanElements, blockElements) * blockElements;
    const int64_t rawBytes = CompactRawBytes(rawElements, typeSize);
    if (rawBytes >= usableUbBytes) {
        return false;
    }

    int64_t totalElements = 1;
    for (size_t i = 0; i < static_cast<size_t>(outSize.GetDimNum()); ++i) {
        totalElements *= outSize.GetDim(i);
    }

    int64_t suffix = 1;
    for (int64_t dim = outSize.GetDimNum() - 1; dim >= 0; --dim) {
        suffix *= outSize.GetDim(dim);
        const int64_t outer = totalElements / suffix;
        const int64_t needBytes = rawBytes + CompactTileBytes(suffix, typeSize);
        if (typeSize == 4 && suffix > COMPACT_SUFFIX_MAX_GATHER_ELEMENTS_B32) {
            continue;
        }
        if (needBytes <= usableUbBytes && outer >= minOuterElements) {
            suffixStartDim = dim;
            suffixElements = suffix;
            suffixOuterElements = outer;
        }
    }

    return suffixElements > 0;
}

static bool FindSmallRank2GroupedSuffix(const gert::Shape& outSize, const gert::Shape& outStride,
                                        int64_t totalOutputElements, int64_t inputSpanElements, int64_t typeSize,
                                        int64_t usableUbBytes, int64_t blockElements, int64_t& suffixStartDim,
                                        int64_t& suffixElements, int64_t& suffixOuterElements)
{
    if (outSize.GetDimNum() != 2 || totalOutputElements >= COMPACT_SUFFIX_SMALL_OUTPUT_ELEMENTS) {
        return false;
    }
    if (outStride.GetDim(1) != 1 && typeSize != static_cast<int64_t>(sizeof(int16_t))) {
        return false;
    }

    const int64_t rawElements = CeilDiv(inputSpanElements, blockElements) * blockElements;
    const int64_t rawBytes = CompactRawBytes(rawElements, typeSize);
    const int64_t groupRowsLimit = (outStride.GetDim(1) == 1) ? COMPACT_SUFFIX_RANK2_GROUP_ROWS : 2;
    const int64_t groupRows = std::min<int64_t>(groupRowsLimit, outSize.GetDim(0));
    const int64_t candidateSuffix = groupRows * outSize.GetDim(1);
    const int64_t needBytes = rawBytes + CompactTileBytes(candidateSuffix, typeSize);
    if (candidateSuffix <= 0 || needBytes > usableUbBytes) {
        return false;
    }

    suffixStartDim = 0;
    suffixElements = candidateSuffix;
    suffixOuterElements = CeilDiv(totalOutputElements, suffixElements);
    return suffixOuterElements > 1;
}

// =========================================================================
// ClassifyPath: determine which tiling path to use based on stride pattern
// =========================================================================
static int64_t ClassifyPath(const gert::Shape& outSize, const gert::Shape& outStride, int64_t totalOutputElements,
                            int64_t lastDimStride, int64_t typeSize, int64_t storageOffset, int64_t inputSpanElements,
                            int64_t usableUbBytes, int64_t blockElements, int64_t coreNum, int64_t& suffixStartDim,
                            int64_t& suffixElements, int64_t& suffixOuterElements)
{
    if (IsContiguousView(outSize, outStride)) {
        if (IsSmallAlignedContiguous(totalOutputElements, typeSize, storageOffset)) {
            return AS_STRIDED_TILING_KEY_CONTIGUOUS_SMALL_ALIGNED;
        }
        return AS_STRIDED_TILING_KEY_CONTIGUOUS;
    }

    if (totalOutputElements > SMALL_SPAN_MAX_OUTPUT_ELEMENTS && outSize.GetDimNum() > 1 &&
        CanUseCompactSpan(inputSpanElements, typeSize, usableUbBytes, blockElements)) {
        if (FindSmallRank2GroupedSuffix(outSize, outStride, totalOutputElements, inputSpanElements, typeSize,
                                        usableUbBytes, blockElements, suffixStartDim, suffixElements,
                                        suffixOuterElements)) {
            return AS_STRIDED_TILING_KEY_COMPACT_SUFFIX;
        }

        int64_t minSuffixOuterElements = coreNum;
        if (totalOutputElements < COMPACT_SUFFIX_SMALL_OUTPUT_ELEMENTS) {
            int64_t smallCoreNum = CeilDiv(totalOutputElements * typeSize, COMPACT_SUFFIX_SMALL_TARGET_BYTES_PER_CORE);
            smallCoreNum = std::max<int64_t>(1, std::min(smallCoreNum, coreNum));
            minSuffixOuterElements = std::min(coreNum, std::max<int64_t>(smallCoreNum * 8, smallCoreNum));
            if (outSize.GetDimNum() == 2 && lastDimStride > 1 && typeSize == static_cast<int64_t>(sizeof(uint32_t))) {
                minSuffixOuterElements = std::min(coreNum, smallCoreNum);
            }
        }
        if (FindCompactSuffix(outSize, inputSpanElements, typeSize, usableUbBytes, blockElements,
                              minSuffixOuterElements, suffixStartDim, suffixElements, suffixOuterElements)) {
            return AS_STRIDED_TILING_KEY_COMPACT_SUFFIX;
        }
        return AS_STRIDED_TILING_KEY_COMPACT_SPAN;
    }

    if (lastDimStride == 0) {
        return AS_STRIDED_TILING_KEY_BROADCAST;
    }

    if (lastDimStride == 1) {
        const int64_t dimNum = outSize.GetDimNum();
        const int64_t lastDimSize = outSize.GetDim(dimNum - 1);
        if (dimNum >= 2 && lastDimSize <= ROW_BATCH_MAX_LAST_DIM &&
            totalOutputElements <= ROW_BATCH_MAX_OUTPUT_ELEMENTS && outStride.GetDim(dimNum - 2) >= lastDimSize) {
            return AS_STRIDED_TILING_KEY_STRIDE1_ROW_BATCH;
        }
        return AS_STRIDED_TILING_KEY_STRIDE_1;
    }

    if (outSize.GetDimNum() == 1 && lastDimStride > 1) {
        const int64_t spanElements = (totalOutputElements - 1) * lastDimStride + 1;
        if (spanElements <= RANK1_SPAN_ELEMENT_LIMIT) {
            return AS_STRIDED_TILING_KEY_RANK1_STRIDE_SPAN;
        }
        return AS_STRIDED_TILING_KEY_RANK1_STRIDE;
    }

    if (lastDimStride > 1) {
        const int64_t lastDimSize = outSize.GetDim(outSize.GetDimNum() - 1);
        if (outSize.GetDimNum() > 1 && lastDimSize <= SMALL_SPAN_MAX_LAST_DIM &&
            totalOutputElements <= SMALL_SPAN_MAX_OUTPUT_ELEMENTS) {
            return AS_STRIDED_TILING_KEY_GENERAL_SMALL_SPAN;
        }
        return AS_STRIDED_TILING_KEY_GENERAL_STRIDE;
    }

    return AS_STRIDED_TILING_KEY_SCALAR;
}

// =========================================================================
// ComputeUbElements: determine UB tile size for each path
// =========================================================================
static int64_t ComputeUbElements(int64_t path, int64_t lastDimSize, int64_t lastDimStride, int64_t typeSize,
                                 int64_t usableUbBytes, int64_t blockElements, int64_t inputSpanElements)
{
    int64_t maxUbElements = usableUbBytes / typeSize;

    switch (path) {
        case AS_STRIDED_TILING_KEY_COMPACT_SUFFIX:
        case AS_STRIDED_TILING_KEY_COMPACT_SPAN: {
            const int64_t rawElements = CeilDiv(inputSpanElements, blockElements) * blockElements;
            const int64_t rawBytes = CompactRawBytes(rawElements, typeSize);
            const int64_t availableBytes = std::max<int64_t>(0, usableUbBytes - rawBytes);
            maxUbElements = MaxCompactTileElements(availableBytes, typeSize, blockElements);
            break;
        }

        case AS_STRIDED_TILING_KEY_CONTIGUOUS_SMALL_ALIGNED:
            // Single-buffer path for small aligned copies.
            maxUbElements = usableUbBytes / typeSize;
            break;

        case AS_STRIDED_TILING_KEY_CONTIGUOUS:
            // Use half UB for double-buffering
            maxUbElements = (usableUbBytes / typeSize) / 2;
            break;

        case AS_STRIDED_TILING_KEY_STRIDE1_ROW_BATCH:
            // Two UB buffers: block-aligned input rows plus packed output rows.
            maxUbElements = ((usableUbBytes / typeSize) / 2) / blockElements;
            break;

        case AS_STRIDED_TILING_KEY_STRIDE_1:
            // One row at a time, chunk if row exceeds UB
            maxUbElements = std::min(maxUbElements, lastDimSize);
            break;

        case AS_STRIDED_TILING_KEY_GENERAL_SMALL_SPAN:
        case AS_STRIDED_TILING_KEY_RANK1_STRIDE_SPAN:
            // Two UB buffers: raw contiguous span plus packed output.
            maxUbElements = (usableUbBytes / typeSize) / 2;
            break;

        case AS_STRIDED_TILING_KEY_GENERAL_STRIDE:
        case AS_STRIDED_TILING_KEY_RANK1_STRIDE:
            // blockCount ≤ 4095 for DataCopy stride params
            maxUbElements = std::min(maxUbElements / blockElements, std::min(lastDimSize, MAX_BLOCK_COUNT));
            break;

        case AS_STRIDED_TILING_KEY_BROADCAST:
            // One row — need to hold lastDimSize replicated elements
            maxUbElements = std::min(maxUbElements, lastDimSize);
            break;

        default: // SCALAR
            maxUbElements = std::min(maxUbElements, static_cast<int64_t>(16));
            break;
    }

    if (path == AS_STRIDED_TILING_KEY_RANK1_STRIDE_SPAN) {
        maxUbElements = std::min(maxUbElements, ((maxUbElements - 1) / lastDimStride) + 1);
    }

    // Align to block boundary
    int64_t ubElements = (maxUbElements / blockElements) * blockElements;
    return std::max<int64_t>(ubElements, blockElements);
}

// =========================================================================
// ComputeMultiCore: split work across cores
// =========================================================================
static int64_t ComputeCoreCountByBytes(int64_t splitElements, int64_t typeSize, int64_t coreNum,
                                       int64_t targetBytesPerCore)
{
    if (splitElements <= 0) {
        return 1;
    }
    int64_t targetCores = CeilDiv(splitElements * typeSize, targetBytesPerCore);
    targetCores = std::max<int64_t>(targetCores, 1);
    targetCores = std::min(targetCores, coreNum);
    return std::min(targetCores, splitElements);
}

static void ComputeMultiCore(int64_t path, int64_t axis0Elements, int64_t totalOutputElements, int64_t coreNum,
                             int64_t typeSize, int64_t lastDimStride, int64_t suffixStartDim, int64_t suffixElements,
                             int64_t suffixOuterElements, int64_t& usedCoreNum, int64_t& perCoreElements,
                             int64_t& lastCoreElements)
{
    int64_t splitElements = axis0Elements;
    if (path == AS_STRIDED_TILING_KEY_SCALAR) {
        usedCoreNum = 1;
    } else if (path == AS_STRIDED_TILING_KEY_COMPACT_SPAN) {
        splitElements = totalOutputElements;
        usedCoreNum = ComputeCoreCountByBytes(splitElements, typeSize, coreNum, COMPACT_SPAN_TARGET_BYTES_PER_CORE);
    } else if (path == AS_STRIDED_TILING_KEY_COMPACT_SUFFIX) {
        splitElements = suffixOuterElements;
        if (totalOutputElements < COMPACT_SUFFIX_SMALL_OUTPUT_ELEMENTS) {
            const bool rank2StridedGroupedSuffix = lastDimStride > 1 &&
                                                   typeSize == static_cast<int64_t>(sizeof(int16_t)) &&
                                                   suffixStartDim == 0 && suffixElements > 0 &&
                                                   suffixOuterElements == CeilDiv(axis0Elements, 2);
            const int64_t targetBytesPerCore = rank2StridedGroupedSuffix ?
                                                   COMPACT_SUFFIX_RANK2_STRIDED_TARGET_BYTES_PER_CORE :
                                                   ((lastDimStride == 1) ?
                                                        COMPACT_SUFFIX_STRIDE1_TARGET_BYTES_PER_CORE :
                                                        COMPACT_SUFFIX_SMALL_TARGET_BYTES_PER_CORE);
            usedCoreNum = ComputeCoreCountByBytes(totalOutputElements, typeSize, coreNum, targetBytesPerCore);
            usedCoreNum = std::min(usedCoreNum, splitElements);
        } else {
            usedCoreNum = std::min(coreNum, splitElements);
        }
    } else if (path == AS_STRIDED_TILING_KEY_CONTIGUOUS_SMALL_ALIGNED) {
        splitElements = totalOutputElements;
        usedCoreNum = ComputeCoreCountByBytes(splitElements, typeSize, coreNum, SMALL_CONTIGUOUS_TARGET_BYTES_PER_CORE);
    } else if (path == AS_STRIDED_TILING_KEY_CONTIGUOUS) {
        splitElements = totalOutputElements;
        usedCoreNum = ComputeCoreCountByBytes(splitElements, typeSize, coreNum, TARGET_BYTES_PER_CORE);
    } else if (path == AS_STRIDED_TILING_KEY_RANK1_STRIDE || path == AS_STRIDED_TILING_KEY_RANK1_STRIDE_SPAN) {
        splitElements = totalOutputElements;
        int64_t targetBytesPerCore = RANK1_STRIDE_TARGET_BYTES_PER_CORE;
        if (path == AS_STRIDED_TILING_KEY_RANK1_STRIDE_SPAN && typeSize == 2 &&
            totalOutputElements >= SMALL_ROW_MC_MIN_OUTPUT_ELEMENTS && lastDimStride <= 4) {
            targetBytesPerCore = RANK1_SPAN_FP16_SMALL_STRIDE_TARGET_BYTES;
        }
        usedCoreNum = ComputeCoreCountByBytes(splitElements, typeSize, coreNum, targetBytesPerCore);
    } else if (path == AS_STRIDED_TILING_KEY_STRIDE1_ROW_BATCH || path == AS_STRIDED_TILING_KEY_GENERAL_SMALL_SPAN) {
        if (totalOutputElements >= SMALL_ROW_MC_MIN_OUTPUT_ELEMENTS && axis0Elements >= SMALL_ROW_MC_MIN_AXIS0) {
            usedCoreNum = std::min(coreNum, CeilDiv(axis0Elements, SMALL_ROW_TARGET_ROWS_PER_CORE));
            usedCoreNum = std::min(usedCoreNum, axis0Elements);
        } else {
            usedCoreNum = 1;
        }
    } else {
        usedCoreNum = std::min(coreNum, axis0Elements);
    }
    usedCoreNum = std::max<int64_t>(usedCoreNum, 1);

    perCoreElements = CeilDiv(splitElements, usedCoreNum);
    lastCoreElements = splitElements - (usedCoreNum - 1) * perCoreElements;
}

// =========================================================================
// FillOutputSizeStride: compute linearization strides for output coordinates
// =========================================================================
static void FillOutputSizeStride(const gert::Shape& outSize, AsStridedTilingData* tiling)
{
    int64_t outSizeData[AS_STRIDED_MAX_DIMS] = {0};
    int64_t outSizeStrideData[AS_STRIDED_MAX_DIMS] = {1, 1, 1, 1, 1, 1, 1, 1};
    for (size_t i = 0; i < static_cast<size_t>(outSize.GetDimNum()); ++i) {
        outSizeData[i] = outSize.GetDim(i);
    }
    if (outSize.GetDimNum() > 0) {
        outSizeStrideData[outSize.GetDimNum() - 1] = 1;
        for (int64_t i = outSize.GetDimNum() - 2; i >= 0; --i) {
            outSizeStrideData[i] = outSizeStrideData[i + 1] * outSizeData[i + 1];
        }
    }
    for (int64_t i = 0; i < AS_STRIDED_MAX_DIMS; ++i) {
        tiling->outSize[i] = outSizeData[i];
        tiling->outSizeStride[i] = outSizeStrideData[i];
    }
}

} // namespace

// =========================================================================
// AsStridedTilingFunc — main tiling entry point
// =========================================================================
ge::graphStatus AsStridedTilingFunc(gert::TilingContext* context)
{
    uint64_t ubSize = 0;
    int64_t coreNum = 0;
    OP_CHECK_IF(GetPlatformInfo(context, ubSize, coreNum) != ge::GRAPH_SUCCESS,
                OP_LOGE(context, "GetPlatformInfo failed"), return ge::GRAPH_FAILED);
    OP_CHECK_IF(GetWorkspaceSize(context) != ge::GRAPH_SUCCESS, OP_LOGE(context, "GetWorkspaceSize failed"),
                return ge::GRAPH_FAILED);

    auto xShapePtr = context->GetInputShape(IN_X);
    OP_CHECK_NULL_WITH_CONTEXT(context, xShapePtr);
    const gert::Shape xShape = EnsureNotScalarShape(xShapePtr->GetStorageShape());

    auto xDesc = context->GetInputDesc(IN_X);
    OP_CHECK_NULL_WITH_CONTEXT(context, xDesc);
    const ge::DataType dataType = xDesc->GetDataType();
    const uint32_t typeSize = GetTypeSize(context, dataType);
    OP_CHECK_IF(typeSize == 0, OP_LOGE(context, "failed to get dtype size"), return ge::GRAPH_FAILED);

    gert::Shape outSize;
    gert::Shape outStride;
    int64_t storageOffset = 0;
    OP_CHECK_IF(!GetRunInfo(context, outSize, outStride, storageOffset), OP_LOGE(context, "GetRunInfo failed"),
                return ge::GRAPH_FAILED);

    int64_t totalOutputElements = 0;
    int64_t inputElementCount = 0;
    OP_CHECK_IF(
        !ValidateInputs(context, xShape, outSize, outStride, storageOffset, totalOutputElements, inputElementCount),
        OP_LOGE(context, "ValidateInputs failed"), return ge::GRAPH_FAILED);

    AsStridedTilingData* tiling = context->GetTilingData<AsStridedTilingData>();
    OP_CHECK_NULL_WITH_CONTEXT(context, tiling);
    *tiling = AsStridedTilingData();

    // --- Extract key dimensional info ---
    int64_t dimNum = outSize.GetDimNum();
    int64_t lastDimSize = (dimNum > 0) ? outSize.GetDim(dimNum - 1) : 1;
    int64_t lastDimStride = (dimNum > 0) ? outStride.GetDim(dimNum - 1) : 1;
    int64_t axis0Elements = (lastDimSize > 0) ? (totalOutputElements / lastDimSize) : 0;
    if (axis0Elements == 0 && totalOutputElements > 0) {
        axis0Elements = 1;
    }

    const int64_t blockElements = std::max<int64_t>(1, 32 / static_cast<int64_t>(typeSize));
    int64_t usableUbBytes = static_cast<int64_t>(ubSize) - RESERVED_UB_BYTES;
    if (usableUbBytes <= 0) {
        usableUbBytes = static_cast<int64_t>(ubSize) / 2;
    }

    // --- Empty output fast path ---
    if (totalOutputElements == 0) {
        tiling->tilingKey = AS_STRIDED_TILING_KEY_SCALAR;
        tiling->outputDimNum = dimNum;
        tiling->lastDimSize = lastDimSize;
        tiling->lastDimStride = lastDimStride;
        tiling->axis0Elements = 0;
        tiling->ubElements = 1;
        tiling->blockElements = blockElements;
        tiling->inputSpanElements = 0;
        tiling->suffixStartDim = dimNum;
        tiling->suffixElements = 0;
        tiling->suffixOuterElements = 0;
        tiling->usedCoreNum = 1;
        tiling->perCoreElements = 0;
        tiling->lastCoreElements = 0;
        context->SetBlockDim(1);
        context->SetTilingKey(0);
        FillOutputSizeStride(outSize, tiling);
        return ge::GRAPH_SUCCESS;
    }

    // --- Path classification ---
    const int64_t inputSpanElements = ComputeInputSpanElements(outSize, outStride);
    int64_t suffixStartDim = dimNum;
    int64_t suffixElements = 0;
    int64_t suffixOuterElements = 0;
    int64_t executionPath = ClassifyPath(
        outSize, outStride, totalOutputElements, lastDimStride, static_cast<int64_t>(typeSize), storageOffset,
        inputSpanElements, usableUbBytes, blockElements, coreNum, suffixStartDim, suffixElements, suffixOuterElements);

    // --- UB tile sizing ---
    int64_t ubElements = ComputeUbElements(executionPath, lastDimSize, lastDimStride, static_cast<int64_t>(typeSize),
                                           usableUbBytes, blockElements, inputSpanElements);
    if (executionPath == AS_STRIDED_TILING_KEY_COMPACT_SUFFIX) {
        ubElements = suffixElements;
    }

    // --- Multi-core split ---
    int64_t usedCoreNum = 0, perCoreElements = 0, lastCoreElements = 0;
    ComputeMultiCore(executionPath, axis0Elements, totalOutputElements, coreNum, static_cast<int64_t>(typeSize),
                     lastDimStride, suffixStartDim, suffixElements, suffixOuterElements, usedCoreNum, perCoreElements,
                     lastCoreElements);

    // --- Fill tiling data ---
    tiling->tilingKey = executionPath;
    tiling->totalOutputElements = totalOutputElements;
    tiling->inputElementCount = inputElementCount;
    tiling->perCoreElements = perCoreElements;
    tiling->lastCoreElements = lastCoreElements;
    tiling->ubElements = ubElements;
    tiling->inputSpanElements = inputSpanElements;
    tiling->suffixStartDim = suffixStartDim;
    tiling->suffixElements = suffixElements;
    tiling->suffixOuterElements = suffixOuterElements;
    tiling->storageOffset = storageOffset;
    tiling->outputDimNum = dimNum;
    tiling->lastDimSize = lastDimSize;
    tiling->lastDimStride = lastDimStride;
    tiling->axis0Elements = axis0Elements;
    tiling->usedCoreNum = usedCoreNum;
    tiling->blockElements = blockElements;

    for (size_t i = 0; i < static_cast<size_t>(outStride.GetDimNum()); ++i) {
        tiling->outStride[i] = outStride.GetDim(i);
    }
    FillOutputSizeStride(outSize, tiling);

    context->SetBlockDim(static_cast<uint32_t>(usedCoreNum));
    context->SetTilingKey(0);
    return ge::GRAPH_SUCCESS;
}

// =========================================================================
// TilingParseForAsStrided — compile-time callback
// =========================================================================
static ge::graphStatus TilingParseForAsStrided(gert::TilingParseContext* context)
{
    auto compileInfo = context->GetCompiledInfo<AsStridedCompileInfo>();
    OP_CHECK_NULL_WITH_CONTEXT(context, compileInfo);
    auto platformInfo = context->GetPlatformInfo();
    OP_CHECK_NULL_WITH_CONTEXT(context, platformInfo);

    auto ascendcPlatform = platform_ascendc::PlatformAscendC(platformInfo);
    compileInfo->maxCoreNum = ascendcPlatform.GetCoreNumAiv();
    OP_CHECK_IF(compileInfo->maxCoreNum == 0, OP_LOGE(context, "maxCoreNum is 0"), return ge::GRAPH_FAILED);

    uint64_t ubSize = 0;
    ascendcPlatform.GetCoreMemSize(platform_ascendc::CoreMemType::UB, ubSize);
    compileInfo->ubSizePlatform = static_cast<uint32_t>(ubSize);
    OP_CHECK_IF(compileInfo->ubSizePlatform == 0, OP_LOGE(context, "ubSizePlatform is 0"), return ge::GRAPH_FAILED);
    return ge::GRAPH_SUCCESS;
}

IMPL_OP_OPTILING(AsStrided)
    .Tiling(AsStridedTilingFunc)
    .TilingParse<AsStridedCompileInfo>(TilingParseForAsStrided)
    .TilingInputsDataDependency({IN_SIZE, IN_STRIDE, IN_STORAGE_OFFSET});

} // namespace optiling
