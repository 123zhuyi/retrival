/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include <algorithm>
#include <array>
#include <cstdint>
#include <map>
#include <set>

#include "log/log.h"
#include "util/math_util.h"
#include "register/op_impl_registry.h"
#include "tiling/platform/platform_ascendc.h"
#include "../op_kernel/is_close_tiling_data.h"
#include "../op_kernel/is_close_tiling_key.h"

namespace optiling {

namespace {
constexpr uint32_t BUFFER_NUM = 2;
constexpr uint32_t BYTE_ALIGN = 256;
constexpr uint32_t COMPARE_ALIGN = 256;
constexpr uint32_t RESERVED_UB_SIZE = 8 * 1024;
constexpr uint32_t FLOAT_BUF_NUM = 8;
constexpr uint32_t HALF_BUF_NUM = 1;
constexpr uint32_t MASK_BUF_NUM = 1;
constexpr uint32_t BROADCAST_SCALAR_BUF_SIZE = 2 * 32;
constexpr uint64_t MULTI_CORE_SIZE_LIMIT = 2048;
constexpr uint64_t SMALL_BROADCAST_MULTI_CORE_THRESHOLD = 512;
constexpr uint64_t SMALL_BROADCAST_ELEMENTS_PER_CORE = 32;
constexpr float DEFAULT_RTOL = 1e-5f;
constexpr float DEFAULT_ATOL = 1e-8f;

const std::set<ge::DataType> SUPPORTED_DTYPE = {ge::DT_FLOAT, ge::DT_FLOAT16, ge::DT_BF16, ge::DT_INT32};
const std::map<ge::DataType, uint32_t> DTYPE_SIZE = {
    {ge::DT_FLOAT, sizeof(float)},
    {ge::DT_FLOAT16, sizeof(uint16_t)},
    {ge::DT_BF16, sizeof(uint16_t)},
    {ge::DT_INT32, sizeof(int32_t)},
};

struct IsCloseCompileInfo {};

struct BroadcastInfo {
    uint64_t totalLength = 1;
    uint32_t rank = 0;
    uint32_t broadcastMode = IS_CLOSE_BROADCAST_MODE_CONTIGUOUS;
    std::array<uint64_t, IS_CLOSE_MAX_BROADCAST_DIM> outShape {};
    std::array<uint64_t, IS_CLOSE_MAX_BROADCAST_DIM> x1Stride {};
    std::array<uint64_t, IS_CLOSE_MAX_BROADCAST_DIM> x2Stride {};
};

bool ReadAlignedShape(
    const gert::Shape& shape, uint32_t rank, std::array<uint64_t, IS_CLOSE_MAX_BROADCAST_DIM>& dims)
{
    if (shape.GetDimNum() > rank || rank > IS_CLOSE_MAX_BROADCAST_DIM) {
        return false;
    }
    dims.fill(1);
    uint32_t dimOffset = rank - static_cast<uint32_t>(shape.GetDimNum());
    for (size_t i = 0; i < shape.GetDimNum(); ++i) {
        int64_t dim = shape.GetDim(i);
        if (dim < 0) {
            return false;
        }
        dims[dimOffset + i] = static_cast<uint64_t>(dim);
    }
    return true;
}

bool IsShapeEqualToDims(
    const gert::Shape& shape, const std::array<uint64_t, IS_CLOSE_MAX_BROADCAST_DIM>& dims, uint32_t rank)
{
    if (shape.GetDimNum() != rank) {
        return false;
    }
    for (uint32_t i = 0; i < rank; ++i) {
        int64_t dim = shape.GetDim(i);
        if (dim < 0 || static_cast<uint64_t>(dim) != dims[i]) {
            return false;
        }
    }
    return true;
}

uint64_t CalcShapeSize(const gert::Shape& shape)
{
    uint64_t shapeSize = 1;
    for (size_t i = 0; i < shape.GetDimNum(); ++i) {
        int64_t dim = shape.GetDim(i);
        if (dim <= 0) {
            return 0;
        }
        shapeSize *= static_cast<uint64_t>(dim);
    }
    return shapeSize;
}

void CalcCompressedStride(
    const std::array<uint64_t, IS_CLOSE_MAX_BROADCAST_DIM>& inputShape,
    const std::array<uint64_t, IS_CLOSE_MAX_BROADCAST_DIM>& outShape, uint32_t rank,
    std::array<uint64_t, IS_CLOSE_MAX_BROADCAST_DIM>& stride)
{
    stride.fill(0);
    uint64_t runningStride = 1;
    for (int32_t i = static_cast<int32_t>(rank) - 1; i >= 0; --i) {
        stride[static_cast<uint32_t>(i)] = runningStride;
        runningStride *= inputShape[static_cast<uint32_t>(i)];
        if (inputShape[static_cast<uint32_t>(i)] == 1 && outShape[static_cast<uint32_t>(i)] != 1) {
            stride[static_cast<uint32_t>(i)] = 0;
        }
    }
}

uint32_t GetBroadcastState(uint64_t x1Dim, uint64_t x2Dim)
{
    if (x1Dim == x2Dim) {
        return 0;
    }
    return x1Dim == 1 ? 1 : 2;
}

void CompressBroadcastInfo(
    const BroadcastInfo& rawInfo, BroadcastInfo& compressedInfo,
    const std::array<uint64_t, IS_CLOSE_MAX_BROADCAST_DIM>& x1Dims,
    const std::array<uint64_t, IS_CLOSE_MAX_BROADCAST_DIM>& x2Dims)
{
    compressedInfo.totalLength = rawInfo.totalLength;
    compressedInfo.broadcastMode = rawInfo.broadcastMode;
    compressedInfo.rank = 0;
    compressedInfo.outShape.fill(0);
    compressedInfo.x1Stride.fill(0);
    compressedInfo.x2Stride.fill(0);

    std::array<uint64_t, IS_CLOSE_MAX_BROADCAST_DIM> compressedX1Dims {};
    std::array<uint64_t, IS_CLOSE_MAX_BROADCAST_DIM> compressedX2Dims {};
    uint32_t lastState = UINT32_MAX;
    for (uint32_t i = 0; i < rawInfo.rank; ++i) {
        if (rawInfo.rank > 1 && rawInfo.outShape[i] == 1) {
            continue;
        }
        uint32_t state = GetBroadcastState(x1Dims[i], x2Dims[i]);
        if (compressedInfo.rank > 0 && state == lastState) {
            uint32_t dst = compressedInfo.rank - 1;
            compressedInfo.outShape[dst] *= rawInfo.outShape[i];
            compressedX1Dims[dst] *= x1Dims[i];
            compressedX2Dims[dst] *= x2Dims[i];
        } else {
            uint32_t dst = compressedInfo.rank++;
            compressedInfo.outShape[dst] = rawInfo.outShape[i];
            compressedX1Dims[dst] = x1Dims[i];
            compressedX2Dims[dst] = x2Dims[i];
            lastState = state;
        }
    }

    if (rawInfo.rank > 0 && compressedInfo.rank == 0) {
        compressedInfo.rank = 1;
        compressedInfo.outShape[0] = 1;
        compressedX1Dims[0] = 1;
        compressedX2Dims[0] = 1;
    }

    CalcCompressedStride(compressedX1Dims, compressedInfo.outShape, compressedInfo.rank, compressedInfo.x1Stride);
    CalcCompressedStride(compressedX2Dims, compressedInfo.outShape, compressedInfo.rank, compressedInfo.x2Stride);
}

ge::graphStatus GetBroadcastInfo(gert::TilingContext* context, BroadcastInfo& info)
{
    auto x1Shape = context->GetInputShape(0);
    OP_CHECK_NULL_WITH_CONTEXT(context, x1Shape);
    auto x2Shape = context->GetInputShape(1);
    OP_CHECK_NULL_WITH_CONTEXT(context, x2Shape);
    auto yShape = context->GetOutputShape(0);
    OP_CHECK_NULL_WITH_CONTEXT(context, yShape);

    const auto& x1StorageShape = x1Shape->GetStorageShape();
    const auto& x2StorageShape = x2Shape->GetStorageShape();
    const auto& yStorageShape = yShape->GetStorageShape();

    uint32_t rank = static_cast<uint32_t>(std::max(x1StorageShape.GetDimNum(), x2StorageShape.GetDimNum()));
    OP_CHECK_IF(
        rank > IS_CLOSE_MAX_BROADCAST_DIM, OP_LOGE(context, "IsClose broadcast rank should be <= 8."),
        return ge::GRAPH_FAILED);

    std::array<uint64_t, IS_CLOSE_MAX_BROADCAST_DIM> x1Dims {};
    std::array<uint64_t, IS_CLOSE_MAX_BROADCAST_DIM> x2Dims {};
    OP_CHECK_IF(
        !ReadAlignedShape(x1StorageShape, rank, x1Dims) || !ReadAlignedShape(x2StorageShape, rank, x2Dims),
        OP_LOGE(context, "IsClose input shape has invalid dimension."), return ge::GRAPH_FAILED);

    BroadcastInfo rawInfo {};
    rawInfo.rank = rank;
    rawInfo.totalLength = 1;
    for (uint32_t i = 0; i < rank; ++i) {
        uint64_t x1Dim = x1Dims[i];
        uint64_t x2Dim = x2Dims[i];
        OP_CHECK_IF(
            x1Dim != x2Dim && x1Dim != 1 && x2Dim != 1,
            OP_LOGE(context, "IsClose x1 and x2 shapes are not broadcastable."), return ge::GRAPH_FAILED);
        rawInfo.outShape[i] = (x1Dim == 0 || x2Dim == 0) ? 0 : std::max(x1Dim, x2Dim);
        rawInfo.totalLength *= rawInfo.outShape[i];
    }

    OP_CHECK_IF(
        !IsShapeEqualToDims(yStorageShape, rawInfo.outShape, rank),
        OP_LOGE(context, "IsClose output shape should equal broadcast shape."), return ge::GRAPH_FAILED);

    CompressBroadcastInfo(rawInfo, info, x1Dims, x2Dims);

    uint64_t x1ElementNum = CalcShapeSize(x1StorageShape);
    uint64_t x2ElementNum = CalcShapeSize(x2StorageShape);
    if (x1ElementNum == info.totalLength && x2ElementNum == info.totalLength) {
        info.broadcastMode = IS_CLOSE_BROADCAST_MODE_CONTIGUOUS;
    } else if (x1ElementNum == 1 && x2ElementNum == info.totalLength) {
        info.broadcastMode = IS_CLOSE_BROADCAST_MODE_X1_SCALAR;
    } else if (x2ElementNum == 1 && x1ElementNum == info.totalLength) {
        info.broadcastMode = IS_CLOSE_BROADCAST_MODE_X2_SCALAR;
    } else if (
        info.rank > 0 && info.outShape[info.rank - 1] > 1 && info.x1Stride[info.rank - 1] == 1 &&
        info.x2Stride[info.rank - 1] == 1) {
        info.broadcastMode = IS_CLOSE_BROADCAST_MODE_TAIL_CONTIGUOUS;
    } else {
        info.broadcastMode = IS_CLOSE_BROADCAST_MODE_GENERAL;
    }
    return ge::GRAPH_SUCCESS;
}

ge::graphStatus GetPlatformInfo(gert::TilingContext* context, uint64_t& ubSize, uint64_t& coreNum)
{
    auto ascendcPlatform = platform_ascendc::PlatformAscendC(context->GetPlatformInfo());
    ascendcPlatform.GetCoreMemSize(platform_ascendc::CoreMemType::UB, ubSize);
    if (ubSize > RESERVED_UB_SIZE) {
        ubSize -= RESERVED_UB_SIZE;
    }
    coreNum = static_cast<uint64_t>(ascendcPlatform.GetCoreNumAiv());
    OP_CHECK_IF(ubSize == 0, OP_LOGE(context, "ubSize is 0"), return ge::GRAPH_FAILED);
    OP_CHECK_IF(coreNum == 0, OP_LOGE(context, "coreNum is 0"), return ge::GRAPH_FAILED);
    return ge::GRAPH_SUCCESS;
}

ge::graphStatus GetInputType(gert::TilingContext* context, ge::DataType& dtype)
{
    auto x1Desc = context->GetInputDesc(0);
    OP_CHECK_NULL_WITH_CONTEXT(context, x1Desc);
    auto x2Desc = context->GetInputDesc(1);
    OP_CHECK_NULL_WITH_CONTEXT(context, x2Desc);
    dtype = x1Desc->GetDataType();
    OP_CHECK_IF(
        dtype != x2Desc->GetDataType(), OP_LOGE(context, "x1 and x2 should have the same dtype."),
        return ge::GRAPH_FAILED);
    OP_CHECK_IF(
        SUPPORTED_DTYPE.count(dtype) == 0, OP_LOGE(context, "IsClose dtype is not supported."),
        return ge::GRAPH_FAILED);
    return ge::GRAPH_SUCCESS;
}

uint32_t GetDtypeTemplateValue(ge::DataType dtype)
{
    if (dtype == ge::DT_FLOAT) {
        return IS_CLOSE_TPL_FP32 - 1;
    }
    if (dtype == ge::DT_FLOAT16) {
        return IS_CLOSE_TPL_FP16 - 1;
    }
    if (dtype == ge::DT_BF16) {
        return IS_CLOSE_TPL_BF16 - 1;
    }
    if (dtype == ge::DT_INT32) {
        return IS_CLOSE_TPL_INT32 - 1;
    }
    return IS_CLOSE_TPL_FP32 - 1;
}

ge::graphStatus GetAttrs(gert::TilingContext* context, float& rtol, float& atol, uint32_t& equalNan)
{
    rtol = DEFAULT_RTOL;
    atol = DEFAULT_ATOL;
    equalNan = 0;

    const gert::RuntimeAttrs* attrs = context->GetAttrs();
    if (attrs != nullptr) {
        const float* rtolPtr = attrs->GetAttrPointer<float>(0);
        const float* atolPtr = attrs->GetAttrPointer<float>(1);
        const bool* equalNanPtr = attrs->GetAttrPointer<bool>(2);
        if (rtolPtr != nullptr) {
            rtol = *rtolPtr;
        }
        if (atolPtr != nullptr) {
            atol = *atolPtr;
        }
        if (equalNanPtr != nullptr) {
            equalNan = *equalNanPtr ? 1U : 0U;
        }
    }

    OP_CHECK_IF(
        rtol < 0.0f || atol < 0.0f, OP_LOGE(context, "rtol and atol should be non-negative."),
        return ge::GRAPH_FAILED);
    return ge::GRAPH_SUCCESS;
}

uint64_t AlignDown(uint64_t value, uint64_t align)
{
    if (align == 0) {
        return value;
    }
    return value / align * align;
}

uint64_t CeilDiv(uint64_t value, uint64_t divisor)
{
    return divisor == 0 ? 0 : (value + divisor - 1) / divisor;
}

uint64_t CalcTileBufferLen(uint64_t ubSize, uint32_t dtypeSize)
{
    uint64_t bytesPerElement =
        BUFFER_NUM * (2ULL * dtypeSize + sizeof(int8_t)) + FLOAT_BUF_NUM * sizeof(float) +
        HALF_BUF_NUM * sizeof(uint16_t) + MASK_BUF_NUM;
    uint64_t usableUbSize = ubSize > BROADCAST_SCALAR_BUF_SIZE ? ubSize - BROADCAST_SCALAR_BUF_SIZE : 0;
    uint64_t tileBufferLen = bytesPerElement == 0 ? 0 : usableUbSize / bytesPerElement;
    uint64_t elementAlign = std::max<uint64_t>(COMPARE_ALIGN, BYTE_ALIGN / dtypeSize);
    tileBufferLen = AlignDown(tileBufferLen, elementAlign);
    return tileBufferLen == 0 ? elementAlign : tileBufferLen;
}

uint64_t CalcUsedCoreNum(uint64_t totalLength, uint64_t coreNum, uint32_t broadcastMode)
{
    if (totalLength == 0) {
        return 1;
    }

    if (broadcastMode != IS_CLOSE_BROADCAST_MODE_CONTIGUOUS &&
        totalLength >= SMALL_BROADCAST_MULTI_CORE_THRESHOLD) {
        return std::min(coreNum, CeilDiv(totalLength, SMALL_BROADCAST_ELEMENTS_PER_CORE));
    }

    if (totalLength <= MULTI_CORE_SIZE_LIMIT) {
        return 1;
    }
    return std::min(coreNum, totalLength);
}

void SetCoreTiling(
    uint64_t totalLength, uint64_t coreNum, uint64_t tileBufferLen, uint32_t broadcastMode, IsCloseTilingData& tiling)
{
    if (totalLength == 0) {
        tiling.formerCoreNum = 1;
        tiling.tileBufferLen = tileBufferLen;
        return;
    }

    uint64_t usedCoreNum = CalcUsedCoreNum(totalLength, coreNum, broadcastMode);
    uint64_t remainder = totalLength % usedCoreNum;
    tiling.formerCoreNum = remainder == 0 ? usedCoreNum : remainder;
    tiling.tailCoreNum = usedCoreNum - tiling.formerCoreNum;
    tiling.formerCoreDataNum = (totalLength + usedCoreNum - 1) / usedCoreNum;
    tiling.tailCoreDataNum = tiling.tailCoreNum == 0 ? 0 : totalLength / usedCoreNum;
    tiling.tileBufferLen = tileBufferLen;

    tiling.formerCoreLoopCount = (tiling.formerCoreDataNum + tileBufferLen - 1) / tileBufferLen;
    tiling.formerCoreFormerDataNum = std::min(tiling.formerCoreDataNum, tileBufferLen);
    tiling.formerCoreTailDataNum =
        tiling.formerCoreDataNum % tileBufferLen == 0 ? tileBufferLen : tiling.formerCoreDataNum % tileBufferLen;

    if (tiling.tailCoreDataNum != 0) {
        tiling.tailCoreLoopCount = (tiling.tailCoreDataNum + tileBufferLen - 1) / tileBufferLen;
        tiling.tailCoreFormerDataNum = std::min(tiling.tailCoreDataNum, tileBufferLen);
        tiling.tailCoreTailDataNum =
            tiling.tailCoreDataNum % tileBufferLen == 0 ? tileBufferLen : tiling.tailCoreDataNum % tileBufferLen;
    }
}

ge::graphStatus SetWorkspaceSize(gert::TilingContext* context)
{
    auto ascendcPlatform = platform_ascendc::PlatformAscendC(context->GetPlatformInfo());
    size_t* currentWorkspace = context->GetWorkspaceSizes(1);
    OP_CHECK_NULL_WITH_CONTEXT(context, currentWorkspace);
    currentWorkspace[0] = ascendcPlatform.GetLibApiWorkSpaceSize();
    return ge::GRAPH_SUCCESS;
}
} // namespace

static ge::graphStatus IsCloseTilingFunc(gert::TilingContext* context)
{
    OP_CHECK_NULL_WITH_CONTEXT(context, context);

    uint64_t ubSize = 0;
    uint64_t coreNum = 0;
    OP_CHECK_IF(
        GetPlatformInfo(context, ubSize, coreNum) != ge::GRAPH_SUCCESS,
        OP_LOGE(context, "GetPlatformInfo failed."), return ge::GRAPH_FAILED);

    BroadcastInfo broadcastInfo {};
    OP_CHECK_IF(
        GetBroadcastInfo(context, broadcastInfo) != ge::GRAPH_SUCCESS,
        OP_LOGE(context, "GetBroadcastInfo failed."), return ge::GRAPH_FAILED);

    ge::DataType dtype = ge::DT_FLOAT;
    OP_CHECK_IF(
        GetInputType(context, dtype) != ge::GRAPH_SUCCESS,
        OP_LOGE(context, "GetInputType failed."), return ge::GRAPH_FAILED);

    auto iter = DTYPE_SIZE.find(dtype);
    OP_CHECK_IF(iter == DTYPE_SIZE.end(), OP_LOGE(context, "dtype size not found."), return ge::GRAPH_FAILED);

    IsCloseTilingData* tiling = context->GetTilingData<IsCloseTilingData>();
    OP_CHECK_NULL_WITH_CONTEXT(context, tiling);
    OP_CHECK_IF(
        memset_s(tiling, sizeof(IsCloseTilingData), 0, sizeof(IsCloseTilingData)) != EOK,
        OP_LOGE(context, "set tiling data error"), return ge::GRAPH_FAILED);

    OP_CHECK_IF(
        GetAttrs(context, tiling->rtol, tiling->atol, tiling->equalNan) != ge::GRAPH_SUCCESS,
        OP_LOGE(context, "GetAttrs failed."), return ge::GRAPH_FAILED);

    uint64_t tileBufferLen = CalcTileBufferLen(ubSize, iter->second);
    SetCoreTiling(broadcastInfo.totalLength, coreNum, tileBufferLen, broadcastInfo.broadcastMode, *tiling);
    tiling->totalLength = broadcastInfo.totalLength;
    tiling->rank = broadcastInfo.rank;
    tiling->broadcastMode = broadcastInfo.broadcastMode;
    for (uint32_t i = 0; i < IS_CLOSE_MAX_BROADCAST_DIM; ++i) {
        tiling->outShape[i] = broadcastInfo.outShape[i];
        tiling->x1Stride[i] = broadcastInfo.x1Stride[i];
        tiling->x2Stride[i] = broadcastInfo.x2Stride[i];
    }
    OP_CHECK_IF(
        SetWorkspaceSize(context) != ge::GRAPH_SUCCESS, OP_LOGE(context, "SetWorkspaceSize failed."),
        return ge::GRAPH_FAILED);

    uint32_t dtypeTemplate = GetDtypeTemplateValue(dtype);
    uint64_t tilingKey = GET_TPL_TILING_KEY(broadcastInfo.broadcastMode * IS_CLOSE_TPL_DTYPE_COUNT + dtypeTemplate);
    OP_CHECK_IF(
        context->SetTilingKey(tilingKey) != ge::GRAPH_SUCCESS,
        OP_LOGE(context, "SetTilingKey failed."), return ge::GRAPH_FAILED);
    context->SetBlockDim(tiling->formerCoreNum + tiling->tailCoreNum);
    return ge::GRAPH_SUCCESS;
}

static ge::graphStatus TilingParseForIsClose([[maybe_unused]] gert::TilingParseContext* context)
{
    return ge::GRAPH_SUCCESS;
}

IMPL_OP_OPTILING(IsClose).Tiling(IsCloseTilingFunc).TilingParse<IsCloseCompileInfo>(TilingParseForIsClose);
} // namespace optiling
