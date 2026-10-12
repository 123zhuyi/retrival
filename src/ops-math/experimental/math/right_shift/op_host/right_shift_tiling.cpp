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
 * \file right_shift_tiling.cpp
 * \brief RightShift tiling
 */

#include <algorithm>
#include <array>
#include <cstdint>
#include <map>
#include <set>
#include "securec.h"

#include "log/log.h"
#include "util/math_util.h"
#include "register/op_impl_registry.h"
#include "tiling/platform/platform_ascendc.h"
#include "../op_kernel/right_shift_tiling_data.h"
#include "../op_kernel/right_shift_tiling_key.h"

namespace optiling {

namespace {
constexpr uint32_t TILING_KEY_BASE_OFFSET = 1;
constexpr uint32_t BROADCAST_STATE_SAME = 0;
constexpr uint32_t BROADCAST_STATE_X = 1;
constexpr uint32_t BROADCAST_STATE_Y = 2;
constexpr uint32_t INVALID_BROADCAST_STATE = UINT32_MAX;
constexpr uint32_t EMPTY_DIM_NUM = 0;
constexpr uint32_t SCALAR_RANK = 1;
constexpr uint32_t FIRST_DIM_INDEX = 0;
constexpr uint32_t PREVIOUS_DIM_OFFSET = 1;
constexpr uint64_t DIM_DEFAULT_VALUE = 1;
constexpr uint64_t BROADCAST_STRIDE_VALUE = 0;
constexpr uint64_t EMPTY_SHAPE_SIZE = 0;
constexpr int64_t INVALID_DIM_THRESHOLD = 0;
constexpr uint32_t NO_TAIL_CORE_NUM = 0;
constexpr uint64_t EMPTY_TILING_LENGTH = 0;
constexpr int32_t MEMSET_VALUE_ZERO = 0;

const std::set<ge::DataType> SUPPORTED_DTYPE = {
    ge::DT_INT8, ge::DT_UINT8, ge::DT_INT16, ge::DT_UINT16, ge::DT_INT32, ge::DT_UINT32, ge::DT_INT64, ge::DT_UINT64,
};

const std::map<ge::DataType, uint32_t> DTYPE_TPL = {
    {ge::DT_INT8, RIGHT_SHIFT_TPL_INT8 - TILING_KEY_BASE_OFFSET},
    {ge::DT_UINT8, RIGHT_SHIFT_TPL_UINT8 - TILING_KEY_BASE_OFFSET},
    {ge::DT_INT16, RIGHT_SHIFT_TPL_INT16 - TILING_KEY_BASE_OFFSET},
    {ge::DT_UINT16, RIGHT_SHIFT_TPL_UINT16 - TILING_KEY_BASE_OFFSET},
    {ge::DT_INT32, RIGHT_SHIFT_TPL_INT32 - TILING_KEY_BASE_OFFSET},
    {ge::DT_UINT32, RIGHT_SHIFT_TPL_UINT32 - TILING_KEY_BASE_OFFSET},
    {ge::DT_INT64, RIGHT_SHIFT_TPL_INT64 - TILING_KEY_BASE_OFFSET},
    {ge::DT_UINT64, RIGHT_SHIFT_TPL_UINT64 - TILING_KEY_BASE_OFFSET},
};

const std::map<ge::DataType, uint32_t> DTYPE_SIZE = {
    {ge::DT_INT8, sizeof(int8_t)},     {ge::DT_UINT8, sizeof(uint8_t)},   {ge::DT_INT16, sizeof(int16_t)},
    {ge::DT_UINT16, sizeof(uint16_t)}, {ge::DT_INT32, sizeof(int32_t)},   {ge::DT_UINT32, sizeof(uint32_t)},
    {ge::DT_INT64, sizeof(int64_t)},   {ge::DT_UINT64, sizeof(uint64_t)},
};

constexpr uint32_t INPUT_X_INDEX = 0;
constexpr uint32_t INPUT_Y_INDEX = 1;
constexpr uint32_t OUTPUT_Z_INDEX = 0;
constexpr uint32_t WORKSPACE_COUNT = 1;
constexpr uint32_t BYTE_ALIGN = 256;
constexpr uint32_t RESERVED_UB_SIZE = 8 * 1024;
constexpr uint32_t MASK_BUFFER_BYTES = 128;
constexpr uint32_t SCALAR_BUFFER_BYTES = 32;
constexpr uint64_t RIGHT_SHIFT_TMP_BUFFER_FACTOR = 5ULL;
constexpr uint64_t MAX_TILE_BUFFER_LEN = 8192;
constexpr uint64_t MULTI_CORE_SIZE_LIMIT = 2048;
constexpr uint64_t SMALL_BROADCAST_MULTI_CORE_THRESHOLD = 512;
constexpr uint64_t SMALL_BROADCAST_ELEMENTS_PER_CORE = 32;

struct RightShiftCompileInfo {};

struct BroadcastInfo {
    uint64_t totalLength = DIM_DEFAULT_VALUE;
    uint32_t rank = SCALAR_RANK;
    uint32_t mode = RIGHT_SHIFT_MODE_GENERAL;
    std::array<uint64_t, RIGHT_SHIFT_MAX_BROADCAST_DIM> outShape{};
    std::array<uint64_t, RIGHT_SHIFT_MAX_BROADCAST_DIM> xStride{};
    std::array<uint64_t, RIGHT_SHIFT_MAX_BROADCAST_DIM> yStride{};
};

uint64_t CeilDiv(uint64_t value, uint64_t divisor)
{
    uint64_t nonZeroDivisor = divisor == EMPTY_SHAPE_SIZE ? DIM_DEFAULT_VALUE : divisor;
    uint64_t ceilValue = (value + nonZeroDivisor - DIM_DEFAULT_VALUE) / nonZeroDivisor;
    return divisor == EMPTY_SHAPE_SIZE ? EMPTY_SHAPE_SIZE : ceilValue;
}

bool ReadAlignedShape(const gert::Shape& shape, uint32_t rank,
                      std::array<uint64_t, RIGHT_SHIFT_MAX_BROADCAST_DIM>& dims)
{
    if (rank == EMPTY_DIM_NUM || rank > RIGHT_SHIFT_MAX_BROADCAST_DIM) {
        return false;
    }

    if (shape.GetDimNum() > rank) {
        return false;
    }

    dims.fill(DIM_DEFAULT_VALUE);
    uint32_t dimOffset = rank - static_cast<uint32_t>(shape.GetDimNum());
    for (size_t i = 0; i < shape.GetDimNum(); ++i) {
        int64_t dim = shape.GetDim(i);
        if (dim < INVALID_DIM_THRESHOLD) {
            return false;
        }
        dims[dimOffset + i] = static_cast<uint64_t>(dim);
    }
    return true;
}

bool IsShapeEqualToDims(const gert::Shape& shape, const std::array<uint64_t, RIGHT_SHIFT_MAX_BROADCAST_DIM>& dims,
                        uint32_t rank)
{
    // Scalar output shape may be represented as dimNum = 0.
    if (shape.GetDimNum() == EMPTY_DIM_NUM && rank == SCALAR_RANK && dims[FIRST_DIM_INDEX] == DIM_DEFAULT_VALUE) {
        return true;
    }

    if (shape.GetDimNum() != rank) {
        return false;
    }
    for (uint32_t i = 0; i < rank; ++i) {
        int64_t dim = shape.GetDim(i);
        if (dim < INVALID_DIM_THRESHOLD || static_cast<uint64_t>(dim) != dims[i]) {
            return false;
        }
    }
    return true;
}

uint64_t CalcShapeSize(const gert::Shape& shape)
{
    uint64_t shapeSize = DIM_DEFAULT_VALUE;
    for (size_t i = 0; i < shape.GetDimNum(); ++i) {
        int64_t dim = shape.GetDim(i);
        if (dim <= INVALID_DIM_THRESHOLD) {
            return EMPTY_SHAPE_SIZE;
        }
        shapeSize *= static_cast<uint64_t>(dim);
    }
    return shapeSize;
}

void CalcCompressedStride(const std::array<uint64_t, RIGHT_SHIFT_MAX_BROADCAST_DIM>& inputShape,
                          const std::array<uint64_t, RIGHT_SHIFT_MAX_BROADCAST_DIM>& outShape, uint32_t rank,
                          std::array<uint64_t, RIGHT_SHIFT_MAX_BROADCAST_DIM>& stride)
{
    stride.fill(BROADCAST_STRIDE_VALUE);
    uint64_t runningStride = DIM_DEFAULT_VALUE;
    for (int32_t i = static_cast<int32_t>(rank) - static_cast<int32_t>(PREVIOUS_DIM_OFFSET);
         i >= static_cast<int32_t>(FIRST_DIM_INDEX); --i) {
        uint32_t idx = static_cast<uint32_t>(i);
        stride[idx] = runningStride;
        runningStride *= inputShape[idx];
        if (inputShape[idx] == DIM_DEFAULT_VALUE && outShape[idx] != DIM_DEFAULT_VALUE) {
            stride[idx] = BROADCAST_STRIDE_VALUE;
        }
    }
}

uint32_t GetBroadcastState(uint64_t xDim, uint64_t yDim)
{
    if (xDim == yDim) {
        return BROADCAST_STATE_SAME;
    }
    return xDim == DIM_DEFAULT_VALUE ? BROADCAST_STATE_X : BROADCAST_STATE_Y;
}

void CompressBroadcastInfo(const BroadcastInfo& rawInfo, BroadcastInfo& compressedInfo,
                           const std::array<uint64_t, RIGHT_SHIFT_MAX_BROADCAST_DIM>& xDims,
                           const std::array<uint64_t, RIGHT_SHIFT_MAX_BROADCAST_DIM>& yDims)
{
    compressedInfo.totalLength = rawInfo.totalLength;
    compressedInfo.mode = rawInfo.mode;
    compressedInfo.rank = EMPTY_DIM_NUM;
    compressedInfo.outShape.fill(EMPTY_DIM_NUM);
    compressedInfo.xStride.fill(BROADCAST_STRIDE_VALUE);
    compressedInfo.yStride.fill(BROADCAST_STRIDE_VALUE);

    std::array<uint64_t, RIGHT_SHIFT_MAX_BROADCAST_DIM> compressedXDims{};
    std::array<uint64_t, RIGHT_SHIFT_MAX_BROADCAST_DIM> compressedYDims{};
    uint32_t lastState = INVALID_BROADCAST_STATE;
    for (uint32_t i = 0; i < rawInfo.rank; ++i) {
        if (rawInfo.rank > SCALAR_RANK && rawInfo.outShape[i] == DIM_DEFAULT_VALUE) {
            continue;
        }

        uint32_t state = GetBroadcastState(xDims[i], yDims[i]);
        if (compressedInfo.rank > EMPTY_DIM_NUM && state == lastState) {
            uint32_t dst = compressedInfo.rank - PREVIOUS_DIM_OFFSET;
            compressedInfo.outShape[dst] *= rawInfo.outShape[i];
            compressedXDims[dst] *= xDims[i];
            compressedYDims[dst] *= yDims[i];
            continue;
        }

        uint32_t dst = compressedInfo.rank++;
        compressedInfo.outShape[dst] = rawInfo.outShape[i];
        compressedXDims[dst] = xDims[i];
        compressedYDims[dst] = yDims[i];
        lastState = state;
    }

    if (rawInfo.rank > EMPTY_DIM_NUM && compressedInfo.rank == EMPTY_DIM_NUM) {
        compressedInfo.rank = SCALAR_RANK;
        compressedInfo.outShape[FIRST_DIM_INDEX] = DIM_DEFAULT_VALUE;
        compressedXDims[FIRST_DIM_INDEX] = DIM_DEFAULT_VALUE;
        compressedYDims[FIRST_DIM_INDEX] = DIM_DEFAULT_VALUE;
    }

    CalcCompressedStride(compressedXDims, compressedInfo.outShape, compressedInfo.rank, compressedInfo.xStride);
    CalcCompressedStride(compressedYDims, compressedInfo.outShape, compressedInfo.rank, compressedInfo.yStride);
}

ge::graphStatus GetInputType(gert::TilingContext* context, ge::DataType& dtype)
{
    auto xDesc = context->GetInputDesc(INPUT_X_INDEX);
    OP_CHECK_NULL_WITH_CONTEXT(context, xDesc);
    auto yDesc = context->GetInputDesc(INPUT_Y_INDEX);
    OP_CHECK_NULL_WITH_CONTEXT(context, yDesc);
    dtype = xDesc->GetDataType();
    OP_CHECK_IF(dtype != yDesc->GetDataType(), OP_LOGE(context, "x and y should have same dtype in RightShift kernel."),
                return ge::GRAPH_FAILED);
    OP_CHECK_IF(SUPPORTED_DTYPE.count(dtype) == EMPTY_DIM_NUM, OP_LOGE(context, "RightShift dtype is not supported."),
                return ge::GRAPH_FAILED);
    return ge::GRAPH_SUCCESS;
}

ge::graphStatus BuildRawBroadcastInfo(gert::TilingContext* context, const gert::Shape& xStorageShape,
                                      const gert::Shape& yStorageShape, const gert::Shape& zStorageShape,
                                      std::array<uint64_t, RIGHT_SHIFT_MAX_BROADCAST_DIM>& xDims,
                                      std::array<uint64_t, RIGHT_SHIFT_MAX_BROADCAST_DIM>& yDims,
                                      BroadcastInfo& rawInfo)
{
    uint32_t rank = static_cast<uint32_t>(std::max(xStorageShape.GetDimNum(), yStorageShape.GetDimNum()));
    rank = rank == EMPTY_DIM_NUM ? SCALAR_RANK : rank;
    OP_CHECK_IF(rank > RIGHT_SHIFT_MAX_BROADCAST_DIM, OP_LOGE(context, "RightShift rank should be <= 8."),
                return ge::GRAPH_FAILED);

    OP_CHECK_IF(!ReadAlignedShape(xStorageShape, rank, xDims) || !ReadAlignedShape(yStorageShape, rank, yDims),
                OP_LOGE(context, "RightShift input shape has invalid dim."), return ge::GRAPH_FAILED);

    rawInfo.rank = rank;
    rawInfo.totalLength = DIM_DEFAULT_VALUE;
    rawInfo.outShape.fill(DIM_DEFAULT_VALUE);
    for (uint32_t i = 0; i < rank; ++i) {
        uint64_t xDim = xDims[i];
        uint64_t yDim = yDims[i];
        OP_CHECK_IF(xDim != yDim && xDim != DIM_DEFAULT_VALUE && yDim != DIM_DEFAULT_VALUE,
                    OP_LOGE(context, "RightShift input shapes are not broadcastable."), return ge::GRAPH_FAILED);

        uint64_t outDim = xDim == yDim ? xDim : (xDim == DIM_DEFAULT_VALUE ? yDim : xDim);
        rawInfo.outShape[i] = outDim;
        rawInfo.totalLength *= outDim;
    }

    OP_CHECK_IF(!IsShapeEqualToDims(zStorageShape, rawInfo.outShape, rank),
                OP_LOGE(context, "RightShift output shape should equal broadcast shape."), return ge::GRAPH_FAILED);

    return ge::GRAPH_SUCCESS;
}

void SetBroadcastMode(const gert::Shape& xStorageShape, const gert::Shape& yStorageShape, BroadcastInfo& info)
{
    uint64_t xElementNum = CalcShapeSize(xStorageShape);
    uint64_t yElementNum = CalcShapeSize(yStorageShape);
    if (xElementNum == info.totalLength && yElementNum == info.totalLength) {
        info.mode = RIGHT_SHIFT_MODE_CONTIGUOUS;
    } else if (xElementNum == DIM_DEFAULT_VALUE && yElementNum == info.totalLength) {
        info.mode = RIGHT_SHIFT_MODE_X_SCALAR;
    } else if (yElementNum == DIM_DEFAULT_VALUE && xElementNum == info.totalLength) {
        info.mode = RIGHT_SHIFT_MODE_Y_SCALAR;
    } else if (info.rank > EMPTY_DIM_NUM && info.outShape[info.rank - PREVIOUS_DIM_OFFSET] > DIM_DEFAULT_VALUE &&
               info.xStride[info.rank - PREVIOUS_DIM_OFFSET] == DIM_DEFAULT_VALUE &&
               info.yStride[info.rank - PREVIOUS_DIM_OFFSET] == DIM_DEFAULT_VALUE) {
        info.mode = RIGHT_SHIFT_MODE_TAIL_CONTIGUOUS;
    } else {
        info.mode = RIGHT_SHIFT_MODE_GENERAL;
    }
}

ge::graphStatus GetBroadcastInfo(gert::TilingContext* context, BroadcastInfo& info)
{
    auto xShapeInfo = context->GetInputShape(INPUT_X_INDEX);
    OP_CHECK_NULL_WITH_CONTEXT(context, xShapeInfo);
    auto yShapeInfo = context->GetInputShape(INPUT_Y_INDEX);
    OP_CHECK_NULL_WITH_CONTEXT(context, yShapeInfo);
    auto zShapeInfo = context->GetOutputShape(OUTPUT_Z_INDEX);
    OP_CHECK_NULL_WITH_CONTEXT(context, zShapeInfo);

    const auto& xStorageShape = xShapeInfo->GetStorageShape();
    const auto& yStorageShape = yShapeInfo->GetStorageShape();
    const auto& zStorageShape = zShapeInfo->GetStorageShape();
    std::array<uint64_t, RIGHT_SHIFT_MAX_BROADCAST_DIM> xDims{};
    std::array<uint64_t, RIGHT_SHIFT_MAX_BROADCAST_DIM> yDims{};
    BroadcastInfo rawInfo{};
    OP_CHECK_IF(BuildRawBroadcastInfo(context, xStorageShape, yStorageShape, zStorageShape, xDims, yDims, rawInfo) !=
                    ge::GRAPH_SUCCESS,
                OP_LOGE(context, "BuildRawBroadcastInfo failed."), return ge::GRAPH_FAILED);

    CompressBroadcastInfo(rawInfo, info, xDims, yDims);
    SetBroadcastMode(xStorageShape, yStorageShape, info);
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
    OP_CHECK_IF(ubSize == EMPTY_TILING_LENGTH, OP_LOGE(context, "ubSize is 0"), return ge::GRAPH_FAILED);
    OP_CHECK_IF(coreNum == EMPTY_TILING_LENGTH, OP_LOGE(context, "coreNum is 0"), return ge::GRAPH_FAILED);
    return ge::GRAPH_SUCCESS;
}

ge::graphStatus SetWorkspaceSize(gert::TilingContext* context)
{
    auto ascendcPlatform = platform_ascendc::PlatformAscendC(context->GetPlatformInfo());
    size_t* currentWorkspace = context->GetWorkspaceSizes(WORKSPACE_COUNT);
    OP_CHECK_NULL_WITH_CONTEXT(context, currentWorkspace);
    currentWorkspace[FIRST_DIM_INDEX] = ascendcPlatform.GetLibApiWorkSpaceSize();
    return ge::GRAPH_SUCCESS;
}

uint64_t AlignDown(uint64_t value, uint64_t align)
{
    uint64_t nonZeroAlign = align == EMPTY_TILING_LENGTH ? DIM_DEFAULT_VALUE : align;
    uint64_t alignValue = value / nonZeroAlign * nonZeroAlign;
    return align == EMPTY_TILING_LENGTH ? value : alignValue;
}

uint64_t CalcTileBufferLen(uint64_t ubSize, uint32_t dtypeSize)
{
    uint64_t nonZeroDtypeSize = dtypeSize == EMPTY_TILING_LENGTH ? DIM_DEFAULT_VALUE : dtypeSize;
    uint64_t reservedBytes = MASK_BUFFER_BYTES + SCALAR_BUFFER_BYTES;
    uint64_t usableUbSize = ubSize > reservedBytes ? ubSize - reservedBytes : EMPTY_TILING_LENGTH;
    uint64_t bytesPerElement = RIGHT_SHIFT_TMP_BUFFER_FACTOR * nonZeroDtypeSize + sizeof(int32_t);
    uint64_t nonZeroBytesPerElement = bytesPerElement == EMPTY_TILING_LENGTH ? DIM_DEFAULT_VALUE : bytesPerElement;
    uint64_t tileBufferLen = usableUbSize / nonZeroBytesPerElement;
    uint64_t elementAlign = std::max<uint64_t>(BYTE_ALIGN, BYTE_ALIGN / nonZeroDtypeSize);
    tileBufferLen = AlignDown(tileBufferLen, elementAlign);
    tileBufferLen = std::min(tileBufferLen, MAX_TILE_BUFFER_LEN);
    uint64_t bufferLen = tileBufferLen == EMPTY_TILING_LENGTH ? elementAlign : tileBufferLen;
    return dtypeSize == EMPTY_TILING_LENGTH ? EMPTY_TILING_LENGTH : bufferLen;
}

uint64_t CalcUsedCoreNum(uint64_t totalLength, uint64_t coreNum, uint32_t mode)
{
    if (totalLength == EMPTY_TILING_LENGTH) {
        return SCALAR_RANK;
    }

    uint64_t effectiveCoreNum = coreNum == EMPTY_TILING_LENGTH ? SCALAR_RANK : coreNum;
    if (mode != RIGHT_SHIFT_MODE_CONTIGUOUS && totalLength >= SMALL_BROADCAST_MULTI_CORE_THRESHOLD) {
        return std::min(effectiveCoreNum, CeilDiv(totalLength, SMALL_BROADCAST_ELEMENTS_PER_CORE));
    }

    if (totalLength <= MULTI_CORE_SIZE_LIMIT) {
        return SCALAR_RANK;
    }
    return std::min(effectiveCoreNum, totalLength);
}

void SetCoreTiling(uint64_t totalLength, uint64_t coreNum, uint64_t tileBufferLen, uint32_t mode,
                   RightShiftTilingData& tiling)
{
    if (totalLength == EMPTY_TILING_LENGTH) {
        tiling.formerCoreNum = SCALAR_RANK;
        tiling.tailCoreNum = NO_TAIL_CORE_NUM;
        tiling.formerCoreDataNum = EMPTY_TILING_LENGTH;
        tiling.tailCoreDataNum = EMPTY_TILING_LENGTH;
        tiling.tileBufferLen = tileBufferLen;
        return;
    }

    uint64_t usedCoreNum = CalcUsedCoreNum(totalLength, coreNum, mode);
    uint64_t nonZeroUsedCoreNum = usedCoreNum == EMPTY_TILING_LENGTH ? DIM_DEFAULT_VALUE : usedCoreNum;
    uint64_t remainder = totalLength % nonZeroUsedCoreNum;
    tiling.formerCoreNum = remainder == EMPTY_TILING_LENGTH ? nonZeroUsedCoreNum : remainder;
    tiling.tailCoreNum = nonZeroUsedCoreNum - tiling.formerCoreNum;
    tiling.formerCoreDataNum = CeilDiv(totalLength, nonZeroUsedCoreNum);
    tiling.tailCoreDataNum = tiling.tailCoreNum == NO_TAIL_CORE_NUM ? EMPTY_TILING_LENGTH :
                                                                      (totalLength / nonZeroUsedCoreNum);
    tiling.tileBufferLen = tileBufferLen;
}
} // namespace

static ge::graphStatus RightShiftTilingFunc(gert::TilingContext* context)
{
    OP_CHECK_NULL_WITH_CONTEXT(context, context);

    ge::DataType dtype = ge::DT_INT32;
    OP_CHECK_IF(GetInputType(context, dtype) != ge::GRAPH_SUCCESS, OP_LOGE(context, "GetInputType failed."),
                return ge::GRAPH_FAILED);

    BroadcastInfo broadcastInfo{};
    OP_CHECK_IF(GetBroadcastInfo(context, broadcastInfo) != ge::GRAPH_SUCCESS,
                OP_LOGE(context, "GetBroadcastInfo failed."), return ge::GRAPH_FAILED);

    uint64_t ubSize = EMPTY_TILING_LENGTH;
    uint64_t coreNum = SCALAR_RANK;
    OP_CHECK_IF(GetPlatformInfo(context, ubSize, coreNum) != ge::GRAPH_SUCCESS,
                OP_LOGE(context, "GetPlatformInfo failed."), return ge::GRAPH_FAILED);

    auto dtypeSizeIter = DTYPE_SIZE.find(dtype);
    OP_CHECK_IF(dtypeSizeIter == DTYPE_SIZE.end(), OP_LOGE(context, "RightShift dtype size not found."),
                return ge::GRAPH_FAILED);

    RightShiftTilingData* tiling = context->GetTilingData<RightShiftTilingData>();
    OP_CHECK_NULL_WITH_CONTEXT(context, tiling);
    OP_CHECK_IF(memset_s(tiling, sizeof(RightShiftTilingData), MEMSET_VALUE_ZERO, sizeof(RightShiftTilingData)) != EOK,
                OP_LOGE(context, "set tiling data error"), return ge::GRAPH_FAILED);

    uint64_t tileBufferLen = CalcTileBufferLen(ubSize, dtypeSizeIter->second);
    SetCoreTiling(broadcastInfo.totalLength, coreNum, tileBufferLen, broadcastInfo.mode, *tiling);
    tiling->totalLength = broadcastInfo.totalLength;
    tiling->rank = broadcastInfo.rank;
    tiling->mode = broadcastInfo.mode;
    for (uint32_t i = 0; i < RIGHT_SHIFT_MAX_BROADCAST_DIM; ++i) {
        tiling->outShape[i] = broadcastInfo.outShape[i];
        tiling->xStride[i] = broadcastInfo.xStride[i];
        tiling->yStride[i] = broadcastInfo.yStride[i];
    }

    OP_CHECK_IF(SetWorkspaceSize(context) != ge::GRAPH_SUCCESS, OP_LOGE(context, "SetWorkspaceSize failed."),
                return ge::GRAPH_FAILED);

    auto iter = DTYPE_TPL.find(dtype);
    OP_CHECK_IF(iter == DTYPE_TPL.end(), OP_LOGE(context, "SetTilingKey dtype unsupported."), return ge::GRAPH_FAILED);
    uint64_t tilingKey = GET_TPL_TILING_KEY(broadcastInfo.mode * RIGHT_SHIFT_TPL_DTYPE_COUNT + iter->second);
    OP_CHECK_IF(context->SetTilingKey(tilingKey) != ge::GRAPH_SUCCESS, OP_LOGE(context, "SetTilingKey failed."),
                return ge::GRAPH_FAILED);

    context->SetBlockDim(static_cast<uint32_t>(tiling->formerCoreNum + tiling->tailCoreNum));
    return ge::GRAPH_SUCCESS;
}

static ge::graphStatus TilingParseForRightShift(gert::TilingParseContext* context)
{
    OP_CHECK_NULL_WITH_CONTEXT(context, context);
    // RightShift does not need extra compile info currently, so this parse callback only validates context.
    return ge::GRAPH_SUCCESS;
}

IMPL_OP_OPTILING(RightShift).Tiling(RightShiftTilingFunc).TilingParse<RightShiftCompileInfo>(TilingParseForRightShift);
} // namespace optiling
