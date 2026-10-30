/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include "log/log.h"
#include "register/op_impl_registry.h"
#include "torch_extension/tiling_utils.h"
#include "tiling/platform/platform_ascendc.h"

#include "../op_kernel/is_pos_inf_tiling_data.h"

namespace optiling {

using namespace ge;

constexpr int64_t CACHE_LINE_BYTE_LENGTH = 512;
constexpr int64_t MAX_DIM_LEN = 8;
constexpr int64_t FLOAT_BUFFER_COEFFICIENT = 10;
constexpr int64_t DEBUG_FP32_TILE_ELEMENTS = 192;
constexpr int64_t DEBUG_BF16_TILE_ELEMENTS = 192;
constexpr size_t IDX_SELF = 0;

static const gert::Shape SCALAR_SHAPE = {1};

enum class IsPosInfDtypeId : int64_t {
    INVALID = -1,
    FP32 = 0,
    FP16 = 1,
    BF16 = 2,
};

static const gert::Shape& EnsureNotScalar(const gert::Shape& shape)
{
    if (shape.GetDimNum() == 0) {
        return SCALAR_SHAPE;
    }
    return shape;
}

static graphStatus GetPlatformInfo(gert::TilingContext* context, int64_t& coreNum, uint64_t& ubSize, uint32_t& wsSysSize)
{
    fe::PlatFormInfos* platformInfo = context->GetPlatformInfo();
    OP_CHECK_NULL_WITH_CONTEXT(context, platformInfo);
    auto ascendcPlatform = platform_ascendc::PlatformAscendC(platformInfo);
    coreNum = ascendcPlatform.GetCoreNumAiv();
    OP_CHECK_IF(coreNum <= 0, OP_LOGE(context, "coreNum is invalid"), return GRAPH_FAILED);
    ascendcPlatform.GetCoreMemSize(platform_ascendc::CoreMemType::UB, ubSize);
    OP_CHECK_IF(ubSize == 0, OP_LOGE(context, "ubSize is invalid"), return GRAPH_FAILED);
    wsSysSize = ascendcPlatform.GetLibApiWorkSpaceSize();
    return GRAPH_SUCCESS;
}

static bool IsSupportedFloatType(DataType dtype)
{
    return dtype == DT_FLOAT || dtype == DT_FLOAT16 || dtype == DT_BF16;
}

static IsPosInfDtypeId GetDtypeId(DataType dtype)
{
    if (dtype == DT_FLOAT) {
        return IsPosInfDtypeId::FP32;
    }
    if (dtype == DT_FLOAT16) {
        return IsPosInfDtypeId::FP16;
    }
    if (dtype == DT_BF16) {
        return IsPosInfDtypeId::BF16;
    }
    return IsPosInfDtypeId::INVALID;
}

static int64_t GetTypeSize(DataType dtype)
{
    switch (dtype) {
        case DT_FLOAT:
            return sizeof(float);
        case DT_FLOAT16:
        case DT_BF16:
            return sizeof(int16_t);
        default:
            return 0;
    }
}

static graphStatus ValidateInput(
    gert::TilingContext* context, int64_t& totalLength, int64_t& dtypeSize, IsPosInfDtypeId& dtypeId)
{
    auto inputShape = context->GetInputShape(IDX_SELF);
    OP_CHECK_NULL_WITH_CONTEXT(context, inputShape);
    auto shape = EnsureNotScalar(inputShape->GetStorageShape());
    OP_CHECK_IF(shape.GetDimNum() > MAX_DIM_LEN, OP_LOGE(context, "dim num exceeds limit"), return GRAPH_FAILED);

    auto inputDesc = context->GetInputDesc(IDX_SELF);
    OP_CHECK_NULL_WITH_CONTEXT(context, inputDesc);
    auto dtype = inputDesc->GetDataType();
    OP_CHECK_IF(!IsSupportedFloatType(dtype), OP_LOGE(context, "dtype is not float path"), return GRAPH_FAILED);

    totalLength = shape.GetShapeSize();
    dtypeSize = GetTypeSize(dtype);
    dtypeId = GetDtypeId(dtype);
    if (dtypeSize == 0) {
        OP_LOGE(context, "dtype size is invalid");
        return GRAPH_FAILED;
    }
    if (dtypeId == IsPosInfDtypeId::INVALID) {
        OP_LOGE(context, "dtype id is invalid");
        return GRAPH_FAILED;
    }
    return GRAPH_SUCCESS;
}

static graphStatus InitWorkspaceAndTiling(gert::TilingContext* context, uint32_t wsSysSize, gert::TilingData*& tiling)
{
    size_t* currentWorkspace = context->GetWorkspaceSizes(1);
    OP_CHECK_NULL_WITH_CONTEXT(context, currentWorkspace);
    currentWorkspace[0] = wsSysSize;

    tiling = context->GetRawTilingData();
    OP_CHECK_NULL_WITH_CONTEXT(context, tiling);
    return GRAPH_SUCCESS;
}

static graphStatus ComputeTilePolicy(int64_t totalLength,
                                     int64_t dtypeSize,
                                     IsPosInfDtypeId dtypeId,
                                     int64_t coreNum,
                                     uint64_t ubSize,
                                     int64_t& usedCoreNum,
                                     int64_t& formerNum,
                                     int64_t& formerLength,
                                     int64_t& tailLength,
                                     int64_t& tileLength)
{
    if (totalLength <= 0) {
        return GRAPH_SUCCESS;
    }
    if (dtypeSize == 0) {
        return GRAPH_FAILED;
    }

    int64_t cacheLineElements = 0;
    if (dtypeSize == 0) {
        return GRAPH_FAILED;
    }
    cacheLineElements = CACHE_LINE_BYTE_LENGTH / dtypeSize;
    if (cacheLineElements == 0) {
        return GRAPH_FAILED;
    }

    if (coreNum == 0) {
        return GRAPH_FAILED;
    }
    int64_t totalLengthCore = (totalLength + coreNum - 1) / coreNum;
    int64_t totalLengthCoreAlign =
        ((totalLengthCore + cacheLineElements - 1) / cacheLineElements) * cacheLineElements;
    if (totalLengthCoreAlign == 0) {
        return GRAPH_FAILED;
    }

    usedCoreNum = (totalLength + totalLengthCoreAlign - 1) / totalLengthCoreAlign;
    formerNum = usedCoreNum - 1;
    formerLength = totalLengthCoreAlign;
    tailLength = totalLength - formerNum * formerLength;

    int64_t maxTileElements = static_cast<int64_t>(ubSize) / FLOAT_BUFFER_COEFFICIENT;
    int64_t alignElements = 0;
    if (dtypeSize == 0) {
        return GRAPH_FAILED;
    }
    alignElements = 256 / dtypeSize;
    if (alignElements == 0) {
        return GRAPH_FAILED;
    }

    tileLength = (maxTileElements / alignElements) * alignElements;
    if (dtypeId == IsPosInfDtypeId::FP32) {
        tileLength = std::min<int64_t>(tileLength, DEBUG_FP32_TILE_ELEMENTS);
    } else if (dtypeId == IsPosInfDtypeId::BF16) {
        tileLength = std::min<int64_t>(tileLength, DEBUG_BF16_TILE_ELEMENTS);
    }
    if (tileLength == 0) {
        tileLength = alignElements;
    }
    return GRAPH_SUCCESS;
}

static graphStatus SaveTilingData(gert::TilingContext* context,
                                  gert::TilingData* tiling,
                                  int64_t usedCoreNum,
                                  int64_t formerNum,
                                  int64_t formerLength,
                                  int64_t tailLength,
                                  int64_t tileLength,
                                  IsPosInfDtypeId dtypeId)
{
    IsPosInfTilingData tilingData;
    tilingData.set_formerNum(formerNum);
    tilingData.set_formerLength(formerLength);
    tilingData.set_tailLength(tailLength);
    tilingData.set_tileLength(tileLength);
    tilingData.set_dtypeId(static_cast<int64_t>(dtypeId));
    tilingData.SaveToBuffer(tiling->GetData(), tiling->GetCapacity());
    tiling->SetDataSize(tilingData.GetDataSize());
    context->SetBlockDim(static_cast<uint32_t>(usedCoreNum));
    return GRAPH_SUCCESS;
}

static graphStatus IsPosInfTilingFunc(gert::TilingContext* context)
{
    int64_t totalLength = 0;
    int64_t dtypeSize = 0;
    IsPosInfDtypeId dtypeId = IsPosInfDtypeId::INVALID;
    OP_CHECK_IF(ValidateInput(context, totalLength, dtypeSize, dtypeId) != GRAPH_SUCCESS,
                OP_LOGE(context, "ValidateInput failed"), return GRAPH_FAILED);

    int64_t coreNum = 0;
    uint64_t ubSize = 0;
    uint32_t wsSysSize = 0;
    OP_CHECK_IF(GetPlatformInfo(context, coreNum, ubSize, wsSysSize) != GRAPH_SUCCESS,
                OP_LOGE(context, "GetPlatformInfo failed"), return GRAPH_FAILED);

    gert::TilingData* tiling = nullptr;
    OP_CHECK_IF(InitWorkspaceAndTiling(context, wsSysSize, tiling) != GRAPH_SUCCESS,
                OP_LOGE(context, "InitWorkspaceAndTiling failed"), return GRAPH_FAILED);

    int64_t usedCoreNum = 1;
    int64_t formerNum = 0;
    int64_t formerLength = 0;
    int64_t tailLength = 0;
    int64_t tileLength = 0;
    OP_CHECK_IF(ComputeTilePolicy(totalLength, dtypeSize, dtypeId, coreNum, ubSize, usedCoreNum,
                                  formerNum, formerLength, tailLength, tileLength) != GRAPH_SUCCESS,
                OP_LOGE(context, "ComputeTilePolicy failed"), return GRAPH_FAILED);

    return SaveTilingData(context, tiling, usedCoreNum, formerNum, formerLength, tailLength, tileLength, dtypeId);
}

static graphStatus TilingParseForIsPosInf([[maybe_unused]] gert::TilingParseContext* context)
{
    return GRAPH_SUCCESS;
}

IMPL_OP_OPTILING(IsPosInf).Tiling(IsPosInfTilingFunc).TilingParse<IsPosInfCompileInfo>(TilingParseForIsPosInf);
}
