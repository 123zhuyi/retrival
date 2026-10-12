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
 * \file asin_grad_tiling.cpp
 * \brief AsinGrad tiling for Atlas A2 training series products and Atlas A3 series products.
 */

#include <algorithm>

#include "register/op_def_registry.h"
#include "tiling/platform/platform_ascendc.h"
#include "op_common/log/log.h"

#include "../op_kernel/asin_grad_tiling_data.h"
#include "../op_kernel/asin_grad_tiling_key.h"

namespace optiling {

constexpr int64_t CACHE_LINE_BYTE_LENGTH = 512;
constexpr int64_t DATA_COPY_ALIGN_BYTES = 32;
constexpr int64_t MIN_CORE_BYTES = 4096;
constexpr int64_t HALF_BUFFER_COEFFICIENT = 16;
constexpr int64_t FLOAT_BUFFER_COEFFICIENT = 28;
constexpr int64_t MAX_DIM_COUNT = 8;

struct AsinGradCompileInfo {};

static const gert::Shape SCALAR_SHAPE = {1};

static const gert::Shape& EnsureNotScalar(const gert::Shape& shape)
{
    if (shape.GetDimNum() == 0) {
        return SCALAR_SHAPE;
    }
    return shape;
}

static int64_t CeilDiv(int64_t value, int64_t divisor)
{
    if (divisor == 0) {
        return 0;
    }
    return (value + divisor - 1) / divisor;
}

static int64_t AlignUp(int64_t value, int64_t align)
{
    if (align == 0) {
        return value;
    }
    return CeilDiv(value, align) * align;
}

static int64_t AlignDown(int64_t value, int64_t align)
{
    if (align == 0) {
        return value;
    }
    return (value / align) * align;
}

static bool IsSupportedDtype(ge::DataType dataType)
{
    return dataType == ge::DT_FLOAT || dataType == ge::DT_FLOAT16 || dataType == ge::DT_BF16;
}

static int64_t GetDtypeSize(ge::DataType dataType) { return dataType == ge::DT_FLOAT ? 4 : 2; }

static int64_t GetBufferCoefficient(ge::DataType dataType)
{
    return dataType == ge::DT_FLOAT16 ? HALF_BUFFER_COEFFICIENT : FLOAT_BUFFER_COEFFICIENT;
}

struct PlatformCaps {
    uint64_t ubBytes = 0;
    int64_t aivCores = 0;
    uint32_t libApiWorkspace = 0;
};

static ge::graphStatus QueryPlatformCaps(gert::TilingContext* context, PlatformCaps& caps, bool queryUbAndCore)
{
    auto* rawPlatform = context->GetPlatformInfo();
    OP_CHECK_NULL_WITH_CONTEXT(context, rawPlatform);

    platform_ascendc::PlatformAscendC ascendPlatform(rawPlatform);
    caps.libApiWorkspace = ascendPlatform.GetLibApiWorkSpaceSize();
    if (!queryUbAndCore) {
        return ge::GRAPH_SUCCESS;
    }

    caps.aivCores = static_cast<int64_t>(ascendPlatform.GetCoreNumAiv());
    OP_CHECK_IF(caps.aivCores == 0, OP_LOGE(context, "AIV core count is 0"), return ge::GRAPH_FAILED);
    ascendPlatform.GetCoreMemSize(platform_ascendc::CoreMemType::UB, caps.ubBytes);
    OP_CHECK_IF(caps.ubBytes == 0, OP_LOGE(context, "UB size is 0"), return ge::GRAPH_FAILED);
    return ge::GRAPH_SUCCESS;
}

static ge::graphStatus GetPlatformInfo(gert::TilingContext* context, uint64_t& ubSize, int64_t& coreNum)
{
    PlatformCaps caps;
    auto status = QueryPlatformCaps(context, caps, true);
    if (status != ge::GRAPH_SUCCESS) {
        return status;
    }
    ubSize = caps.ubBytes;
    coreNum = caps.aivCores;
    return ge::GRAPH_SUCCESS;
}

static ge::graphStatus GetWorkspaceSize(gert::TilingContext* context)
{
    PlatformCaps caps;
    auto status = QueryPlatformCaps(context, caps, false);
    if (status != ge::GRAPH_SUCCESS) {
        return status;
    }

    size_t* currentWorkspace = context->GetWorkspaceSizes(1);
    OP_CHECK_NULL_WITH_CONTEXT(context, currentWorkspace);
    currentWorkspace[0] = caps.libApiWorkspace;
    return ge::GRAPH_SUCCESS;
}

static ge::graphStatus CheckInput(gert::TilingContext* context, int64_t& totalLength, ge::DataType& dataType)
{
    auto* yShape = context->GetInputShape(0);
    auto* dyShape = context->GetInputShape(1);
    auto* zShape = context->GetOutputShape(0);
    OP_CHECK_NULL_WITH_CONTEXT(context, yShape);
    OP_CHECK_NULL_WITH_CONTEXT(context, dyShape);
    OP_CHECK_NULL_WITH_CONTEXT(context, zShape);
    auto yOriginShape = yShape->GetOriginShape();
    auto dyOriginShape = dyShape->GetOriginShape();
    auto zOriginShape = zShape->GetOriginShape();
    OP_CHECK_IF(yOriginShape != dyOriginShape, OP_LOGE(context, "y and dy origin shapes are different"),
                return ge::GRAPH_FAILED);
    OP_CHECK_IF(yOriginShape != zOriginShape, OP_LOGE(context, "input and output origin shapes are different"),
                return ge::GRAPH_FAILED);
    OP_CHECK_IF(yOriginShape.GetDimNum() > MAX_DIM_COUNT, OP_LOGE(context, "input rank exceeds max dim count"),
                return ge::GRAPH_FAILED);

    auto* yDesc = context->GetInputDesc(0);
    auto* dyDesc = context->GetInputDesc(1);
    auto* zDesc = context->GetOutputDesc(0);
    OP_CHECK_NULL_WITH_CONTEXT(context, yDesc);
    OP_CHECK_NULL_WITH_CONTEXT(context, dyDesc);
    OP_CHECK_NULL_WITH_CONTEXT(context, zDesc);
    dataType = yDesc->GetDataType();
    OP_CHECK_IF(!IsSupportedDtype(dataType), OP_LOGE(context, "unsupported dtype %d", static_cast<int>(dataType)),
                return ge::GRAPH_FAILED);
    OP_CHECK_IF(dyDesc->GetDataType() != dataType || zDesc->GetDataType() != dataType,
                OP_LOGE(context, "input and output dtypes are not consistent"), return ge::GRAPH_FAILED);

    totalLength = EnsureNotScalar(yOriginShape).GetShapeSize();
    return ge::GRAPH_SUCCESS;
}

static void SetTemplateParam(gert::TilingContext* context, ge::DataType dataType)
{
    ASCENDC_TPL_SEL_PARAM(context, static_cast<uint32_t>(dataType));
}

static void FillEmptyTiling(AsinGradTilingData* tiling, ge::DataType dataType, gert::TilingContext* context)
{
    tiling->formerNum = 0;
    tiling->formerLength = 0;
    tiling->tailLength = 0;
    tiling->tileLength = 1;
    tiling->dtypeId = static_cast<int64_t>(dataType);
    context->SetBlockDim(1);
    SetTemplateParam(context, dataType);
}

static ge::graphStatus FillNormalTiling(gert::TilingContext* context, AsinGradTilingData* tiling, int64_t totalLength,
                                        ge::DataType dataType, uint64_t ubSize, int64_t coreNum)
{
    int64_t dtypeSize = GetDtypeSize(dataType);

    int64_t cacheLineElements = CeilDiv(CACHE_LINE_BYTE_LENGTH, dtypeSize);
    int64_t minCoreElements = AlignUp(CeilDiv(MIN_CORE_BYTES, dtypeSize), cacheLineElements);
    int64_t totalLengthCore = CeilDiv(totalLength, coreNum);
    int64_t totalLengthCoreAlign = AlignUp(std::max<int64_t>(totalLengthCore, minCoreElements), cacheLineElements);
    OP_CHECK_IF(totalLengthCoreAlign == 0, OP_LOGE(context, "totalLengthCoreAlign is 0"), return ge::GRAPH_FAILED);

    int64_t usedCoreNum = CeilDiv(totalLength, totalLengthCoreAlign);
    usedCoreNum = std::max<int64_t>(1, usedCoreNum);
    int64_t formerNum = usedCoreNum - 1;
    int64_t formerLength = totalLengthCoreAlign;
    int64_t tailLength = totalLength - formerNum * formerLength;

    int64_t maxTileElements = static_cast<int64_t>(ubSize) / GetBufferCoefficient(dataType);
    // Align tile so that both the T-typed DMA copy (32B) and the fp32 compute (32B) stay aligned.
    int64_t dataAlignElements = CeilDiv(DATA_COPY_ALIGN_BYTES, dtypeSize);
    int64_t fp32AlignElements = CeilDiv(DATA_COPY_ALIGN_BYTES, static_cast<int64_t>(sizeof(float)));
    int64_t alignElements = std::max<int64_t>(dataAlignElements, fp32AlignElements);
    int64_t tileLength = AlignDown(maxTileElements, alignElements);
    if (tileLength == 0) {
        tileLength = alignElements;
    }

    tiling->formerNum = formerNum;
    tiling->formerLength = formerLength;
    tiling->tailLength = tailLength;
    tiling->tileLength = tileLength;
    tiling->dtypeId = static_cast<int64_t>(dataType);

    context->SetBlockDim(static_cast<uint32_t>(usedCoreNum));
    SetTemplateParam(context, dataType);
    return ge::GRAPH_SUCCESS;
}

static ge::graphStatus AsinGradTilingFunc(gert::TilingContext* context)
{
    OP_CHECK_IF(context == nullptr, OP_LOGE(context, "context is nullptr"), return ge::GRAPH_FAILED);
    uint64_t ubSize = 0;
    int64_t coreNum = 0;
    OP_CHECK_IF(GetPlatformInfo(context, ubSize, coreNum) != ge::GRAPH_SUCCESS,
                OP_LOGE(context, "GetPlatformInfo failed"), return ge::GRAPH_FAILED);

    int64_t totalLength = 0;
    ge::DataType dataType = ge::DT_FLOAT;
    OP_CHECK_IF(CheckInput(context, totalLength, dataType) != ge::GRAPH_SUCCESS, OP_LOGE(context, "CheckInput failed"),
                return ge::GRAPH_FAILED);
    OP_CHECK_IF(GetWorkspaceSize(context) != ge::GRAPH_SUCCESS, OP_LOGE(context, "GetWorkspaceSize failed"),
                return ge::GRAPH_FAILED);

    auto* tiling = context->GetTilingData<AsinGradTilingData>();
    OP_CHECK_NULL_WITH_CONTEXT(context, tiling);

    if (totalLength <= 0) {
        FillEmptyTiling(tiling, dataType, context);
        return ge::GRAPH_SUCCESS;
    }
    return FillNormalTiling(context, tiling, totalLength, dataType, ubSize, coreNum);
}

static ge::graphStatus TilingParseForAsinGrad(gert::TilingParseContext* context)
{
    (void)context;
    return ge::GRAPH_SUCCESS;
}

IMPL_OP_OPTILING(AsinGrad).Tiling(AsinGradTilingFunc).TilingParse<AsinGradCompileInfo>(TilingParseForAsinGrad);
} // namespace optiling
