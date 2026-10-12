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
#include <set>

#include <graph/utils/type_utils.h>

#include "log/log.h"
#include "register/op_impl_registry.h"
#include "securec.h"
#include "tiling/platform/platform_ascendc.h"

#include "../op_kernel/sqrt_grad_tiling_data.h"
#include "../op_kernel/sqrt_grad_tiling_key.h"

namespace optiling {

constexpr int64_t CACHE_LINE_BYTE_LENGTH = 512;
constexpr int64_t DATA_COPY_ALIGN_BYTES = 32;
constexpr int64_t COMPARE_ALIGN_BYTES = 256;
constexpr int64_t BUFFER_COEFFICIENT_FP32 = 25;
constexpr int64_t BUFFER_COEFFICIENT_CAST = 25;
constexpr int64_t MAX_DIM_COUNT = 8;

struct SqrtGradCompileInfo {};

static const gert::Shape SCALAR_SHAPE = {1};
static const std::set<ge::DataType> SUPPORTED_DTYPES = {ge::DT_FLOAT, ge::DT_FLOAT16, ge::DT_BF16};

static const gert::Shape &EnsureNotScalar(const gert::Shape &shape)
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

static ge::graphStatus GetWorkspaceSize(gert::TilingContext *context)
{
    auto ascendcPlatform = platform_ascendc::PlatformAscendC(context->GetPlatformInfo());
    uint32_t sysWorkspaceSize = ascendcPlatform.GetLibApiWorkSpaceSize();
    size_t *currentWorkspace = context->GetWorkspaceSizes(1);
    OP_CHECK_NULL_WITH_CONTEXT(context, currentWorkspace);
    currentWorkspace[0] = sysWorkspaceSize;
    return ge::GRAPH_SUCCESS;
}

static ge::graphStatus GetPlatformInfo(gert::TilingContext *context, uint64_t &ubSize, int64_t &coreNum)
{
    auto *platformInfo = context->GetPlatformInfo();
    OP_CHECK_NULL_WITH_CONTEXT(context, platformInfo);
    auto ascendcPlatform = platform_ascendc::PlatformAscendC(platformInfo);
    coreNum = static_cast<int64_t>(ascendcPlatform.GetCoreNumAiv());
    OP_CHECK_IF(coreNum == 0, OP_LOGE(context, "coreNum is 0"), return ge::GRAPH_FAILED);
    ascendcPlatform.GetCoreMemSize(platform_ascendc::CoreMemType::UB, ubSize);
    OP_CHECK_IF(ubSize == 0, OP_LOGE(context, "ubSize is 0"), return ge::GRAPH_FAILED);
    return ge::GRAPH_SUCCESS;
}

static ge::graphStatus GetDtypeSize(gert::TilingContext *context, ge::DataType dataType, int64_t &dtypeSize)
{
    uint32_t typeLength = 0;
    ge::TypeUtils::GetDataTypeLength(dataType, typeLength);
    OP_CHECK_IF(typeLength == 0, OP_LOGE(context, "typeLength is 0"), return ge::GRAPH_FAILED);
    dtypeSize = static_cast<int64_t>(typeLength);
    return ge::GRAPH_SUCCESS;
}

static ge::graphStatus CheckInput(gert::TilingContext *context, int64_t &totalLength, ge::DataType &dataType)
{
    auto *yShape = context->GetInputShape(0);
    auto *dyShape = context->GetInputShape(1);
    OP_CHECK_NULL_WITH_CONTEXT(context, yShape);
    OP_CHECK_NULL_WITH_CONTEXT(context, dyShape);
    auto yStorageShape = yShape->GetStorageShape();
    auto dyStorageShape = dyShape->GetStorageShape();
    OP_CHECK_IF(yStorageShape != dyStorageShape, OP_LOGE(context, "shape mismatch"), return ge::GRAPH_FAILED);
    OP_CHECK_IF(yStorageShape.GetDimNum() > MAX_DIM_COUNT, OP_LOGE(context, "dim count overflow"),
                return ge::GRAPH_FAILED);

    auto *yDesc = context->GetInputDesc(0);
    auto *dyDesc = context->GetInputDesc(1);
    OP_CHECK_NULL_WITH_CONTEXT(context, yDesc);
    OP_CHECK_NULL_WITH_CONTEXT(context, dyDesc);
    dataType = yDesc->GetDataType();
    OP_CHECK_IF(SUPPORTED_DTYPES.count(dataType) == 0, OP_LOGE(context, "unsupported dtype"),
                return ge::GRAPH_FAILED);
    OP_CHECK_IF(dyDesc->GetDataType() != dataType, OP_LOGE(context, "dtype mismatch"), return ge::GRAPH_FAILED);

    auto *zDesc = context->GetOutputDesc(0);
    OP_CHECK_NULL_WITH_CONTEXT(context, zDesc);
    OP_CHECK_IF(zDesc->GetDataType() != dataType, OP_LOGE(context, "output dtype mismatch"),
                return ge::GRAPH_FAILED);

    totalLength = EnsureNotScalar(yStorageShape).GetShapeSize();
    return ge::GRAPH_SUCCESS;
}

static void SetTemplateParam(gert::TilingContext *context, ge::DataType dataType)
{
    ASCENDC_TPL_SEL_PARAM(context, static_cast<uint32_t>(dataType));
}

static void FillEmptyTiling(SqrtGradTilingData *tiling, ge::DataType dataType, gert::TilingContext *context)
{
    tiling->formerNum = 0;
    tiling->formerLength = 0;
    tiling->tailLength = 0;
    tiling->tileLength = 1;
    tiling->dtypeId = static_cast<int64_t>(dataType);
    context->SetBlockDim(1);
    SetTemplateParam(context, dataType);
}

static ge::graphStatus FillNormalTiling(gert::TilingContext *context, SqrtGradTilingData *tiling, int64_t totalLength,
                                        ge::DataType dataType, uint64_t ubSize, int64_t coreNum)
{
    int64_t dtypeSize = 0;
    OP_CHECK_IF(GetDtypeSize(context, dataType, dtypeSize) != ge::GRAPH_SUCCESS, OP_LOGE(context, "dtype size error"),
                return ge::GRAPH_FAILED);

    int64_t cacheLineElements = CeilDiv(CACHE_LINE_BYTE_LENGTH, dtypeSize);
    int64_t totalLengthCore = CeilDiv(totalLength, coreNum);
    int64_t totalLengthCoreAlign = AlignUp(totalLengthCore, cacheLineElements);
    OP_CHECK_IF(totalLengthCoreAlign == 0, OP_LOGE(context, "totalLengthCoreAlign is 0"), return ge::GRAPH_FAILED);

    int64_t usedCoreNum = CeilDiv(totalLength, totalLengthCoreAlign);
    usedCoreNum = std::max<int64_t>(1, usedCoreNum);
    int64_t formerNum = usedCoreNum - 1;
    int64_t formerLength = totalLengthCoreAlign;
    int64_t tailLength = totalLength - formerNum * formerLength;

    int64_t bufferCoefficient = (dataType == ge::DT_FLOAT) ? BUFFER_COEFFICIENT_FP32 : BUFFER_COEFFICIENT_CAST;
    if (bufferCoefficient == 0) {
        return ge::GRAPH_FAILED;
    }
    int64_t maxTileElements = static_cast<int64_t>(ubSize) / bufferCoefficient;
    int64_t dataAlignElements = CeilDiv(DATA_COPY_ALIGN_BYTES, dtypeSize);
    int64_t compareAlignElements = CeilDiv(COMPARE_ALIGN_BYTES, static_cast<int64_t>(sizeof(float)));
    int64_t alignElements = std::max<int64_t>(dataAlignElements, compareAlignElements);
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

static ge::graphStatus SqrtGradTilingFunc(gert::TilingContext *context)
{
    uint64_t ubSize = 0;
    int64_t coreNum = 0;
    OP_CHECK_IF(GetPlatformInfo(context, ubSize, coreNum) != ge::GRAPH_SUCCESS,
                OP_LOGE(context, "GetPlatformInfo error"), return ge::GRAPH_FAILED);

    int64_t totalLength = 0;
    ge::DataType dataType = ge::DT_FLOAT;
    OP_CHECK_IF(CheckInput(context, totalLength, dataType) != ge::GRAPH_SUCCESS,
                OP_LOGE(context, "CheckInput error"), return ge::GRAPH_FAILED);
    OP_CHECK_IF(GetWorkspaceSize(context) != ge::GRAPH_SUCCESS,
                OP_LOGE(context, "GetWorkspaceSize error"), return ge::GRAPH_FAILED);

    auto *tiling = context->GetTilingData<SqrtGradTilingData>();
    OP_CHECK_NULL_WITH_CONTEXT(context, tiling);
    OP_CHECK_IF(memset_s(tiling, sizeof(SqrtGradTilingData), 0, sizeof(SqrtGradTilingData)) != EOK,
                OP_LOGE(context, "memset tiling failed"), return ge::GRAPH_FAILED);

    if (totalLength <= 0) {
        FillEmptyTiling(tiling, dataType, context);
        return ge::GRAPH_SUCCESS;
    }
    return FillNormalTiling(context, tiling, totalLength, dataType, ubSize, coreNum);
}

static ge::graphStatus TilingParseForSqrtGrad([[maybe_unused]] gert::TilingParseContext *context)
{
    return ge::GRAPH_SUCCESS;
}

IMPL_OP_OPTILING(SqrtGrad)
    .Tiling(SqrtGradTilingFunc)
    .TilingParse<SqrtGradCompileInfo>(TilingParseForSqrtGrad);
}  // namespace optiling
