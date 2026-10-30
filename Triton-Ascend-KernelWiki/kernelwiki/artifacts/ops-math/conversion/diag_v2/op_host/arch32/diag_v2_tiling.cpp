/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

/*!
 * \file diag_v2_tiling.cpp
 * \brief
 */
#include "diag_v2_tiling.h"
#include "log/log.h"
#include "platform/platform_ascendc.h"
#include "util/math_util.h"

namespace optiling {

static const int32_t SIZE_1 = 1;
static const int32_t SIZE_2 = 2;
static const int32_t SIZE_4 = 4;
static const int32_t SIZE_8 = 8;
static const int32_t SIZE_16 = 16;
static const int32_t LENGTH_1024 = 1024;
static const int32_t LENGTH_128 = 128;
static const int32_t LENGTH_64 = 64;
static const int32_t LENGTH_32 = 32;

inline static ge::graphStatus DiagV2SetTilingData(gert::TilingContext* context, DiagV2TilingData& tilingData)
{
    auto rawTilingData = context->GetRawTilingData();
    tilingData.SaveToBuffer(rawTilingData->GetData(), rawTilingData->GetCapacity());
    rawTilingData->SetDataSize(tilingData.GetDataSize());

    return ge::GRAPH_SUCCESS;
}

static inline ge::graphStatus CalcAuxMatrixTiling(const int32_t typeSize, DiagV2TilingData& tilingData)
{
    switch (typeSize) {
        case SIZE_1:
            tilingData.set_tilingKey(static_cast<int64_t>(DiagV2TilingKey::ASSIST_SIZE_1));
            tilingData.set_matrixRowLength(LENGTH_128);
            break;
        case SIZE_2:
            tilingData.set_tilingKey(static_cast<int64_t>(DiagV2TilingKey::ASSIST_SIZE_2));
            tilingData.set_matrixRowLength(LENGTH_128);
            break;
        case SIZE_4:
            tilingData.set_tilingKey(static_cast<int64_t>(DiagV2TilingKey::ASSIST_SIZE_4));
            tilingData.set_matrixRowLength(LENGTH_64);
            break;
        case SIZE_8:
            tilingData.set_tilingKey(static_cast<int64_t>(DiagV2TilingKey::ASSIST_SIZE_8));
            tilingData.set_matrixRowLength(LENGTH_64);
            break;
        case SIZE_16:
            tilingData.set_tilingKey(static_cast<int64_t>(DiagV2TilingKey::ASSIST_SIZE_16));
            tilingData.set_matrixRowLength(LENGTH_32);
            break;
        default:
            break;
    }

    return ge::GRAPH_SUCCESS;
}

static inline ge::graphStatus processDiagFlat(gert::TilingContext* context)
{
    // call DiagFlat tiling for DiagV2
    return TilingDiagFlat(context);
}

static inline int64_t CalcLeastNumPerCore(const int32_t typeSize, const gert::TilingContext* context)
{
    OP_CHECK_IF(
        (typeSize <= 0), OP_LOGE(context, "Tiling4DiagV2 typeSize is invalid %d, please check.", typeSize), return -1);

    int64_t leastNumPerCore = BLOCK_SIZE / typeSize;
    return (leastNumPerCore > 0) ? leastNumPerCore : 1;
}

static ge::graphStatus Tiling4DiagV2(gert::TilingContext* context)
{
    OP_LOGD(context, "Tiling4DiagV2 running begin");
    auto inputShapePtr = context->GetInputShape(0);
    OP_CHECK_NULL_WITH_CONTEXT(context, inputShapePtr);
    auto inputShape = inputShapePtr->GetStorageShape();
    OP_CHECK_IF(
        (inputShape.GetDimNum() != SIZE_2 && inputShape.GetDimNum() != 1),
        OP_LOGE(
            context, "Tiling4DiagV2 get input shape dim(=%zu) is not 1 or 2, please check.", inputShape.GetDimNum()),
        return ge::GRAPH_FAILED);
    if (inputShape.GetDimNum() == 1) {
        return processDiagFlat(context);
    }

    DiagV2TilingData tilingData;
    tilingData.set_xHeight(inputShape.GetDim(0));
    tilingData.set_xWidth(inputShape.GetDim(1));
    auto attrs = context->GetAttrs();
    OP_CHECK_NULL_WITH_CONTEXT(context, attrs);
    const int64_t* diagonalPtr = attrs->GetAttrPointer<int64_t>(0);
    OP_CHECK_NULL_WITH_CONTEXT(context, diagonalPtr);
    const int64_t diagonal = *diagonalPtr;
    OP_CHECK_IF(
        (diagonal >= 0 && diagonal > tilingData.get_xWidth()) ||
            (diagonal < 0 && std::abs(diagonal) > tilingData.get_xHeight()),
        OP_LOGE(
            context, "Tiling4DiagV2 attr diagonal(=%ld) is wrong, please check. w=%ld, h=%ld", diagonal,
            tilingData.get_xWidth(), tilingData.get_xHeight()),
        return ge::GRAPH_FAILED);
    tilingData.set_gmOffset(diagonal > 0 ? diagonal : std::abs(diagonal) * tilingData.get_xWidth());
    tilingData.set_numOut(
        (diagonal >= 0) ? std::min((tilingData.get_xWidth() - diagonal), tilingData.get_xHeight()) :
                          std::min((tilingData.get_xHeight() + diagonal), tilingData.get_xWidth()));
    auto inputDesc = context->GetInputDesc(0);
    OP_CHECK_NULL_WITH_CONTEXT(context, inputDesc);
    auto dataType = inputDesc->GetDataType();
    const int32_t typeSize = ge::GetSizeByDataType(dataType);
    int64_t leastNumPerCore = CalcLeastNumPerCore(typeSize, context);
    OP_CHECK_IF(
        (leastNumPerCore <= 0), OP_LOGE(context, "Tiling4DiagV2 leastNumPerCore is invalid, please check."),
        return ge::GRAPH_FAILED);

    auto compileInfo = reinterpret_cast<const DiagV2CompileInfo*>(context->GetCompileInfo());
    OP_CHECK_NULL_WITH_CONTEXT(context, compileInfo);
    uint32_t coreNum = compileInfo->totalCoreNum;
    int64_t tmpRealCoreNum = tilingData.get_numOut() < coreNum ? tilingData.get_numOut() : coreNum;
    int64_t tmpCorePerNum = Ops::Base::CeilDiv(tilingData.get_numOut(), tmpRealCoreNum);
    tilingData.set_numPerCore(Ops::Base::CeilDiv(tmpCorePerNum, leastNumPerCore) * leastNumPerCore);
    tilingData.set_realCoreNum(Ops::Base::CeilDiv(tilingData.get_numOut(), tilingData.get_numPerCore()));
    tilingData.set_tailNum(tilingData.get_numOut() - (tilingData.get_realCoreNum() - 1) * tilingData.get_numPerCore());
    OP_CHECK_IF(
        CalcAuxMatrixTiling(typeSize, tilingData) != ge::GRAPH_SUCCESS, OP_LOGE(context, "CalcAuxMatrixTiling fail."),
        return ge::GRAPH_FAILED);
    OP_CHECK_IF(
        DiagV2SetTilingData(context, tilingData) != ge::GRAPH_SUCCESS,
        OP_LOGE(context, "DiagV2SetTilingData set tiling data fail."), return ge::GRAPH_FAILED);
    context->SetBlockDim(tilingData.get_realCoreNum());
    context->SetTilingKey(tilingData.get_tilingKey());

    size_t* workspaces = context->GetWorkspaceSizes(1);
    workspaces[0] = SIZE_16 * LENGTH_1024 * LENGTH_1024;

    OP_LOGD(
        context,
        "tilingData is xWidth:%ld, xHeight:%ld, gmOffset:%ld, numOut:%ld, realCoreNum:%ld, numPerCore:%ld, tailNum:%ld,\
                              tilingKey:%ld, matrixRowLength:%ld",
        tilingData.get_xWidth(), tilingData.get_xHeight(), tilingData.get_gmOffset(), tilingData.get_numOut(),
        tilingData.get_realCoreNum(), tilingData.get_numPerCore(), tilingData.get_tailNum(), tilingData.get_tilingKey(),
        tilingData.get_matrixRowLength());

    return ge::GRAPH_SUCCESS;
}

static ge::graphStatus TilingPrepare4DiagV2(gert::TilingParseContext* context)
{
    auto compileInfo = context->GetCompiledInfo<DiagV2CompileInfo>();
    OP_CHECK_NULL_WITH_CONTEXT(context, compileInfo);
    auto platformInfo = context->GetPlatformInfo();
    OP_CHECK_NULL_WITH_CONTEXT(context, platformInfo);
    auto ascendcPlatform = platform_ascendc::PlatformAscendC(platformInfo);
    compileInfo->totalCoreNum = ascendcPlatform.GetCoreNumAiv();
    OP_CHECK_IF(
        (compileInfo->totalCoreNum <= 0), OP_LOGE(context, "TilingPrepare4DiagV2 fail to get core num."),
        return ge::GRAPH_FAILED);

    uint64_t ubSizePlatForm;
    ascendcPlatform.GetCoreMemSize(platform_ascendc::CoreMemType::UB, ubSizePlatForm);
    compileInfo->ubSizePlatForm = static_cast<int64_t>(ubSizePlatForm);
    OP_CHECK_IF(
        (compileInfo->ubSizePlatForm <= 0), OP_LOGE(context, "TilingPrepare4DiagFlat fail to get ub size."),
        return ge::GRAPH_FAILED);

    return ge::GRAPH_SUCCESS;
}

IMPL_OP_OPTILING(DiagV2).Tiling(Tiling4DiagV2).TilingParse<DiagV2CompileInfo>(TilingPrepare4DiagV2);
} // namespace optiling
