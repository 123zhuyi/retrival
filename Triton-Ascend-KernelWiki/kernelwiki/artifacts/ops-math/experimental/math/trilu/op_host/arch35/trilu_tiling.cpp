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
#include "util/math_util.h"
#include "util/platform_util.h"
#include "op_host/tiling_base_util.h"
#include "tiling/platform/platform_ascendc.h"
#include "../../op_kernel/arch35/trilu_tiling_data.h"
#include "../../op_kernel/arch35/trilu_tiling_key.h"

namespace optiling {

constexpr uint32_t WS_SYS_SIZE = 0U;
constexpr uint32_t DCACHE_SIZE = 128 * 1024;
constexpr int64_t MIN_PER_CORE_ELEMENTS = 1024;
constexpr int64_t ALIGN_SIZE = 32;

struct TriluCompileInfo {};

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

static ge::graphStatus GetShapeAndAttrsInfo(
    gert::TilingContext* context, int64_t& totalElements, int64_t& h, int64_t& w, int64_t& diagonal, int64_t& upper,
    ge::DataType& dataType)
{
    auto inputShape = context->GetInputShape(0);
    OP_CHECK_NULL_WITH_CONTEXT(context, inputShape);
    auto inShape = Ops::Base::EnsureNotScalar(inputShape->GetStorageShape());
    totalElements = inShape.GetShapeSize();

    size_t dimNum = inShape.GetDimNum();
    if (dimNum >= 2) {
        h = inShape.GetDim(dimNum - 2);
        w = inShape.GetDim(dimNum - 1);
    } else if (dimNum == 1) {
        h = 1;
        w = inShape.GetDim(0);
    } else {
        h = 1;
        w = 1;
    }

    auto attrs = context->GetAttrs();
    OP_CHECK_NULL_WITH_CONTEXT(context, attrs);
    const int64_t* diagonalPtr = attrs->GetAttrPointer<int64_t>(0);
    OP_CHECK_NULL_WITH_CONTEXT(context, diagonalPtr);
    diagonal = *diagonalPtr;
    const int64_t* upperPtr = attrs->GetAttrPointer<int64_t>(1);
    OP_CHECK_NULL_WITH_CONTEXT(context, upperPtr);
    upper = *upperPtr;

    const std::set<ge::DataType> supportedDtype = {ge::DT_FLOAT, ge::DT_FLOAT16, ge::DT_INT32, ge::DT_INT64,
                                                   ge::DT_INT8,  ge::DT_INT16,   ge::DT_UINT8, ge::DT_UINT16};
    auto inputDesc = context->GetInputDesc(0);
    OP_CHECK_NULL_WITH_CONTEXT(context, inputDesc);
    dataType = inputDesc->GetDataType();
    if (supportedDtype.count(dataType) == 0) {
        OP_LOGE(context, "Trilu: unsupported dtype %d", static_cast<int>(dataType));
        return ge::GRAPH_FAILED;
    }
    return ge::GRAPH_SUCCESS;
}

static ge::graphStatus GetWorkspaceSize(gert::TilingContext* context)
{
    size_t* currentWorkspace = context->GetWorkspaceSizes(1);
    OP_CHECK_NULL_WITH_CONTEXT(context, currentWorkspace);
    currentWorkspace[0] = WS_SYS_SIZE;
    return ge::GRAPH_SUCCESS;
}

static ge::graphStatus GetTilingKeyByDtype(gert::TilingContext* context, ge::DataType dataType, uint64_t& tilingKey)
{
    static const std::map<ge::DataType, int64_t> dtypeModeMap = {
        {ge::DT_FLOAT, TRILU_TPL_SCH_MODE_0}, {ge::DT_FLOAT16, TRILU_TPL_SCH_MODE_1},
        {ge::DT_INT32, TRILU_TPL_SCH_MODE_2}, {ge::DT_INT64, TRILU_TPL_SCH_MODE_3},
        {ge::DT_INT8, TRILU_TPL_SCH_MODE_4},  {ge::DT_INT16, TRILU_TPL_SCH_MODE_5},
        {ge::DT_UINT8, TRILU_TPL_SCH_MODE_6}, {ge::DT_UINT16, TRILU_TPL_SCH_MODE_7},
    };
    auto it = dtypeModeMap.find(dataType);
    OP_CHECK_IF(
        it == dtypeModeMap.end(),
        OP_LOGE(context, "Trilu: unsupported dtype %d for tiling key", static_cast<int>(dataType)),
        return ge::GRAPH_FAILED);
    tilingKey = GET_TPL_TILING_KEY(static_cast<uint64_t>(it->second));
    return ge::GRAPH_SUCCESS;
}

static ge::graphStatus CalcTilingParams(
    gert::TilingContext* context, int64_t totalElements, int64_t coreNum, uint64_t ubSize, int64_t diagonal,
    int64_t upper, int64_t h, int64_t w)
{
    TriluTilingData* tiling = context->GetTilingData<TriluTilingData>();
    OP_CHECK_NULL_WITH_CONTEXT(context, tiling);
    OP_CHECK_IF(
        memset_s(tiling, sizeof(TriluTilingData), 0, sizeof(TriluTilingData)) != EOK,
        OP_LOGE(context, "set tiling data error"), return ge::GRAPH_FAILED);

    tiling->totalElements = totalElements;

    int64_t perCoreElements = Ops::Base::CeilDiv(totalElements, coreNum);
    perCoreElements = Ops::Base::CeilAlign(perCoreElements, ALIGN_SIZE);
    perCoreElements = std::max(MIN_PER_CORE_ELEMENTS, perCoreElements);
    tiling->perCoreElements = perCoreElements;

    int32_t needCoreNum = static_cast<int32_t>(Ops::Base::CeilDiv(totalElements, perCoreElements));
    tiling->needCoreNum = needCoreNum;

    int64_t lastCoreElements = totalElements - perCoreElements * (needCoreNum - 1);
    tiling->lastCoreElements = lastCoreElements;

    tiling->diagonal = diagonal;
    tiling->upper = static_cast<int32_t>(upper);
    tiling->h = h;
    tiling->w = w;

    context->SetBlockDim(needCoreNum);
    context->SetLocalMemorySize(static_cast<uint32_t>(ubSize - DCACHE_SIZE));
    return ge::GRAPH_SUCCESS;
}

static ge::graphStatus TriluTilingFunc(gert::TilingContext* context)
{
    uint64_t ubSize = 0;
    int64_t coreNum = 0;
    OP_CHECK_IF(
        GetPlatformInfo(context, ubSize, coreNum) != ge::GRAPH_SUCCESS, OP_LOGE(context, "GetPlatformInfo error"),
        return ge::GRAPH_FAILED);

    int64_t totalElements = 0;
    int64_t h = 0;
    int64_t w = 0;
    int64_t diagonal = 0;
    int64_t upper = 1;
    ge::DataType dataType;
    OP_CHECK_IF(
        GetShapeAndAttrsInfo(context, totalElements, h, w, diagonal, upper, dataType) != ge::GRAPH_SUCCESS,
        OP_LOGE(context, "GetShapeAndAttrsInfo error"), return ge::GRAPH_FAILED);

    OP_CHECK_IF(
        GetWorkspaceSize(context) != ge::GRAPH_SUCCESS, OP_LOGE(context, "GetWorkspaceSize error"),
        return ge::GRAPH_FAILED);

    OP_CHECK_IF(
        CalcTilingParams(context, totalElements, coreNum, ubSize, diagonal, upper, h, w) != ge::GRAPH_SUCCESS,
        OP_LOGE(context, "CalcTilingParams error"), return ge::GRAPH_FAILED);

    uint64_t tilingKey = 0;
    OP_CHECK_IF(
        GetTilingKeyByDtype(context, dataType, tilingKey) != ge::GRAPH_SUCCESS,
        OP_LOGE(context, "GetTilingKeyByDtype error"), return ge::GRAPH_FAILED);
    context->SetTilingKey(tilingKey);

    return ge::GRAPH_SUCCESS;
}

static ge::graphStatus TilingParseForTrilu([[maybe_unused]] gert::TilingParseContext* context)
{
    return ge::GRAPH_SUCCESS;
}
IMPL_OP_OPTILING(Trilu).Tiling(TriluTilingFunc).TilingParse<TriluCompileInfo>(TilingParseForTrilu);
} // namespace optiling
