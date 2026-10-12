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
 * \file tabulate_fusion_tiling.cpp
 * \brief Tiling implementation for tabulate_fusion operator
 *
 * Multi-core split: two-step method (physical-core average -> reverse actual core num)
 * with PER_CORE_MIN=1024 lower-bound protection aligned to 32.
 * Work unit = (nloc, lastLayerSize), Grid-Stride traversal in kernel.
 */

#include "log/log.h"
#include "platform/platform_ascendc.h"
#include "util/math_util.h"
#include "util/platform_util.h"
#include "op_host/tiling_base_util.h"
#include "op_host/math_tiling_templates_registry.h"
#include "../../op_kernel/arch35/tabulate_fusion_tiling_data.h"
#include "../../op_kernel/arch35/tabulate_fusion_tiling_key.h"

namespace optiling {

using Ops::Base::EnsureNotScalar;

using namespace Ops::Math::OpTiling;

constexpr int32_t ALIGN_32 = 32;
constexpr int32_t ALIGN_64 = 64;
constexpr int64_t PER_CORE_MIN = 1024;       // lower bound, aligned to 32
constexpr uint32_t DCACHE_SIZE = 128 * 1024; // DCache 128KB for table gather caching
constexpr uint32_t STATIC_UB_ESTIMATE = 0;   // no static __ubuf__ arrays, all in registers
constexpr int32_t COEFF_COUNT = 6;           // polynomial coefficient count a0~a5
constexpr int64_t TABLE_INFO_MIN_SIZE = 5;   // minimum fields in table_info (lower/upper/max/stride0/stride1)

struct TabulateFusionCompileInfo {};

// Get platform info: ubSize and AIV core num
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

// Validate em shape, em_x total elements, and last_layer_size attr (subset of MDE 3.3)
static ge::graphStatus ValidateEmAndAttrs(gert::TilingContext* context, int64_t& nloc, int64_t& nnei,
                                          int64_t& lastLayerSize)
{
    // em: [nloc, nnei, 4]
    auto emInput = context->GetInputShape(3);
    OP_CHECK_NULL_WITH_CONTEXT(context, emInput);
    auto emShape = EnsureNotScalar(emInput->GetStorageShape());
    OP_CHECK_IF(emShape.GetDimNum() != 3,
                OP_LOGE(context, "TabulateFusion: em should be 3D, got %zu", emShape.GetDimNum()),
                return ge::GRAPH_FAILED);
    OP_CHECK_IF(emShape.GetDim(2) != 4,
                OP_LOGE(context, "TabulateFusion: em.shape[2] should be 4, got %ld", emShape.GetDim(2)),
                return ge::GRAPH_FAILED);
    nloc = emShape.GetDim(0);
    nnei = emShape.GetDim(1);
    OP_CHECK_IF(nloc <= 0, OP_LOGE(context, "TabulateFusion: nloc should be > 0"), return ge::GRAPH_FAILED);
    OP_CHECK_IF(nnei <= 0, OP_LOGE(context, "TabulateFusion: nnei should be > 0"), return ge::GRAPH_FAILED);

    // em_x: total elements should == nloc * nnei
    auto emXInput = context->GetInputShape(2);
    OP_CHECK_NULL_WITH_CONTEXT(context, emXInput);
    auto emXShape = EnsureNotScalar(emXInput->GetStorageShape());
    int64_t emXTotal = emXShape.GetShapeSize();
    OP_CHECK_IF(emXTotal != nloc * nnei,
                OP_LOGE(context, "TabulateFusion: em_x total %ld != nloc*nnei %ld*%ld", emXTotal, nloc, nnei),
                return ge::GRAPH_FAILED);

    // required attr last_layer_size
    auto attrs = context->GetAttrs();
    OP_CHECK_NULL_WITH_CONTEXT(context, attrs);
    const auto* lastLayerSizePtr = attrs->GetAttrPointer<int64_t>(0);
    OP_CHECK_NULL_WITH_CONTEXT(context, lastLayerSizePtr);
    lastLayerSize = *lastLayerSizePtr;
    OP_CHECK_IF(lastLayerSize <= 0,
                OP_LOGE(context, "TabulateFusion: last_layer_size should be > 0, got %ld", lastLayerSize),
                return ge::GRAPH_FAILED);

    return ge::GRAPH_SUCCESS;
}

// Validate table shape (2D, shape[1]==lastSizeAlign*COEFF_COUNT) and table_info size (subset of MDE 3.3)
static ge::graphStatus ValidateTableShapes(gert::TilingContext* context, int64_t lastLayerSize)
{
    // table: 2D, shape[1] == lastSizeAlign * COEFF_COUNT
    auto tableInput = context->GetInputShape(0);
    OP_CHECK_NULL_WITH_CONTEXT(context, tableInput);
    auto tableShape = EnsureNotScalar(tableInput->GetStorageShape());
    OP_CHECK_IF(tableShape.GetDimNum() != 2,
                OP_LOGE(context, "TabulateFusion: table should be 2D, got %zu", tableShape.GetDimNum()),
                return ge::GRAPH_FAILED);
    int64_t lastSizeAlign = ((lastLayerSize + ALIGN_64 - 1) / ALIGN_64) * ALIGN_64;
    OP_CHECK_IF(tableShape.GetDim(1) != lastSizeAlign * COEFF_COUNT,
                OP_LOGE(context, "TabulateFusion: table.shape[1] %ld != lastSizeAlign*6 %ld", tableShape.GetDim(1),
                        lastSizeAlign * COEFF_COUNT),
                return ge::GRAPH_FAILED);

    // table_info: 1D, total elements should >= TABLE_INFO_MIN_SIZE
    auto tableInfoInput = context->GetInputShape(1);
    OP_CHECK_NULL_WITH_CONTEXT(context, tableInfoInput);
    auto tableInfoShape = EnsureNotScalar(tableInfoInput->GetStorageShape());
    int64_t tableInfoTotal = tableInfoShape.GetShapeSize();
    OP_CHECK_IF(tableInfoTotal < TABLE_INFO_MIN_SIZE,
                OP_LOGE(context, "TabulateFusion: size of table_info should be >= 5, got %ld", tableInfoTotal),
                return ge::GRAPH_FAILED);

    return ge::GRAPH_SUCCESS;
}

// Validate dtype support and consistency across all inputs (subset of MDE 3.3)
static ge::graphStatus ValidateDtypeConsistency(gert::TilingContext* context)
{
    const std::set<ge::DataType> supportedDtype = {ge::DT_FLOAT, ge::DT_FLOAT16};
    for (size_t i = 0; i < 4; i++) {
        auto desc = context->GetInputDesc(i);
        OP_CHECK_NULL_WITH_CONTEXT(context, desc);
        ge::DataType dt = desc->GetDataType();
        OP_CHECK_IF(supportedDtype.count(dt) == 0,
                    OP_LOGE(context, "TabulateFusion: unsupported dtype at input %zu", i), return ge::GRAPH_FAILED);
        OP_CHECK_IF(i > 0 && dt != context->GetInputDesc(0)->GetDataType(),
                    OP_LOGE(context, "TabulateFusion: dtype of inputs should be the same"), return ge::GRAPH_FAILED);
    }
    return ge::GRAPH_SUCCESS;
}

// Get shape & attr info, with validation per MDE 3.3 (orchestrates sub-validators)
static ge::graphStatus GetShapeAttrsInfo(gert::TilingContext* context, int64_t& nloc, int64_t& nnei,
                                         int64_t& lastLayerSize)
{
    OP_CHECK_IF(ValidateEmAndAttrs(context, nloc, nnei, lastLayerSize) != ge::GRAPH_SUCCESS,
                OP_LOGE(context, "ValidateEmAndAttrs error"), return ge::GRAPH_FAILED);
    OP_CHECK_IF(ValidateTableShapes(context, lastLayerSize) != ge::GRAPH_SUCCESS,
                OP_LOGE(context, "ValidateTableShapes error"), return ge::GRAPH_FAILED);
    OP_CHECK_IF(ValidateDtypeConsistency(context) != ge::GRAPH_SUCCESS,
                OP_LOGE(context, "ValidateDtypeConsistency error"), return ge::GRAPH_FAILED);
    return ge::GRAPH_SUCCESS;
}

// Workspace: no user workspace, but must include system workspace (framework atomic ops need it)
static ge::graphStatus GetWorkspaceSize(gert::TilingContext* context)
{
    int64_t userWorkspaceSize = 0;
    auto ascendcPlatform = platform_ascendc::PlatformAscendC(context->GetPlatformInfo());
    uint64_t sysWorkspaceSize = ascendcPlatform.GetLibApiWorkSpaceSize();
    size_t* currentWorkspace = context->GetWorkspaceSizes(1);
    OP_CHECK_NULL_WITH_CONTEXT(context, currentWorkspace);
    currentWorkspace[0] = static_cast<size_t>(userWorkspaceSize + static_cast<int64_t>(sysWorkspaceSize));
    return ge::GRAPH_SUCCESS;
}

// Setup tiling data, multi-core split, UB config and tiling key
static ge::graphStatus SetupTilingDataAndConfig(gert::TilingContext* context, int64_t nloc, int64_t nnei,
                                                int64_t lastLayerSize, int64_t coreNum, uint64_t ubSize)
{
    int64_t lastSizeAlign = ((lastLayerSize + ALIGN_64 - 1) / ALIGN_64) * ALIGN_64;
    int64_t tableRowSize = lastSizeAlign * COEFF_COUNT;

    // table rows (for tableIdx bounds clamping in kernel, matching golden's OOB protection)
    auto tableInput = context->GetInputShape(0);
    OP_CHECK_NULL_WITH_CONTEXT(context, tableInput);
    auto tableShape = EnsureNotScalar(tableInput->GetStorageShape());
    int64_t tableRows = tableShape.GetDim(0);

    // multi-core split: two-step method + PER_CORE_MIN protection
    int64_t totalWork = nloc * lastLayerSize;
    int64_t perCoreElements = Ops::Base::CeilDiv(totalWork, coreNum);
    if (perCoreElements < PER_CORE_MIN) {
        perCoreElements = ((PER_CORE_MIN + ALIGN_32 - 1) / ALIGN_32) * ALIGN_32;
    }
    int64_t needCoreNum = Ops::Base::CeilDiv(totalWork, perCoreElements);
    if (needCoreNum < 1) {
        needCoreNum = 1;
    }

    // fill tiling data
    TabulateFusionTilingData* tiling = context->GetTilingData<TabulateFusionTilingData>();
    OP_CHECK_NULL_WITH_CONTEXT(context, tiling);
    OP_CHECK_IF(memset_s(tiling, sizeof(TabulateFusionTilingData), 0, sizeof(TabulateFusionTilingData)) != EOK,
                OP_LOGE(context, "set tiling data error"), return ge::GRAPH_FAILED);
    tiling->needCoreNum = static_cast<int32_t>(needCoreNum);
    tiling->nloc = static_cast<int32_t>(nloc);
    tiling->nnei = static_cast<int32_t>(nnei);
    tiling->lastLayerSize = static_cast<int32_t>(lastLayerSize);
    tiling->lastSizeAlign = static_cast<int32_t>(lastSizeAlign);
    tiling->tableRowSize = static_cast<int32_t>(tableRowSize);
    tiling->tableRows = static_cast<int32_t>(tableRows);

    context->SetBlockDim(needCoreNum);
    OP_CHECK_IF(ubSize <= DCACHE_SIZE + STATIC_UB_ESTIMATE,
                OP_LOGE(context, "ubSize %lu <= DCACHE_SIZE + STATIC_UB_ESTIMATE", ubSize), return ge::GRAPH_FAILED);
    auto res = context->SetLocalMemorySize(static_cast<uint32_t>(ubSize - DCACHE_SIZE - STATIC_UB_ESTIMATE));
    OP_CHECK_IF(res != ge::GRAPH_SUCCESS,
                OP_LOGE(context, "SetLocalMemorySize failed, ubSize=%lu, DCACHE_SIZE=%u, STATIC_UB_ESTIMATE=%u", ubSize,
                        DCACHE_SIZE, STATIC_UB_ESTIMATE),
                return ge::GRAPH_FAILED);

    // single scene mode, dtype handled by DTYPE_TABLE macro
    uint64_t tilingKey = GET_TPL_TILING_KEY(TABULATE_FUSION_MODE_DEFAULT);
    context->SetTilingKey(tilingKey);

    return ge::GRAPH_SUCCESS;
}

// Main tiling entry (orchestrates: platform → shape → workspace → tiling setup)
static ge::graphStatus TabulateFusionTilingFunc(gert::TilingContext* context)
{
    uint64_t ubSize;
    int64_t coreNum;
    OP_CHECK_IF(GetPlatformInfo(context, ubSize, coreNum) != ge::GRAPH_SUCCESS,
                OP_LOGE(context, "GetPlatformInfo error"), return ge::GRAPH_FAILED);

    int64_t nloc = 0, nnei = 0, lastLayerSize = 0;
    OP_CHECK_IF(GetShapeAttrsInfo(context, nloc, nnei, lastLayerSize) != ge::GRAPH_SUCCESS,
                OP_LOGE(context, "GetShapeAttrsInfo error"), return ge::GRAPH_FAILED);

    OP_CHECK_IF(GetWorkspaceSize(context) != ge::GRAPH_SUCCESS, OP_LOGE(context, "GetWorkspaceSize error"),
                return ge::GRAPH_FAILED);

    OP_CHECK_IF(SetupTilingDataAndConfig(context, nloc, nnei, lastLayerSize, coreNum, ubSize) != ge::GRAPH_SUCCESS,
                OP_LOGE(context, "SetupTilingDataAndConfig error"), return ge::GRAPH_FAILED);

    return ge::GRAPH_SUCCESS;
}

// TilingParse callback (required by R15 three-part registration)
static ge::graphStatus TilingParseForTabulateFusion([[maybe_unused]] gert::TilingParseContext* context)
{
    return ge::GRAPH_SUCCESS;
}

IMPL_OP_OPTILING(TabulateFusion)
    .Tiling(TabulateFusionTilingFunc)
    .TilingParse<TabulateFusionCompileInfo>(TilingParseForTabulateFusion);
} // namespace optiling
