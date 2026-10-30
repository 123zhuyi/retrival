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
 * \file diag_v2_tiling.cpp
 * \brief DiagV2 Host-side Tiling implementation (arch35, DAV_3510)
 *
 * Design: DESIGN.md v2.5
 *
 * One-way dependency: diag_v2 → diag_flat.
 *   rank<=1 → call TilingDiagFlatArch35() (defined in diag_flat), fill own TilingData, IS_1D_INPUT=1
 *   rank>=2 → local 2D→1D tiling, IS_1D_INPUT=0
 */

#include "diag_v2_tiling.h"
#include "register/op_def_registry.h"
#include "op_common/log/log.h"
#include "op_common/op_host/util/math_util.h"
#include "op_common/op_host/util/platform_util.h"
#include "../../../diag_flat/op_host/arch35/diag_flat_tiling.h"
#include "../../op_kernel/arch35/diag_v2_tiling_data.h"
#include "../../op_kernel/arch35/diag_v2_tiling_key.h"
#include <algorithm>
#include <cstring>

namespace optiling {

using Ops::Base::CeilDiv;

constexpr int64_t MIN_WORK_PER_CORE = 256;
constexpr int64_t TILE_LENGTH = 2048;
constexpr size_t ATTR_DIAGONAL_IDX = 0;
constexpr size_t WORKSPACE_NUM = 1;
constexpr uint32_t WS_SYS_SIZE = 0U;

// ============================================================================
// Helper: Get platform info (ubSize, coreNum)
// ============================================================================
static ge::graphStatus GetPlatformInfo(gert::TilingContext* context,
                                        uint64_t* ubSize, int64_t* coreNum)
{
    fe::PlatFormInfos* platformInfoPtr = context->GetPlatformInfo();
    OP_CHECK_NULL_WITH_CONTEXT(context, platformInfoPtr);
    auto ascendcPlatform = platform_ascendc::PlatformAscendC(platformInfoPtr);
    *coreNum = ascendcPlatform.GetCoreNumAiv();
    OP_CHECK_IF(*coreNum == 0, OP_LOGE(context, "coreNum is 0"),
                return ge::GRAPH_FAILED);
    ascendcPlatform.GetCoreMemSize(platform_ascendc::CoreMemType::UB, *ubSize);
    OP_CHECK_IF(*ubSize == 0, OP_LOGE(context, "ubSize is 0"),
                return ge::GRAPH_FAILED);
    return ge::GRAPH_SUCCESS;
}

// ============================================================================
// Helper: 32B alignment
// ============================================================================
static int64_t AlignUp(int64_t n, int64_t align)
{
    return ((n + align - 1) / align) * align;
}

// ============================================================================
// Helper: Get dtype size
// ============================================================================
static int64_t GetDtypeSize(ge::DataType dataType)
{
    switch (dataType) {
        case ge::DT_FLOAT16: case ge::DT_BF16:
        case ge::DT_INT16: case ge::DT_UINT16:
            return 2;
        case ge::DT_FLOAT: case ge::DT_INT32: case ge::DT_UINT32:
            return 4;
        case ge::DT_DOUBLE: case ge::DT_INT64: case ge::DT_UINT64:
        case ge::DT_COMPLEX64:
            return 8;
        case ge::DT_INT8: case ge::DT_UINT8: case ge::DT_BOOL:
            return 1;
        default:
            return 4;
    }
}

// ============================================================================
// 1D→2D via diag_flat (one-way: diag_v2 → diag_flat)
// ============================================================================
static inline ge::graphStatus ProcessDiagFlat(gert::TilingContext* context)
{
    DiagFlatTilingOutput out;
    OP_CHECK_IF(
        TilingDiagFlatArch35(context, &out) != ge::GRAPH_SUCCESS,
        OP_LOGE(context, "TilingDiagFlatArch35 error"),
        return ge::GRAPH_FAILED);

    // Fill diag_v2's own TilingData from diag_flat's output
    DiagV2Arch35TilingData* tiling = context->GetTilingData<DiagV2Arch35TilingData>();
    OP_CHECK_NULL_WITH_CONTEXT(context, tiling);
    OP_CHECK_IF(
        memset_s(tiling, sizeof(DiagV2Arch35TilingData), 0, sizeof(DiagV2Arch35TilingData)) != EOK,
        OP_LOGE(context, "set tiling data error"),
        return ge::GRAPH_FAILED);

    tiling->diagonal    = out.diagonal;
    tiling->realCoreNum = out.realCoreNum;
    tiling->tileLength  = out.tileLength;
    tiling->numInput    = out.numInput;
    tiling->outWidth    = out.outWidth;
    tiling->outTotal    = out.outTotal;
    tiling->outPerCore  = out.outPerCore;

    // Select diag_v2's own TilingKey: IS_1D_INPUT=1 → DiagFlatSimd kernel
    ASCENDC_TPL_SEL_PARAM(context, static_cast<uint32_t>(1));

    size_t* currentWorkspace = context->GetWorkspaceSizes(WORKSPACE_NUM);
    OP_CHECK_NULL_WITH_CONTEXT(context, currentWorkspace);
    currentWorkspace[0] = WS_SYS_SIZE;

    return ge::GRAPH_SUCCESS;
}

// ============================================================================
// 2D→1D tiling (IS_1D_INPUT=0)
// ============================================================================
static ge::graphStatus Tiling2Dto1D(gert::TilingContext* context,
                                     int64_t hwCoreNum, int64_t dtypeSize,
                                     uint64_t ubSize)
{
    auto inputX = context->GetInputShape(0);
    OP_CHECK_NULL_WITH_CONTEXT(context, inputX);
    auto inputShape = inputX->GetStorageShape();
    int64_t xHeight = inputShape.GetDim(0);
    int64_t xWidth  = inputShape.GetDim(1);

    auto attrs = context->GetAttrs();
    OP_CHECK_NULL_WITH_CONTEXT(context, attrs);
    const int64_t* diagonalPtr = attrs->GetAttrPointer<int64_t>(ATTR_DIAGONAL_IDX);
    int64_t diagonal = (diagonalPtr != nullptr) ? *diagonalPtr : 0;

    int64_t numOut = 0;
    int64_t gmOffset = 0;

    if (xHeight == 0 || xWidth == 0) {
        numOut = 0;
    } else if (diagonal >= 0) {
        if (diagonal >= xWidth) {
            numOut = 0;
        } else {
            numOut = std::min(xHeight, xWidth - diagonal);
            gmOffset = diagonal;
        }
    } else {
        if (-diagonal >= xHeight) {
            numOut = 0;
        } else {
            numOut = std::min(xHeight + diagonal, xWidth);
            gmOffset = -diagonal * xWidth;
        }
    }

    int64_t realCoreNum = std::min(hwCoreNum, std::max<int64_t>(1, numOut / MIN_WORK_PER_CORE));

    int64_t alignElems = std::max<int64_t>(1, 32 / dtypeSize);
    int64_t numPerCore = 0;
    if (numOut > 0) {
        numPerCore = AlignUp(CeilDiv(numOut, realCoreNum), alignElems);
    }
    int64_t tailNum = numOut - (realCoreNum - 1) * numPerCore;

    int64_t tileLength = std::min(TILE_LENGTH, numPerCore);
    int64_t threadNum = std::min(TILE_LENGTH, numPerCore);

    DiagV2Arch35TilingData* tiling = context->GetTilingData<DiagV2Arch35TilingData>();
    OP_CHECK_NULL_WITH_CONTEXT(context, tiling);
    OP_CHECK_IF(
        memset_s(tiling, sizeof(DiagV2Arch35TilingData), 0, sizeof(DiagV2Arch35TilingData)) != EOK,
        OP_LOGE(context, "set tiling data error"),
        return ge::GRAPH_FAILED);

    tiling->xWidth      = xWidth;
    tiling->xHeight     = xHeight;
    tiling->gmOffset    = gmOffset;
    tiling->numOut      = numOut;
    tiling->realCoreNum = realCoreNum;
    tiling->numPerCore  = numPerCore;
    tiling->tailNum     = tailNum;
    tiling->diagonal    = diagonal;
    tiling->tileLength  = tileLength;
    tiling->threadNum   = threadNum;

    context->SetLocalMemorySize(ubSize - 64 * 1024);

    context->SetBlockDim(realCoreNum);
    return ge::GRAPH_SUCCESS;
}

// ============================================================================
// Tiling entry
// ============================================================================
static ge::graphStatus DiagV2TilingFunc(gert::TilingContext* context)
{
    auto inputX = context->GetInputShape(0);
    OP_CHECK_NULL_WITH_CONTEXT(context, inputX);
    auto inputShape = inputX->GetStorageShape();
    int64_t rank = inputShape.GetDimNum();

    // Route: rank<=1 → delegate to diag_flat, rank>=2 → 2D→1D
    if (rank <= 1) {
        return ProcessDiagFlat(context);
    }

    // 2D→1D path
    uint64_t ubSize;
    int64_t hwCoreNum;
    OP_CHECK_IF(
        GetPlatformInfo(context, &ubSize, &hwCoreNum) != ge::GRAPH_SUCCESS,
        OP_LOGE(context, "GetPlatformInfo error"),
        return ge::GRAPH_FAILED);

    auto inputDesc = context->GetInputDesc(0);
    OP_CHECK_NULL_WITH_CONTEXT(context, inputDesc);
    int64_t dtypeSize = GetDtypeSize(inputDesc->GetDataType());

    ge::graphStatus ret = Tiling2Dto1D(context, hwCoreNum, dtypeSize, ubSize);
    OP_CHECK_IF(ret != ge::GRAPH_SUCCESS, OP_LOGE(context, "Tiling2Dto1D error"), return ret);

    size_t* currentWorkspace = context->GetWorkspaceSizes(WORKSPACE_NUM);
    OP_CHECK_NULL_WITH_CONTEXT(context, currentWorkspace);
    currentWorkspace[0] = WS_SYS_SIZE;

    ASCENDC_TPL_SEL_PARAM(context, static_cast<uint32_t>(0));  // IS_1D_INPUT=0

    return ge::GRAPH_SUCCESS;
}

// ============================================================================
// TilingParse
// ============================================================================
static ge::graphStatus TilingParseForDiagV2([[maybe_unused]] gert::TilingParseContext* context)
{
    return ge::GRAPH_SUCCESS;
}

// ============================================================================
// Tiling registration
// ============================================================================
IMPL_OP_OPTILING(DiagV2)
    .Tiling(DiagV2TilingFunc)
    .TilingParse<DiagV2CompileInfo>(TilingParseForDiagV2);

} // namespace optiling
