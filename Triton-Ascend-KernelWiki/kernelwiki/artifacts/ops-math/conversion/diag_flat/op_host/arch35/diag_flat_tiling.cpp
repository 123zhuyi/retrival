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
 * \file diag_flat_tiling.cpp
 * \brief DiagFlat Host-side Tiling implementation (arch35, DAV_3510)
 *
 * Design: DESIGN.md v2.1 Sec 3.3
 *
 * TilingDiagFlatArch35() is non-static, exported via diag_flat_tiling.h.
 * Called by both DiagFlat (own tiling) and DiagV2 (rank==1 delegation).
 * One-way dependency: this file includes nothing from diag_v2.
 */

#include "diag_flat_tiling.h"
#include "register/op_def_registry.h"
#include "op_common/log/log.h"
#include "op_common/op_host/util/math_util.h"
#include "op_common/op_host/util/platform_util.h"
#include "../../op_kernel/arch35/diag_flat_tiling_data.h"
#include "../../op_kernel/arch35/diag_flat_tiling_key.h"
#include <algorithm>
#include <cstring>

namespace optiling {

using Ops::Base::CeilDiv;

constexpr int64_t MIN_WORK_PER_CORE = 256;
constexpr int64_t DIAG_FLAT_BUFFER_NUM = 2;
constexpr int64_t SIMT_DCACHE_SIZE = 64 * 1024;

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
        case ge::DT_INT8: case ge::DT_UINT8:
            return 1;
        default:
            return 4;
    }
}

// ============================================================================
// Exported: core 1D→2D tiling computation (one-way dependency: no diag_v2 includes)
// ============================================================================
ge::graphStatus TilingDiagFlatArch35(gert::TilingContext* context, DiagFlatTilingOutput* out)
{
    // 1. Get platform info
    uint64_t ubSize;
    int64_t hwCoreNum;
    OP_CHECK_IF(
        GetPlatformInfo(context, &ubSize, &hwCoreNum) != ge::GRAPH_SUCCESS,
        OP_LOGE(context, "GetPlatformInfo error"),
        return ge::GRAPH_FAILED);

    // 2. Get input shape and numInput
    auto inputX = context->GetInputShape(0);
    OP_CHECK_NULL_WITH_CONTEXT(context, inputX);
    auto inputShape = inputX->GetStorageShape();
    int64_t numInput = inputShape.GetShapeSize();

    // 3. Get diagonal attribute
    auto attrs = context->GetAttrs();
    OP_CHECK_NULL_WITH_CONTEXT(context, attrs);
    const int64_t* diagonalPtr = attrs->GetAttrPointer<int64_t>(0);
    int64_t diagonal = (diagonalPtr != nullptr) ? *diagonalPtr : 0;

    // 4. Get dtype size
    auto inputDesc = context->GetInputDesc(0);
    OP_CHECK_NULL_WITH_CONTEXT(context, inputDesc);
    auto dataType = inputDesc->GetDataType();
    int64_t dtypeSize = GetDtypeSize(dataType);

    // 5. Compute output dimensions
    int64_t outWidth = numInput + (diagonal >= 0 ? diagonal : -diagonal);
    int64_t outTotal = outWidth * outWidth;

    // 6. Multi-core split
    int64_t realCoreNum = std::min(hwCoreNum,
        std::max<int64_t>(1, (numInput + outTotal) / MIN_WORK_PER_CORE / 2));

    // 7. outPerCore
    int64_t outPerCore = 0;
    if (outTotal > 0) {
        outPerCore = CeilDiv(outTotal, realCoreNum);
    }

    // 8. tileLength
    OP_CHECK_IF(dtypeSize <= 0, OP_LOGE(context, "invalid dtypeSize %ld", dtypeSize),
                return ge::GRAPH_FAILED);
    int64_t ubPerBuf = (ubSize - SIMT_DCACHE_SIZE) / DIAG_FLAT_BUFFER_NUM;
    int64_t tileLength = std::min<int64_t>({ubPerBuf / dtypeSize, outPerCore});

    // 9. Fill output struct
    out->numInput    = numInput;
    out->diagonal    = diagonal;
    out->outWidth    = outWidth;
    out->outTotal    = outTotal;
    out->outPerCore  = outPerCore;
    out->tileLength  = tileLength;
    out->realCoreNum = realCoreNum;
    out->localMemSize = static_cast<uint32_t>(ubSize - SIMT_DCACHE_SIZE);

    // 10. Set context-level params (common to both callers)
    auto ret = context->SetLocalMemorySize(out->localMemSize);
    OP_CHECK_IF(ret != ge::GRAPH_SUCCESS,
                OP_LOGE(context, "set local memory size failed."), return ret);

    context->SetBlockDim(realCoreNum);

    size_t* currentWorkspace = context->GetWorkspaceSizes(1);
    OP_CHECK_NULL_WITH_CONTEXT(context, currentWorkspace);
    currentWorkspace[0] = 0;

    return ge::GRAPH_SUCCESS;
}

// ============================================================================
// DiagFlat's own tiling entry (static wrapper)
// ============================================================================
static ge::graphStatus DiagFlatTilingFunc(gert::TilingContext* context)
{
    DiagFlatTilingOutput out;
    OP_CHECK_IF(
        TilingDiagFlatArch35(context, &out) != ge::GRAPH_SUCCESS,
        OP_LOGE(context, "TilingDiagFlatArch35 error"),
        return ge::GRAPH_FAILED);

    // Fill DiagFlat's own TilingData
    DiagFlatArch35TilingData* tiling = context->GetTilingData<DiagFlatArch35TilingData>();
    OP_CHECK_NULL_WITH_CONTEXT(context, tiling);
    OP_CHECK_IF(
        memset_s(tiling, sizeof(DiagFlatArch35TilingData), 0, sizeof(DiagFlatArch35TilingData)) != EOK,
        OP_LOGE(context, "set tiling data error"),
        return ge::GRAPH_FAILED);

    tiling->numInput    = out.numInput;
    tiling->diagonal    = out.diagonal;
    tiling->outWidth    = out.outWidth;
    tiling->outTotal    = out.outTotal;
    tiling->outPerCore  = out.outPerCore;
    tiling->tileLength  = out.tileLength;
    tiling->realCoreNum = out.realCoreNum;

    // DiagFlat's own TilingKey
    ASCENDC_TPL_SEL_PARAM(context, static_cast<uint32_t>(3501));

    return ge::GRAPH_SUCCESS;
}

// ============================================================================
// TilingParse
// ============================================================================
static ge::graphStatus TilingParseForDiagFlat([[maybe_unused]] gert::TilingParseContext* context)
{
    return ge::GRAPH_SUCCESS;
}

struct DiagFlatCompileInfo {};

// ============================================================================
// Tiling registration
// ============================================================================
IMPL_OP_OPTILING(DiagFlat)
    .Tiling(DiagFlatTilingFunc)
    .TilingParse<DiagFlatCompileInfo>(TilingParseForDiagFlat);

} // namespace optiling
