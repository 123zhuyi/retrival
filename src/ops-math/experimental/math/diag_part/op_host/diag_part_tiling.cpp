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
 * \file diag_part_tiling.cpp
 * \brief
 */

#include "log/log.h"
#include "util/math_util.h"
#include "util/platform_util.h"
#include "tiling/platform/platform_ascendc.h"
#include "register/op_impl_registry.h"
#include "../op_kernel/diag_part_tiling_data.h"
#include "../op_kernel/diag_part_tiling_key.h"

namespace optiling {

struct DiagPartCompileInfo {};

static ge::graphStatus TilingParseForDiagPart([[maybe_unused]] gert::TilingParseContext* context)
{
    OP_CHECK_IF(context == nullptr, OP_LOGE(context, "context is nullptr"), return ge::GRAPH_FAILED);
    return ge::GRAPH_SUCCESS;
}

static ge::graphStatus GetWorkspaceSize(gert::TilingContext* context)
{
    OP_CHECK_IF(context == nullptr, OP_LOGE(context, "context is nullptr"), return ge::GRAPH_FAILED);
    size_t usrSize = 0;
    auto ascendcPlatform = platform_ascendc::PlatformAscendC(context->GetPlatformInfo());
    uint32_t sysWorkspaceSize = ascendcPlatform.GetLibApiWorkSpaceSize();
    size_t* currentWorkspace = context->GetWorkspaceSizes(1);
    currentWorkspace[0] = usrSize + sysWorkspaceSize;
    return ge::GRAPH_SUCCESS;
}

static ge::graphStatus Tiling4DiagPart(gert::TilingContext* context)
{
    OP_LOGD(context->GetNodeName(), "Tiling4DiagPart running begin");

    OP_CHECK_IF(context == nullptr, OP_LOGE(context, "context is nullptr"), return ge::GRAPH_FAILED);

    // Get platform info
    auto ascendcPlatform = platform_ascendc::PlatformAscendC(context->GetPlatformInfo());
    int64_t coreNum = ascendcPlatform.GetCoreNum();
    OP_CHECK_IF(coreNum == 0, OP_LOGE(context, "coreNum is 0"), return ge::GRAPH_FAILED);

    // Get workspace
    if (GetWorkspaceSize(context) != ge::GRAPH_SUCCESS) {
        OP_LOGE(context, "GetWorkspaceSize error");
        return ge::GRAPH_FAILED;
    }

    // Calculate sideLength = product of first half of input dims
    auto xShape = context->GetInputShape(0)->GetStorageShape();
    auto xDimNum = xShape.GetDimNum();
    OP_CHECK_IF(xDimNum % 2 != 0, OP_LOGE(context, "Input x dimNum must be even"), return ge::GRAPH_FAILED);

    // Extract dtype from input descriptor
    auto inputDesc = context->GetInputDesc(0);
    ge::DataType dtype = inputDesc->GetDataType();
    // Convert ge::DataType to tiling dtype value
    uint32_t tilingDtype = (dtype == ge::DT_INT32) ? 3 : (dtype == ge::DT_FLOAT16) ? 1 : 0;

    uint64_t sideLength = 1;
    uint64_t halfDimNum = xDimNum / 2;
    for (uint64_t i = 0; i < halfDimNum; i++) {
        sideLength *= xShape.GetDim(i);
    }

    // NPU vector engine requires 32-byte minimum transfers
    // float/int32: 8 elements, float16: 16 elements
    int32_t alignSize = 8;
    if (dtype == ge::DT_FLOAT16) {
        alignSize = 16;
    }

    uint64_t totalElements = sideLength;
    // numPerCore = sub-block size for Gather-based diagonal extraction
    // Must match ALIGN in the kernel
    uint64_t numPerCore = static_cast<uint64_t>(alignSize);
    // tailNum = remaining elements in the last sub-block
    uint64_t tailNum = totalElements % numPerCore;

    uint64_t totalBlocks = (totalElements + numPerCore - 1) / numPerCore;

    uint64_t actualCoreNum = static_cast<uint64_t>(coreNum);
    if (totalBlocks < actualCoreNum) {
        actualCoreNum = totalBlocks;
    }

    // Set tiling data
    DiagPartTilingData* tiling = context->GetTilingData<DiagPartTilingData>();
    OP_CHECK_NULL_WITH_CONTEXT(context, tiling);

    tiling->sideLength = sideLength;
    tiling->dtype = tilingDtype;
    tiling->realCoreNum = actualCoreNum;
    tiling->numPerCore = numPerCore; // sub-block size (ALIGN)
    tiling->tailNum = tailNum;       // remaining elements in last sub-block
    tiling->blockSize = 0;           // unused

    context->SetBlockDim(actualCoreNum);
    context->SetTilingKey(GET_TPL_TILING_KEY(ELEMENTWISE_TPL_SCH_MODE_0, tilingDtype));

    OP_LOGD(context->GetNodeName(), "Tiling4DiagPart running end");
    return ge::GRAPH_SUCCESS;
}

IMPL_OP_OPTILING(DiagPart).Tiling(Tiling4DiagPart).TilingParse<DiagPartCompileInfo>(TilingParseForDiagPart);

} // namespace optiling
