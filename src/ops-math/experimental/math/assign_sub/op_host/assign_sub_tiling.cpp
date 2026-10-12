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
 * \file assign_sub_tiling.cpp
 * \brief
 */

#include "register/op_def_registry.h"
#include "op_common/log/log.h"
#include "op_common/op_host/util/math_util.h"
#include "op_common/op_host/util/platform_util.h"
#include "../op_kernel/assign_sub_tiling_data.h"
#include "../op_kernel/assign_sub_tiling_key.h"

namespace optiling {

using Ops::Base::CeilAlign;
using Ops::Base::CeilDiv;
using Ops::Base::FloorAlign;
using Ops::Base::FloorDiv;
using Ops::Base::GetUbBlockSize;

constexpr uint32_t WS_SYS_SIZE = 0U;
constexpr int64_t COARSE_ALIGN_BYTES = 512;

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

static ge::graphStatus GetWorkspaceSize(gert::TilingContext* context)
{
    size_t* currentWorkspace = context->GetWorkspaceSizes(1);
    OP_CHECK_NULL_WITH_CONTEXT(context, currentWorkspace);
    currentWorkspace[0] = WS_SYS_SIZE;
    return ge::GRAPH_SUCCESS;
}

static ge::graphStatus GetDtypeInfo(gert::TilingContext* context, ge::DataType dtype, int64_t& dtypeSize,
                                    int64_t& perElemBytes, uint64_t& tilingKey)
{
    constexpr int64_t bufferNum = 2;
    constexpr int64_t queueNum = 3;
    switch (dtype) {
        case ge::DT_FLOAT16:
            dtypeSize = 2;
            perElemBytes = queueNum * bufferNum * dtypeSize;
            tilingKey = GET_TPL_TILING_KEY(ASSIGNSUB_TPL_SCH_MODE_0);
            break;
        case ge::DT_INT8:
            dtypeSize = 1;
            perElemBytes = queueNum * bufferNum * dtypeSize + 2 * static_cast<int64_t>(sizeof(uint16_t));
            tilingKey = GET_TPL_TILING_KEY(ASSIGNSUB_TPL_SCH_MODE_1);
            break;
        case ge::DT_FLOAT:
            dtypeSize = 4;
            perElemBytes = queueNum * bufferNum * dtypeSize;
            tilingKey = GET_TPL_TILING_KEY(ASSIGNSUB_TPL_SCH_MODE_2);
            break;
        case ge::DT_INT32:
            dtypeSize = 4;
            perElemBytes = queueNum * bufferNum * dtypeSize;
            tilingKey = GET_TPL_TILING_KEY(ASSIGNSUB_TPL_SCH_MODE_3);
            break;
        case ge::DT_UINT8:
            dtypeSize = 1;
            perElemBytes = queueNum * bufferNum * dtypeSize + 2 * static_cast<int64_t>(sizeof(uint16_t));
            tilingKey = GET_TPL_TILING_KEY(ASSIGNSUB_TPL_SCH_MODE_4);
            break;
        case ge::DT_BF16:
            dtypeSize = 2;
            perElemBytes = queueNum * bufferNum * dtypeSize + 2 * static_cast<int64_t>(sizeof(float));
            tilingKey = GET_TPL_TILING_KEY(ASSIGNSUB_TPL_SCH_MODE_5);
            break;
        case ge::DT_INT64:
            dtypeSize = 8;
            perElemBytes = queueNum * bufferNum * dtypeSize + 2 * static_cast<int64_t>(sizeof(int32_t));
            tilingKey = GET_TPL_TILING_KEY(ASSIGNSUB_TPL_SCH_MODE_6);
            break;
        default:
            OP_LOGE(context, "unsupported dtype %d", static_cast<int32_t>(dtype));
            return ge::GRAPH_FAILED;
    }
    return ge::GRAPH_SUCCESS;
}

static ge::graphStatus AssignSubTilingFunc(gert::TilingContext* context)
{
    uint64_t ubSize;
    int64_t coreNum;
    OP_CHECK_IF(GetPlatformInfo(context, ubSize, coreNum) != ge::GRAPH_SUCCESS,
                OP_LOGE(context, "GetPlatformInfo error"), return ge::GRAPH_FAILED);

    OP_CHECK_IF(GetWorkspaceSize(context) != ge::GRAPH_SUCCESS, OP_LOGE(context, "GetWorkspaceSize error"),
                return ge::GRAPH_FAILED);

    auto inputShape = context->GetInputShape(0);
    OP_CHECK_NULL_WITH_CONTEXT(context, inputShape);
    int64_t totalNum = inputShape->GetStorageShape().GetShapeSize();

    auto inputDesc = context->GetInputDesc(0);
    OP_CHECK_NULL_WITH_CONTEXT(context, inputDesc);
    int64_t dtypeSize = 1;
    int64_t perElemBytes = 1;
    uint64_t tilingKey = GET_TPL_TILING_KEY(ASSIGNSUB_TPL_SCH_MODE_1);
    OP_CHECK_IF(
        GetDtypeInfo(context, inputDesc->GetDataType(), dtypeSize, perElemBytes, tilingKey) != ge::GRAPH_SUCCESS,
        OP_LOGE(context, "GetDtypeInfo error"), return ge::GRAPH_FAILED);

    AssignSubTilingData* tiling = context->GetTilingData<AssignSubTilingData>();
    OP_CHECK_NULL_WITH_CONTEXT(context, tiling);

    int64_t alignNum = GetUbBlockSize(context) / dtypeSize;
    if (alignNum <= 0) {
        alignNum = 1;
    }

    int64_t blockFactor = 1;
    int64_t ubFactor = alignNum;
    int64_t usedCoreNum = 1;

    if (totalNum <= 0) {
        tiling->totalNum = 0;
        tiling->blockFactor = alignNum;
        tiling->ubFactor = alignNum;
        context->SetBlockDim(1);
        context->SetTilingKey(tilingKey);
        return ge::GRAPH_SUCCESS;
    }

    int64_t perCore = CeilDiv(totalNum, coreNum);
    blockFactor = CeilAlign(perCore, alignNum);
    if (blockFactor < alignNum) {
        blockFactor = alignNum;
    }
    usedCoreNum = CeilDiv(totalNum, blockFactor);
    if (usedCoreNum < 1) {
        usedCoreNum = 1;
    }

    int64_t coarseAlign = COARSE_ALIGN_BYTES / dtypeSize;
    if (coarseAlign > alignNum) {
        int64_t blockFactorCoarse = CeilAlign(perCore, coarseAlign);
        int64_t usedCoreCoarse = CeilDiv(totalNum, blockFactorCoarse);
        if (usedCoreCoarse == usedCoreNum) {
            blockFactor = blockFactorCoarse;
        }
    }

    int64_t ubCapacity = static_cast<int64_t>(ubSize) / perElemBytes;
    ubFactor = FloorAlign(ubCapacity, alignNum);
    if (ubFactor < alignNum) {
        ubFactor = alignNum;
    }
    if (ubFactor > blockFactor) {
        ubFactor = blockFactor;
    }

    tiling->totalNum = totalNum;
    tiling->blockFactor = blockFactor;
    tiling->ubFactor = ubFactor;

    context->SetBlockDim(usedCoreNum);
    context->SetTilingKey(tilingKey);
    return ge::GRAPH_SUCCESS;
}

static ge::graphStatus TilingParseForAssignSub([[maybe_unused]] gert::TilingParseContext* context)
{
    return ge::GRAPH_SUCCESS;
}

struct AssignSubCompileInfo {};

IMPL_OP_OPTILING(AssignSub).Tiling(AssignSubTilingFunc).TilingParse<AssignSubCompileInfo>(TilingParseForAssignSub);

} // namespace optiling
