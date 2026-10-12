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
 * \file diag_tiling_arch35.cpp
 * \brief Tiling implementation for Diag operator (arch35)
 *      
 */

#include "diag_tiling_arch35.h"
#include <graph/utils/type_utils.h>
#include "log/log.h"
#include "util/math_util.h"
#include "platform/platform_ascendc.h"

namespace optiling {

static constexpr int64_t HALF_VL_LEN = 128L;
static constexpr size_t MIN_INPUT_DIM = 1;
static constexpr size_t MAX_INPUT_DIM = 4;

static void CalcSimtTiling(
    int64_t nSize, int32_t dtypeSize, int64_t coreNum, int64_t ubSize, DiagSimtTilingData* tilingData)
{
    int64_t sideLengthFactor = HALF_VL_LEN / dtypeSize;
    int64_t blockNum = Ops::Base::CeilDiv(nSize, sideLengthFactor);
    int64_t cores = std::min(coreNum, blockNum);

    int64_t mainFactor = Ops::Base::CeilDiv(nSize, cores);
    int64_t mainCount = (nSize % cores == 0) ? cores : (nSize % cores);
    int64_t tailFactor = nSize / cores;

    tilingData->nSize = nSize;
    tilingData->ubSize = ubSize;
    tilingData->realCoreNum = cores;
    tilingData->mainBlockCount = mainCount;
    tilingData->mainBlockFactor = mainFactor;
    tilingData->tailBlockFactor = tailFactor;
}

static ge::graphStatus TilingGetCompileInfo(gert::TilingContext* context, DiagCompileInfo* compileInfo)
{
    OP_LOGD(context->GetNodeName(), "Enter TilingGetCompileInfo.");

    auto platformInfo = context->GetPlatformInfo();
    OP_CHECK_NULL_WITH_CONTEXT(context, platformInfo);
    auto ascendcPlatform = platform_ascendc::PlatformAscendC(platformInfo);

    compileInfo->coreNum = ascendcPlatform.GetCoreNumAiv();
    if (compileInfo->coreNum == 0) {
        OP_LOGE_FOR_INVALID_VALUE_WITH_REASON(context->GetNodeName(), "coreNum",
            std::to_string(compileInfo->coreNum).c_str(),
            "The value of coreNum must be greater than 0.");
        return ge::GRAPH_FAILED;
    }

    uint64_t ubSize = 0;
    ascendcPlatform.GetCoreMemSize(platform_ascendc::CoreMemType::UB, ubSize);
    compileInfo->ubSize = static_cast<uint32_t>(ubSize);
    if (compileInfo->ubSize == 0) {
        OP_LOGE_FOR_INVALID_VALUE_WITH_REASON(context->GetNodeName(), "ubSize",
            std::to_string(compileInfo->ubSize).c_str(),
            "The value of ubSize must be greater than 0.");
        return ge::GRAPH_FAILED;
    }

    OP_LOGD(context->GetNodeName(), "Exit TilingGetCompileInfo.");
    return ge::GRAPH_SUCCESS;
}

static ge::graphStatus TilingCheckInputParams(
    gert::TilingContext* context, int64_t* nSize, int32_t* dtypeSize)
{
    OP_LOGD(context->GetNodeName(), "Enter TilingCheckInputParams.");

    auto input = context->GetInputTensor(0);
    OP_CHECK_NULL_WITH_CONTEXT(context, input);

    auto inputShape = input->GetStorageShape();
    if (inputShape.GetDimNum() < MIN_INPUT_DIM || inputShape.GetDimNum() > MAX_INPUT_DIM) {
        OP_LOGE_FOR_INVALID_SHAPEDIM_WITH_REASON(context->GetNodeName(), "input",
            std::to_string(inputShape.GetDimNum()).c_str(),
            "The dimNum of input must be within the range [1, 4].");
        return ge::GRAPH_FAILED;
    }

    auto dataType = input->GetDataType();
    *dtypeSize = ge::GetSizeByDataType(dataType);
    if (*dtypeSize <= 0) {
        OP_LOGE_FOR_INVALID_DTYPE_WITH_REASON(context->GetNodeName(), "input",
            ge::TypeUtils::DataTypeToSerialString(dataType).c_str(),
            "The dtype size of input must be greater than 0.");
        return ge::GRAPH_FAILED;
    }

    *nSize = inputShape.GetShapeSize();

    OP_LOGD(context->GetNodeName(), "Exit TilingCheckInputParams.");
    return ge::GRAPH_SUCCESS;
}

static ge::graphStatus TilingEmptyTensor(
    gert::TilingContext* context, int64_t ubSize, DiagSimtTilingData* tilingData)
{
    tilingData->nSize = 0;
    tilingData->ubSize = ubSize;
    tilingData->realCoreNum = 1;
    tilingData->mainBlockCount = 0;
    tilingData->mainBlockFactor = 0;
    tilingData->tailBlockFactor = 0;
    context->SetBlockDim(1);
    context->SetTilingKey(TILING_SIMT);
    size_t sysWorkspaceSize = 16 * 1024 * 1024;
    size_t* currentWorkspace = context->GetWorkspaceSizes(1);
    OP_CHECK_NULL_WITH_CONTEXT(context, currentWorkspace);
    currentWorkspace[0] = sysWorkspaceSize;
    OP_LOGI(context->GetNodeName(), "Diag tiling: empty tensor, nSize=0, realCoreNum=1");
    return ge::GRAPH_SUCCESS;
}

static ge::graphStatus Tiling4Diag(gert::TilingContext* context)
{
    OP_LOGD(context->GetNodeName(), "Tiling4Diag running begin");

    DiagCompileInfo compileInfo;
    auto ret = TilingGetCompileInfo(context, &compileInfo);
    if (ret != ge::GRAPH_SUCCESS) {
        return ret;
    }

    int64_t nSize = 0;
    int32_t dtypeSize = 0;
    ret = TilingCheckInputParams(context, &nSize, &dtypeSize);
    if (ret != ge::GRAPH_SUCCESS) {
        return ret;
    }

    int64_t coreNum = static_cast<int64_t>(compileInfo.coreNum);
    int64_t ubSize = static_cast<int64_t>(compileInfo.ubSize);

    auto* tilingData = context->GetTilingData<DiagSimtTilingData>();
    OP_CHECK_NULL_WITH_CONTEXT(context, tilingData);

    if (nSize == 0) {
        return TilingEmptyTensor(context, ubSize, tilingData);
    }

    CalcSimtTiling(nSize, dtypeSize, coreNum, ubSize, tilingData);
    context->SetBlockDim(static_cast<uint32_t>(tilingData->realCoreNum));
    context->SetTilingKey(TILING_SIMT);

    size_t sysWorkspaceSize = 16 * 1024 * 1024;
    size_t* currentWorkspace = context->GetWorkspaceSizes(1);
    OP_CHECK_NULL_WITH_CONTEXT(context, currentWorkspace);
    currentWorkspace[0] = sysWorkspaceSize;

    OP_LOGI(
        context->GetNodeName(),
        "Diag tiling: nSize=%ld, batchSize=1, realCoreNum=%ld, mainBlockCount=%ld, mainBlockFactor=%ld, "
        "tailBlockFactor=%ld",
        tilingData->nSize, tilingData->realCoreNum, tilingData->mainBlockCount, tilingData->mainBlockFactor,
        tilingData->tailBlockFactor);

    return ge::GRAPH_SUCCESS;
}

static ge::graphStatus TilingPrepare4Diag(gert::TilingParseContext* context)
{
    auto compileInfo = context->GetCompiledInfo<DiagCompileInfo>();
    OP_CHECK_NULL_WITH_CONTEXT(context, compileInfo);
    auto platformInfo = context->GetPlatformInfo();
    OP_CHECK_NULL_WITH_CONTEXT(context, platformInfo);
    auto ascendcPlatform = platform_ascendc::PlatformAscendC(platformInfo);

    compileInfo->coreNum = ascendcPlatform.GetCoreNumAiv();
    if (compileInfo->coreNum == 0) {
        OP_LOGE_FOR_INVALID_VALUE_WITH_REASON(context->GetNodeName(), "coreNum",
            std::to_string(compileInfo->coreNum).c_str(),
            "The value of coreNum must be greater than 0.");
        return ge::GRAPH_FAILED;
    }

    uint64_t ubSize = 0;
    ascendcPlatform.GetCoreMemSize(platform_ascendc::CoreMemType::UB, ubSize);
    compileInfo->ubSize = static_cast<uint32_t>(ubSize);
    if (compileInfo->ubSize == 0) {
        OP_LOGE_FOR_INVALID_VALUE_WITH_REASON(context->GetNodeName(), "ubSize",
            std::to_string(compileInfo->ubSize).c_str(),
            "The value of ubSize must be greater than 0.");
        return ge::GRAPH_FAILED;
    }

    return ge::GRAPH_SUCCESS;
}

IMPL_OP_OPTILING(Diag).Tiling(Tiling4Diag).TilingParse<DiagCompileInfo>(TilingPrepare4Diag);
} // namespace optiling
