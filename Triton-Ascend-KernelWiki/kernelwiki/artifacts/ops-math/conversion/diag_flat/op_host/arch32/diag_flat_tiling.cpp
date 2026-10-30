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
 * \file diag_flat_tiling.cpp
 * \brief
 */

#include "../../../diag_v2/op_host/arch32/diag_v2_tiling.h"
#include "util/math_util.h"
#include "log/log.h"
#include "platform/platform_ascendc.h"
#include "platform/platform_infos_def.h"

namespace optiling {

static const int32_t B16_INPUT_LENGTH = 16;

static ge::graphStatus DiagFlatSetTilingData(gert::TilingContext* context, DiagV2TilingData& tilingData)
{
    tilingData.SaveToBuffer(context->GetRawTilingData()->GetData(), context->GetRawTilingData()->GetCapacity());
    context->GetRawTilingData()->SetDataSize(tilingData.GetDataSize());
    
    return ge::GRAPH_SUCCESS;
}

static inline ge::graphStatus CalcScalarTiling(const gert::TilingContext* context, DiagV2TilingData& tilingData)
{
    tilingData.set_totalCoreNum(tilingData.get_totalCoreNum());
    tilingData.set_usedCoreNum(1);
    tilingData.set_normalCoreHandleNum(tilingData.get_inputNum());
    tilingData.set_lastCoreHandleNum(0);

    auto input = context->GetInputTensor(0);
    auto dataType = input->GetDataType();
    int32_t dtypeSize = ge::GetSizeByDataType(dataType);
    if (dtypeSize != B16_INPUT_LENGTH) {
        tilingData.set_tilingKey(SCALAR_UNEQUAL_SIZE16);
    } else {
        tilingData.set_tilingKey(SCALAR_EQUAL_SIZE16);
    }
    return ge::GRAPH_SUCCESS;
}

static inline ge::graphStatus CalcAuxMatrixTiling(const gert::TilingContext* context, DiagV2TilingData& tilingData)
{
    int64_t tmpNum =
        std::max(Ops::Base::CeilDiv(tilingData.get_inputNum(), tilingData.get_totalCoreNum()), LEAST_NUM_PER_CORE);
    tmpNum = Ops::Base::CeilAlign(tmpNum, LEAST_NUM_PER_CORE);
    tilingData.set_usedCoreNum(Ops::Base::CeilDiv(tilingData.get_inputNum(), tmpNum));
    tilingData.set_normalCoreHandleNum(tmpNum);
    tilingData.set_lastCoreHandleNum(
        tilingData.get_inputNum() - (tilingData.get_usedCoreNum() - 1) * tilingData.get_normalCoreHandleNum());

    auto input = context->GetInputTensor(0);
    auto dataType = input->GetDataType();
    int32_t dtypeSize = ge::GetSizeByDataType(dataType);
    if (dtypeSize != B16_INPUT_LENGTH) {
        tilingData.set_tilingKey(ASSIST_UNEQUAL_SIZE16);
    } else {
        tilingData.set_tilingKey(ASSIST_EQUAL_SIZE16);
    }
    return ge::GRAPH_SUCCESS;
}

static void PrintTilingData(DiagV2TilingData& tilingData)
{
    OP_LOGI("---[DiagFlat]---", "normalCoreHandleNum: %ld", tilingData.get_normalCoreHandleNum());
    OP_LOGI("---[DiagFlat]---", "lastCoreHandleNum: %ld", tilingData.get_lastCoreHandleNum());
    OP_LOGI("---[DiagFlat]---", "totalCoreNum: %ld", tilingData.get_totalCoreNum());
    OP_LOGI("---[DiagFlat]---", "usedCoreNum: %ld", tilingData.get_usedCoreNum());
    OP_LOGI("---[DiagFlat]---", "inputNum: %ld", tilingData.get_inputNum());
    OP_LOGI("---[DiagFlat]---", "diagonal: %ld", tilingData.get_diagonal());
    OP_LOGI("---[DiagFlat]---", "ubInputSize: %ld", tilingData.get_ubInputSize());
    OP_LOGI("---[DiagFlat]---", "ubOutputSize: %ld", tilingData.get_ubOutputSize());
    OP_LOGI("---[DiagFlat]---", "workspaceSize: %ld", tilingData.get_workspaceSize());
    OP_LOGI("---[DiagFlat]---", "tilingKey: %ld", tilingData.get_tilingKey());
}

ge::graphStatus TilingDiagFlat(gert::TilingContext* context)
{
    OP_LOGD(context->GetNodeName(), "TilingDiagFlat running begin");
    auto input = context->GetInputTensor(0);
    OP_CHECK_NULL_WITH_CONTEXT(context, input);
    auto inputShape = input->GetStorageShape();
    DiagV2TilingData tilingData;
    auto attrs = context->GetAttrs();
    OP_CHECK_NULL_WITH_CONTEXT(context, attrs);
    const int64_t* diagonalPtr = attrs->GetAttrPointer<int64_t>(0);
    OP_CHECK_NULL_WITH_CONTEXT(context, diagonalPtr);
    const int64_t diagonal = *diagonalPtr;
    tilingData.set_diagonal(diagonal);
    // set coreNum
    auto compileInfo = reinterpret_cast<const DiagV2CompileInfo*>(context->GetCompileInfo());
    OP_CHECK_NULL_WITH_CONTEXT(context, compileInfo);
    int64_t totalCoreNum = static_cast<int64_t>(compileInfo->totalCoreNum);
    tilingData.set_totalCoreNum(totalCoreNum);
    // set ubSize
    int64_t ubSize = compileInfo->ubSizePlatForm;
    auto dataType = input->GetDataType();
    auto auxMatrix = SCALAR_THRESHOLD_NUM * SCALAR_THRESHOLD_NUM * sizeof(dataType);
    auto outputDataSize = SCALAR_THRESHOLD_NUM * SCALAR_THRESHOLD_NUM * sizeof(dataType);
    auto inputDataSize = ubSize - auxMatrix - outputDataSize;
    tilingData.set_ubInputSize(inputDataSize);
    tilingData.set_ubOutputSize(outputDataSize);
    // set inputNum
    tilingData.set_inputNum(inputShape.GetShapeSize());
    if (tilingData.get_inputNum() <= SCALAR_THRESHOLD_NUM) {
        OP_CHECK_IF(
            CalcScalarTiling(context, tilingData) != ge::GRAPH_SUCCESS,
            OP_LOGE(context->GetNodeName(), "CalcScalarTiling fail."), return ge::GRAPH_FAILED);
    } else {
        OP_CHECK_IF(
            CalcAuxMatrixTiling(context, tilingData) != ge::GRAPH_SUCCESS,
            OP_LOGE(context->GetNodeName(), "CalcAuxMatrixTiling fail."), return ge::GRAPH_FAILED);
    }
    size_t userSize = tilingData.get_totalCoreNum() * 32;
    size_t sysWorkspaceSize = 16 * 1024 * 1024;
    size_t* currentWorkspace = context->GetWorkspaceSizes(1);
    currentWorkspace[0] = userSize + sysWorkspaceSize;
    tilingData.set_workspaceSize(currentWorkspace[0]);
    // set tilingData
    OP_CHECK_IF(
        DiagFlatSetTilingData(context, tilingData) != ge::GRAPH_SUCCESS,
        OP_LOGE(context->GetNodeName(), "DiagFlatSetTilingData set tiling data fail."), return ge::GRAPH_FAILED);
    context->SetBlockDim(tilingData.get_totalCoreNum());
    context->SetTilingKey(tilingData.get_tilingKey());
    context->SetScheduleMode(1); // SyncAll need set
    PrintTilingData(tilingData);
    return ge::GRAPH_SUCCESS;
}

static ge::graphStatus TilingPrepareForDiagFlat(gert::TilingParseContext* context)
{
    auto compileInfo = context->GetCompiledInfo<DiagV2CompileInfo>();
    OP_CHECK_NULL_WITH_CONTEXT(context, compileInfo);
    auto platformInfo = context->GetPlatformInfo();
    OP_CHECK_NULL_WITH_CONTEXT(context, platformInfo);
    auto ascendcPlatform = platform_ascendc::PlatformAscendC(platformInfo);
    compileInfo->totalCoreNum = ascendcPlatform.GetCoreNumAiv();
    OP_CHECK_IF(
        (compileInfo->totalCoreNum <= 0),
        OP_LOGE(context->GetNodeName(), "TilingPrepare4DiagFlat fail to get core num."), return ge::GRAPH_FAILED);

    uint64_t ubSizePlatForm;
    ascendcPlatform.GetCoreMemSize(platform_ascendc::CoreMemType::UB, ubSizePlatForm);
    compileInfo->ubSizePlatForm = static_cast<int64_t>(ubSizePlatForm);
    OP_CHECK_IF(
        (compileInfo->ubSizePlatForm <= 0),
        OP_LOGE(context->GetNodeName(), "TilingPrepare4DiagFlat fail to get ub size."), return ge::GRAPH_FAILED);

    return ge::GRAPH_SUCCESS;
}

// 向框架注册入口函数
IMPL_OP_OPTILING(DiagFlat).Tiling(TilingDiagFlat).TilingParse<DiagV2CompileInfo>(TilingPrepareForDiagFlat);
} // namespace optiling
