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
 * \file abs_grad_tiling_arch35.cpp
 * \brief
 */
#include <iostream>

#include "abs_grad_tiling_arch35.h"
#include <graph/utils/type_utils.h>
#include "tiling/tiling_api.h"
#include "tiling/platform/platform_ascendc.h"
#include "register/op_def_registry.h"
#include "register/tilingdata_base.h"
#include "../../op_kernel/arch35/abs_grad_dag.h"

using namespace ge;
using namespace AbsGradNs;

namespace optiling {
constexpr uint64_t ABS_GRAD_TILING_KEY_ELEMENTWISE = 101;
constexpr uint64_t ABS_GRAD_WORKSPACE_RESERVE_BYTE = 8 * 1024;
ge::graphStatus AbsGradTiling::SetTilingData()
{
    size_t* currentWorkspace = tilingContext->GetWorkspaceSizes(1);
    OP_CHECK_NULL_WITH_CONTEXT(tilingContext, currentWorkspace);
    currentWorkspace[0] = ABS_GRAD_WORKSPACE_RESERVE_BYTE;
    tilingContext->SetTilingKey(ABS_GRAD_TILING_KEY_ELEMENTWISE);
    tilingContext->SetBlockDim(tiling->baseTiling.blockNum);
    return ge::GRAPH_SUCCESS;
}

ge::graphStatus AbsGradTiling::CalcOutputDtype()
{
    auto inputDescY = tilingContext->GetInputDesc(0);
    OP_CHECK_NULL_WITH_CONTEXT(tilingContext, inputDescY);
    this->inputDtype = inputDescY->GetDataType();
    auto inputDescDy = tilingContext->GetInputDesc(1);
    OP_CHECK_NULL_WITH_CONTEXT(tilingContext, inputDescDy);
    ge::DataType inputDtypeDy = inputDescDy->GetDataType();
    auto outputDesc = tilingContext->GetOutputDesc(0);
    OP_CHECK_NULL_WITH_CONTEXT(tilingContext, outputDesc);
    this->outputDtype = outputDesc->GetDataType();
    if (this->inputDtype != ge::DT_FLOAT16 && this->inputDtype != ge::DT_FLOAT && this->inputDtype != ge::DT_BF16) {
        OP_LOGE(tilingContext, "AbsGrad op input y dtype must be float16 or float or bfloat16, check fail");
        return ge::GRAPH_FAILED;
    }
    if (inputDtypeDy != ge::DT_FLOAT16 && inputDtypeDy != ge::DT_FLOAT && inputDtypeDy != ge::DT_BF16) {
        OP_LOGE(tilingContext, "AbsGrad op input dy dtype must be float16 or float or bfloat16, check fail");
        return ge::GRAPH_FAILED;
    }
    OP_CHECK_IF(this->inputDtype != inputDtypeDy || this->inputDtype != this->outputDtype,
                OP_LOGE(tilingContext, "AbsGrad op all inputs and outputs dtype must be consistent, check fail"),
                return ge::GRAPH_FAILED);
    return ge::GRAPH_SUCCESS;
}

ge::graphStatus AbsGradTiling::RunTiling()
{
    ElewiseBaseTiling elewiseBaseTiling(tilingContext);
    OP_CHECK_IF(CalcOutputDtype() == ge::GRAPH_FAILED, OP_LOGE(tilingContext, "get output dtype failed"),
                return ge::GRAPH_FAILED);
    ge::graphStatus res = ge::GRAPH_FAILED;
    tiling = tilingContext->GetTilingData<AbsGradTilingData>();
    OP_CHECK_NULL_WITH_CONTEXT(tilingContext, tiling);
    if (this->inputDtype == ge::DT_FLOAT16) {
        res = elewiseBaseTiling.DoTiling<AbsGradOp::AbsGradDag<half>::OpDag>(tiling->baseTiling);
    } else if (this->inputDtype == ge::DT_FLOAT) {
        res = elewiseBaseTiling.DoTiling<AbsGradOp::AbsGradDag<float>::OpDag>(tiling->baseTiling);
    } else if (this->inputDtype == ge::DT_BF16) {
        res = elewiseBaseTiling.DoTiling<AbsGradOp::AbsGradDag<bfloat16_t>::OpDag>(tiling->baseTiling);
    } else {
        OP_LOGE(tilingContext, "data type check failed. dtype:%u", static_cast<uint32_t>(this->inputDtype));
    }
    OP_CHECK_IF(res == ge::GRAPH_FAILED, OP_LOGE(tilingContext, "DoTiling failed"), return ge::GRAPH_FAILED);
    ge::graphStatus result = SetTilingData();
    return result;
}

static ge::graphStatus TilingForAbsGrad(gert::TilingContext* context)
{
    OP_LOGD("AbsGradTiling", "Enter TilingForAbsGrad");
    OP_CHECK_IF(context == nullptr, OP_LOGE(context, "Tiling context is null"), return ge::GRAPH_FAILED);
    OP_LOGD("AbsGradTiling", "Enter new AbsGradTiling");
    AbsGradTiling absgradTiling(context);
    return absgradTiling.RunTiling();
}

ge::graphStatus TilingPrepareForAbsGrad(gert::TilingParseContext* context)
{
    auto compileInfoPtr = context->GetCompiledInfo<AbsGradCompileInfo>();
    OP_CHECK_NULL_WITH_CONTEXT(context, compileInfoPtr);
    fe::PlatFormInfos* platformInfoPtr = context->GetPlatformInfo();
    OP_CHECK_NULL_WITH_CONTEXT(context, platformInfoPtr);
    auto ascendcPlatform = platform_ascendc::PlatformAscendC(platformInfoPtr);
    compileInfoPtr->coreNum = ascendcPlatform.GetCoreNumAiv();
    ascendcPlatform.GetCoreMemSize(platform_ascendc::CoreMemType::UB, compileInfoPtr->ubSize);
    return ge::GRAPH_SUCCESS;
}

IMPL_OP_OPTILING(AbsGrad).Tiling(TilingForAbsGrad).TilingParse<AbsGradCompileInfo>(TilingPrepareForAbsGrad);
} // namespace optiling
