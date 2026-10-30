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
 * \file complex_abs_tiling_arch35.cpp
 * \brief
 */
#include <iostream>

#include "complex_abs_tiling_arch35.h"
#include <graph/utils/type_utils.h>
#include "tiling/tiling_api.h"
#include "tiling/platform/platform_ascendc.h"
#include "register/op_def_registry.h"
#include "register/tilingdata_base.h"
#include "../../op_kernel/arch35/complex_abs_dag.h"

using namespace ge;
using namespace ComplexAbsNs;

namespace optiling {
constexpr uint64_t COMPLEX_ABS_TILING_KEY_ELEMENTWISE_COMPLEX = 101;
constexpr uint64_t COMPLEX_ABS_WORKSPACE_RESERVE_BYTE = 8 * 1024;

ge::graphStatus ComplexAbsTiling::SetTilingData()
{
    size_t* currentWorkspace = tilingContext->GetWorkspaceSizes(1);
    OP_CHECK_NULL_WITH_CONTEXT(tilingContext, currentWorkspace);
    currentWorkspace[0] = COMPLEX_ABS_WORKSPACE_RESERVE_BYTE;
    if (this->inputDtype == ge::DT_COMPLEX64 || this->inputDtype == ge::DT_COMPLEX32) {
        tilingContext->SetTilingKey(COMPLEX_ABS_TILING_KEY_ELEMENTWISE_COMPLEX);
    }
    tilingContext->SetBlockDim(tiling->baseTiling.blockNum);
    return ge::GRAPH_SUCCESS;
}

ge::graphStatus ComplexAbsTiling::CalcOutputDtype()
{
    auto inputDesc = tilingContext->GetInputDesc(0);
    OP_CHECK_NULL_WITH_CONTEXT(tilingContext, inputDesc);
    this->inputDtype = inputDesc->GetDataType();

    auto outputDesc = tilingContext->GetOutputDesc(0);
    OP_CHECK_NULL_WITH_CONTEXT(tilingContext, outputDesc);
    this->outputDtype = outputDesc->GetDataType();

    if (inputDtype == ge::DT_COMPLEX64) {
        OP_CHECK_IF(this->outputDtype != ge::DT_FLOAT,
                    OP_LOGE(tilingContext, "ComplexAbs op complex64 input must output float, check fail"),
                    return ge::GRAPH_FAILED);
    } else if (inputDtype == ge::DT_COMPLEX32) {
        OP_CHECK_IF(this->outputDtype != ge::DT_FLOAT16,
                    OP_LOGE(tilingContext, "ComplexAbs op complex32 input must output float16, check fail"),
                    return ge::GRAPH_FAILED);
    }
    return ge::GRAPH_SUCCESS;
}

ge::graphStatus ComplexAbsTiling::RunTiling()
{
    ElewiseBaseTiling elewiseBaseTiling(tilingContext);
    OP_CHECK_IF(CalcOutputDtype() == ge::GRAPH_FAILED, OP_LOGE(tilingContext, "get output dtype failed"),
                return ge::GRAPH_FAILED);

    ge::graphStatus res = ge::GRAPH_FAILED;
    tiling = tilingContext->GetTilingData<ComplexAbsTilingData>();
    OP_CHECK_NULL_WITH_CONTEXT(tilingContext, tiling);
    if (this->inputDtype == ge::DT_COMPLEX64) {
        res = elewiseBaseTiling.DoTiling<ComplexAbsOp::ComplexAbsDag<int64_t, float>::OpDag>(tiling->baseTiling);
    } else if (this->inputDtype == ge::DT_COMPLEX32) {
        res = elewiseBaseTiling.DoTiling<ComplexAbsOp::ComplexAbsDag<int32_t, half>::OpDag>(tiling->baseTiling);
    } else {
        OP_LOGE(tilingContext, "data type check failed. getype:%u", static_cast<uint32_t>(this->inputDtype));
        return ge::GRAPH_FAILED;
    }

    OP_CHECK_IF(res == ge::GRAPH_FAILED, OP_LOGE(tilingContext, "DoTiling failed"), return ge::GRAPH_FAILED);

    ge::graphStatus result = SetTilingData();
    return result;
}

static ge::graphStatus TilingForComplexAbs(gert::TilingContext* context)
{
    OP_LOGD("ComplexAbsTiling", "Enter TilingForComplexAbs");
    OP_CHECK_IF(context == nullptr, OP_LOGE(context, "Tiling context is null"), return ge::GRAPH_FAILED);

    // 走新的模板tiling
    OP_LOGD("ComplexAbsTiling", "Enter new ComplexAbsTiling");
    ComplexAbsTiling complexabsTiling(context);
    return complexabsTiling.RunTiling();
}

ge::graphStatus TilingPrepareForComplexAbs(gert::TilingParseContext* context)
{
    auto compileInfoPtr = context->GetCompiledInfo<ComplexAbsCompileInfo>();
    OP_CHECK_NULL_WITH_CONTEXT(context, compileInfoPtr);
    fe::PlatFormInfos* platformInfoPtr = context->GetPlatformInfo();
    OP_CHECK_NULL_WITH_CONTEXT(context, platformInfoPtr);
    auto ascendcPlatform = platform_ascendc::PlatformAscendC(platformInfoPtr);
    compileInfoPtr->coreNum = ascendcPlatform.GetCoreNumAiv();
    ascendcPlatform.GetCoreMemSize(platform_ascendc::CoreMemType::UB, compileInfoPtr->ubSize);
    return ge::GRAPH_SUCCESS;
}

IMPL_OP_OPTILING(ComplexAbs).Tiling(TilingForComplexAbs).TilingParse<ComplexAbsCompileInfo>(TilingPrepareForComplexAbs);
} // namespace optiling
