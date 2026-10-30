/**
 * Copyright (c) 2025-2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

/*!
 * \file abs_tiling_arch35.cpp
 * \brief
 */
#include <iostream>

#include "abs_tiling_arch35.h"
#include <graph/utils/type_utils.h>
#include "tiling/tiling_api.h"
#include "tiling/platform/platform_ascendc.h"
#include "register/op_def_registry.h"
#include "register/tilingdata_base.h"
#include "../../op_kernel/arch35/abs_dag.h"
#include "../../op_kernel/arch35/abs_complex_dag.h"

using namespace ge;
using namespace AbsNs;

namespace optiling {
constexpr uint64_t ABS_TILING_KEY_ELEMENTWISE_BF16 = 101;
constexpr uint64_t ABS_TILING_KEY_ELEMENTWISE_OTHER = 102;
constexpr uint64_t ABS_TILING_KEY_ELEMENTWISE_COMPLEX = 103;
constexpr uint64_t ABS_WORKSPACE_RESERVE_BYTE = 16777216;

ge::graphStatus AbsTiling::SetTilingData()
{
    size_t* currentWorkspace = tilingContext->GetWorkspaceSizes(1);
    OP_CHECK_NULL_WITH_CONTEXT(tilingContext, currentWorkspace);
    currentWorkspace[0] = ABS_WORKSPACE_RESERVE_BYTE;
    if (this->outputDtype == ge::DT_BF16) {
        tilingContext->SetTilingKey(ABS_TILING_KEY_ELEMENTWISE_BF16);
    } else if (this->inputDtype == ge::DT_COMPLEX64 || this->inputDtype == ge::DT_COMPLEX32) { // 新增complex分支
        tilingContext->SetTilingKey(ABS_TILING_KEY_ELEMENTWISE_COMPLEX);
    } else {
        tilingContext->SetTilingKey(ABS_TILING_KEY_ELEMENTWISE_OTHER);
    }
    tilingContext->SetBlockDim(tiling->baseTiling.blockNum);
    return ge::GRAPH_SUCCESS;
}

ge::graphStatus AbsTiling::CalcOutputDtype()
{
    auto inputDesc = tilingContext->GetInputDesc(0);
    OP_CHECK_NULL_WITH_CONTEXT(tilingContext, inputDesc);
    this->inputDtype = inputDesc->GetDataType();

    auto outputDesc = tilingContext->GetOutputDesc(0);
    OP_CHECK_NULL_WITH_CONTEXT(tilingContext, outputDesc);
    this->outputDtype = outputDesc->GetDataType();

    if (this->inputDtype != ge::DT_COMPLEX64 && this->inputDtype != ge::DT_COMPLEX32) {
        OP_CHECK_IF(this->inputDtype != this->outputDtype,
                    OP_LOGE_FOR_INVALID_DTYPES_WITH_REASON(tilingContext->GetNodeName(), "x, y",
                                                               std::string(ge::TypeUtils::DataTypeToSerialString(this->inputDtype)) + ", " + std::string(ge::TypeUtils::DataTypeToSerialString(this->outputDtype)),
                                                               "The dtypes of x and y must be the same when the dtype of x is not COMPLEX64 or COMPLEX32"),
                    return ge::GRAPH_FAILED);
    } else if (inputDtype == ge::DT_COMPLEX64) {
        OP_CHECK_IF(this->outputDtype != ge::DT_FLOAT, 
                    OP_LOGE_FOR_INVALID_DTYPE_WITH_REASON(tilingContext->GetNodeName(), "outputDtype",
                                                         ge::TypeUtils::DataTypeToSerialString(this->outputDtype),
                                                         "The dtype of outputDtype must be FLOAT when the dtype of inputDtype is COMPLEX64"),
                    return ge::GRAPH_FAILED); 
    } else if (inputDtype == ge::DT_COMPLEX32) {
        OP_CHECK_IF(this->outputDtype != ge::DT_FLOAT16, 
                    OP_LOGE_FOR_INVALID_DTYPE_WITH_REASON(tilingContext->GetNodeName(), "outputDtype",
                                                         ge::TypeUtils::DataTypeToSerialString(this->outputDtype),
                                                         "The dtype of outputDtype must be FLOAT16 when the dtype of inputDtype is COMPLEX32"),
                    return ge::GRAPH_FAILED); 
    }
    return ge::GRAPH_SUCCESS;
}

ge::graphStatus AbsTiling::RunTiling()
{
    ElewiseBaseTiling elewiseBaseTiling(tilingContext);
    OP_CHECK_IF(CalcOutputDtype() == ge::GRAPH_FAILED,
        OP_LOGE(tilingContext, "get output dtype failed"),
        return ge::GRAPH_FAILED);

    ge::graphStatus res = ge::GRAPH_FAILED;
    tiling = tilingContext->GetTilingData<AbsTilingData>();
    if (this->inputDtype == ge::DT_FLOAT16) {
        res = elewiseBaseTiling.DoTiling<AbsOp::AbsDag<half, half>::OpDag>(tiling->baseTiling);
    } else if (this->inputDtype == ge::DT_COMPLEX64) {
        res = elewiseBaseTiling.DoTiling<AbsOp::AbscomplexDag<int64_t, float>::OpDag>(tiling->baseTiling);
    } else if (this->inputDtype == ge::DT_COMPLEX32) {
        res = elewiseBaseTiling.DoTiling<AbsOp::AbscomplexDag<int32_t, half>::OpDag>(tiling->baseTiling); 
    } else if (this->inputDtype == ge::DT_FLOAT) {
        res = elewiseBaseTiling.DoTiling<AbsOp::AbsDag<float, float>::OpDag>(tiling->baseTiling);
    } else if (this->inputDtype == ge::DT_BF16) {
        res = elewiseBaseTiling.DoTiling<AbsOp::AbsDag<bfloat16_t, float>::OpDag>(tiling->baseTiling);
    } else if (this->inputDtype == ge::DT_INT8) {
        res = elewiseBaseTiling.DoTiling<AbsOp::AbsDag<int8_t, int8_t>::OpDag>(tiling->baseTiling);
    } else if (this->inputDtype == ge::DT_INT16) {
        res = elewiseBaseTiling.DoTiling<AbsOp::AbsDag<int16_t, int16_t>::OpDag>(tiling->baseTiling);
    } else if (this->inputDtype == ge::DT_INT32) {
        res = elewiseBaseTiling.DoTiling<AbsOp::AbsDag<int32_t, int32_t>::OpDag>(tiling->baseTiling);
    } else if (this->inputDtype == ge::DT_INT64) {
        res = elewiseBaseTiling.DoTiling<AbsOp::AbsDag<int64_t, int64_t>::OpDag>(tiling->baseTiling);
    } else {
        OP_LOGE_FOR_INVALID_DTYPE(tilingContext->GetNodeName(), "inputDtype",
                                       ge::TypeUtils::DataTypeToSerialString(this->inputDtype),
                                       "FLOAT16, FLOAT, BF16, INT8, INT16, INT32, INT64, COMPLEX64, COMPLEX32");
        return ge::GRAPH_FAILED;
    }

    OP_CHECK_IF(res == ge::GRAPH_FAILED,
        OP_LOGE(tilingContext, "DoTiling failed"),
        return ge::GRAPH_FAILED);

    ge::graphStatus result = SetTilingData();
    return result;
}

static ge::graphStatus TilingForAbs(gert::TilingContext *context)
{
    OP_LOGD("AbsTiling", "Enter TilingForAbs");
    OP_CHECK_IF(context == nullptr,
        OP_LOGE(context, "Tiling context is null"),
        return ge::GRAPH_FAILED);

    // 走新的模板tiling
    OP_LOGD("AbsTiling", "Enter new AbsTiling");
    AbsTiling absTiling(context);
    return absTiling.RunTiling();
}

ge::graphStatus TilingPrepareForAbs(gert::TilingParseContext* context)
{
    auto compileInfoPtr = context->GetCompiledInfo<AbsCompileInfo>();
    OP_CHECK_NULL_WITH_CONTEXT(context, compileInfoPtr);
    fe::PlatFormInfos* platformInfoPtr = context->GetPlatformInfo();
    OP_CHECK_NULL_WITH_CONTEXT(context, platformInfoPtr);
    auto ascendcPlatform = platform_ascendc::PlatformAscendC(platformInfoPtr);
    compileInfoPtr->coreNum = ascendcPlatform.GetCoreNumAiv();
    ascendcPlatform.GetCoreMemSize(platform_ascendc::CoreMemType::UB, compileInfoPtr->ubSize);
    return ge::GRAPH_SUCCESS;
}

IMPL_OP_OPTILING(Abs).Tiling(TilingForAbs)
    .TilingParse<AbsCompileInfo>(TilingPrepareForAbs);
}  // namespace optiling