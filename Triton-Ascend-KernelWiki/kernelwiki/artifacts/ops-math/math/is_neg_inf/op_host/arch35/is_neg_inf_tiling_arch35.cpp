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
 * \file is_neg_inf_tiling_arch35.cpp
 * \brief
 */
#include <graph/utils/type_utils.h>
#include "register/op_impl_registry.h"
#include "log/log.h"
#include "tiling/platform/platform_ascendc.h"
#include "atvoss/elewise/elewise_tiling.h"
#include "op_host/tiling_base_util.h"
#include "math/is_neg_inf/op_kernel/arch35/is_neg_inf_dag.h"
#include "math/is_neg_inf/op_kernel/arch35/is_neg_inf_struct.h"
#include "is_neg_inf_tiling_arch35.h"

using namespace IsNegInfOp;

namespace optiling {
const int64_t ASCEND_WORKSPACE = 16 * 1024 * 1024;
constexpr int64_t MAX_DIM_NUM = 8;

ge::graphStatus IsNegInfRegbaseTiling::CalcInputDtype()
{
    auto inputDesc = tilingContext->GetInputDesc(0);
    OP_CHECK_NULL_WITH_CONTEXT(tilingContext, inputDesc);
    this->inputDtype = inputDesc->GetDataType();
    OP_CHECK_IF(
        this->inputDtype != ge::DT_FLOAT16 && this->inputDtype != ge::DT_BF16 && this->inputDtype != ge::DT_FLOAT,
        OP_LOGE_FOR_INVALID_DTYPE(tilingContext->GetNodeName(), "x",
            ge::TypeUtils::DataTypeToSerialString(this->inputDtype),
            "FLOAT16, BF16, FLOAT"),
        return ge::GRAPH_FAILED);
    return ge::GRAPH_SUCCESS;
}

ge::graphStatus IsNegInfRegbaseTiling::CalcOutputDtype()
{
    auto outputDesc = tilingContext->GetOutputDesc(0);
    OP_CHECK_NULL_WITH_CONTEXT(tilingContext, outputDesc);
    this->outputDtype = outputDesc->GetDataType();
    OP_CHECK_IF(
        this->outputDtype != ge::DT_BOOL,
        OP_LOGE_FOR_INVALID_DTYPE(tilingContext->GetNodeName(), "y",
            ge::TypeUtils::DataTypeToSerialString(this->outputDtype),
            "BOOL"),
        return ge::GRAPH_FAILED);
    return ge::GRAPH_SUCCESS;
}

ge::graphStatus IsNegInfRegbaseTiling::CheckShape()
{
    auto inputStorageShape = tilingContext->GetInputShape(0);
    OP_CHECK_NULL_WITH_CONTEXT(tilingContext, inputStorageShape);
    const gert::Shape& inputXShape = Ops::Base::EnsureNotScalar(inputStorageShape->GetStorageShape());

    auto outputStorageShape = tilingContext->GetOutputShape(0);
    OP_CHECK_NULL_WITH_CONTEXT(tilingContext, outputStorageShape);
    const gert::Shape& outputYShape = Ops::Base::EnsureNotScalar(outputStorageShape->GetStorageShape());

    OP_CHECK_IF(
        inputXShape.GetDimNum() > MAX_DIM_NUM,
        OP_LOGE_FOR_INVALID_SHAPEDIM_WITH_REASON(tilingContext->GetNodeName(), "x", std::to_string(inputXShape.GetDimNum()), "dim num should be no more than 8"), return ge::GRAPH_FAILED);

    OP_CHECK_IF(
        inputXShape != outputYShape,
        OP_LOGE_FOR_INVALID_SHAPES_WITH_REASON(tilingContext->GetNodeName(), "x, y", (Ops::Base::ToString(inputXShape) + ", " + Ops::Base::ToString(outputYShape)).c_str(), "The shapes of x and y must be the same"), return ge::GRAPH_FAILED);
    return ge::GRAPH_SUCCESS;
}

ge::graphStatus IsNegInfRegbaseTiling::RunTiling()
{
    ElewiseBaseTiling elewiseBaseTiling(tilingContext);
    OP_CHECK_IF(
        CalcInputDtype() == ge::GRAPH_FAILED, OP_LOGE(tilingContext->GetNodeName(), "get input dtype failed"),
        return ge::GRAPH_FAILED);
    OP_CHECK_IF(
        CalcOutputDtype() == ge::GRAPH_FAILED, OP_LOGE(tilingContext->GetNodeName(), "get output dtype failed"),
        return ge::GRAPH_FAILED);
    OP_CHECK_IF(
        CheckShape() == ge::GRAPH_FAILED, OP_LOGE(tilingContext->GetNodeName(), "check shape failed"), return ge::GRAPH_FAILED);

    auto tiling = tilingContext->GetTilingData<EleBaseTilingDataV2>();
    OP_CHECK_IF(
        (tiling == nullptr), OP_LOGE_FOR_INVALID_VALUE(tilingContext->GetNodeName(), "tiling_data", "nullptr", "not nullptr"),
        return ge::GRAPH_FAILED);

    ge::graphStatus baseTilingResult = ge::GRAPH_FAILED;
    if (this->inputDtype == ge::DT_FLOAT16) {
        dType = TPL_FP16;
        baseTilingResult = elewiseBaseTiling.DoTiling<IsNegInfOp::IsNegInfDAG<half>::OpDag>(*tiling, ASCEND_API_BUFFER);
    } else if (this->inputDtype == ge::DT_BF16) {
        dType = TPL_BF16;
        baseTilingResult =
            elewiseBaseTiling.DoTiling<IsNegInfOp::IsNegInfDAG<bfloat16_t>::OpDag>(*tiling, ASCEND_API_BUFFER);
    } else if (this->inputDtype == ge::DT_FLOAT) {
        dType = TPL_FP32;
        baseTilingResult =
            elewiseBaseTiling.DoTiling<IsNegInfOp::IsNegInfDAG<float>::OpDag>(*tiling, ASCEND_API_BUFFER);
    } else {
        OP_LOGE_FOR_INVALID_DTYPE(
            tilingContext->GetNodeName(), "x",
            ge::TypeUtils::DataTypeToSerialString(this->inputDtype),
            "FLOAT16, BF16, FLOAT");
        return ge::GRAPH_FAILED;
    }
    OP_CHECK_IF(
        baseTilingResult == ge::GRAPH_FAILED, OP_LOGE(tilingContext->GetNodeName(), "elewiseBaseTiling failed"),
        return ge::GRAPH_FAILED);

    // set workspace/tilingkey/blockdim
    size_t* currentWorkspace = tilingContext->GetWorkspaceSizes(1);
    OP_CHECK_NULL_WITH_CONTEXT(tilingContext, currentWorkspace);
    currentWorkspace[0] = ASCEND_WORKSPACE;

    const uint64_t tilingKey = GET_TPL_TILING_KEY(tiling->scheMode, dType);
    tilingContext->SetTilingKey(tilingKey);
    tilingContext->SetBlockDim(tiling->blockNum);

    return ge::GRAPH_SUCCESS;
}

static ge::graphStatus Tiling4IsNegInf(gert::TilingContext* tilingContextGen)
{
    OP_LOGD(tilingContextGen->GetNodeName(), "Tiling4IsNegInf rt2.0 is running.");
    auto compileInfo = tilingContextGen->GetCompileInfo<IsNegInfCompileInfo>();
    OP_CHECK_NULL_WITH_CONTEXT(tilingContextGen, compileInfo);
    IsNegInfRegbaseTiling baseOpTiling(tilingContextGen);
    return baseOpTiling.RunTiling();
}

static ge::graphStatus TilingPrepareForIsNegInf(gert::TilingParseContext* context)
{
    auto compileInfoPtr = context->GetCompiledInfo<IsNegInfCompileInfo>();
    OP_CHECK_NULL_WITH_CONTEXT(context, compileInfoPtr);

    fe::PlatFormInfos* platformInfoPtr = context->GetPlatformInfo();
    OP_CHECK_NULL_WITH_CONTEXT(context, platformInfoPtr);
    auto ascendcPlatform = platform_ascendc::PlatformAscendC(platformInfoPtr);
    compileInfoPtr->coreNum = ascendcPlatform.GetCoreNumAiv();
    ascendcPlatform.GetCoreMemSize(platform_ascendc::CoreMemType::UB, compileInfoPtr->ubSize);
    return ge::GRAPH_SUCCESS;
}

IMPL_OP_OPTILING(IsNegInf).Tiling(Tiling4IsNegInf).TilingParse<IsNegInfCompileInfo>(TilingPrepareForIsNegInf);

} // namespace optiling