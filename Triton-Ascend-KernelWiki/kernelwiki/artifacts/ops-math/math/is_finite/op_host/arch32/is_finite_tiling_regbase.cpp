/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

/* !
 * \file is_finite_tiling_regbase.cpp
 * \brief
 */

#include "is_finite_tiling_regbase.h"
#include "op_host/tiling_base_util.h"
#include "log/log.h"
#include "graph/utils/type_utils.h"
#include "../../op_kernel/arch35/is_finite_dag.h"
#include "../../op_kernel/arch35/is_finite_struct_arch35.h"

using namespace IsFiniteOp;

namespace optiling {
ge::graphStatus IsFiniteRegbaseTiling::CalcInputDtype()
{
    auto inputDesc = tilingContext->GetInputDesc(0);
    OP_CHECK_NULL_WITH_CONTEXT(tilingContext, inputDesc);
    this->inputDtype = inputDesc->GetDataType();
    OP_CHECK_IF(
        this->inputDtype != ge::DT_BF16 && this->inputDtype != ge::DT_FLOAT16 && this->inputDtype != ge::DT_FLOAT,
        OP_LOGE(
            tilingContext, "Input x dtype not supported, only support [DT_FLOAT, DT_FLOAT16, DT_BF16], got %s",
            ge::TypeUtils::DataTypeToSerialString(this->inputDtype).c_str()),
        return ge::GRAPH_FAILED);
    return ge::GRAPH_SUCCESS;
}

ge::graphStatus IsFiniteRegbaseTiling::CalcOutputDtype()
{
    auto outputDesc = tilingContext->GetOutputDesc(0);
    OP_CHECK_NULL_WITH_CONTEXT(tilingContext, outputDesc);
    this->outputDtype = outputDesc->GetDataType();
    OP_CHECK_IF(
        this->outputDtype != ge::DT_BOOL,
        OP_LOGE(
            tilingContext, "Output y dtype not supported, only support [DT_BOOL], got %s",
            ge::TypeUtils::DataTypeToSerialString(this->outputDtype).c_str()),
        return ge::GRAPH_FAILED);
    return ge::GRAPH_SUCCESS;
}

ge::graphStatus IsFiniteRegbaseTiling::CheckShape()
{
    auto inputStorageShape = tilingContext->GetInputShape(0);
    OP_CHECK_NULL_WITH_CONTEXT(tilingContext, inputStorageShape);
    const gert::Shape& inputXShape = Ops::Base::EnsureNotScalar(inputStorageShape->GetStorageShape());

    auto outputStorageShape = tilingContext->GetOutputShape(0);
    OP_CHECK_NULL_WITH_CONTEXT(tilingContext, outputStorageShape);
    const gert::Shape& outputYShape = Ops::Base::EnsureNotScalar(outputStorageShape->GetStorageShape());

    OP_CHECK_IF(
        inputXShape != outputYShape, OP_LOGE(tilingContext, "Input x and output y shape not the same"),
        return ge::GRAPH_FAILED);
    return ge::GRAPH_SUCCESS;
}

ge::graphStatus IsFiniteRegbaseTiling::SetTilingData()
{
    auto rawTilingData = tilingContext->GetRawTilingData();
    OP_CHECK_NULL_WITH_CONTEXT(tilingContext, rawTilingData);

    size_t* currentWorkspace = tilingContext->GetWorkspaceSizes(1);
    currentWorkspace[0] = static_cast<uint64_t>(ASCEND_WORKSPACE);

    const uint64_t tilingKey = GET_TPL_TILING_KEY(TPL_EXTRA, tiling_->scheMode, dType);
    OP_LOGD(tilingContext, "[TilingData] : tilingKey=%lu", tilingKey);
    tilingContext->SetTilingKey(tilingKey);
    tilingContext->SetBlockDim(tiling_->blockNum);
    return ge::GRAPH_SUCCESS;
}

ge::graphStatus IsFiniteRegbaseTiling::RunTiling()
{
    ElewiseBaseTiling elewiseBaseTiling(tilingContext);
    OP_CHECK_IF(
        CalcInputDtype() == ge::GRAPH_FAILED, OP_LOGE(tilingContext, "Get input dtype failed"),
        return ge::GRAPH_FAILED);
    OP_CHECK_IF(
        CalcOutputDtype() == ge::GRAPH_FAILED, OP_LOGE(tilingContext, "Get output dtype failed"),
        return ge::GRAPH_FAILED);
    OP_CHECK_IF(
        CheckShape() == ge::GRAPH_FAILED, OP_LOGE(tilingContext, "Check shape failed"), return ge::GRAPH_FAILED);

    tiling_ = tilingContext->GetTilingData<EleBaseTilingDataV2>();
    OP_CHECK_IF(
        (tiling_ == nullptr), OP_LOGE(tilingContext, "Get EleBaseTilingDataV2 from context failed"),
        return ge::GRAPH_FAILED);

    ge::graphStatus baseTilingResult = ge::GRAPH_FAILED;
    if (this->inputDtype == ge::DT_FLOAT16) {
        dType = TPL_FP16;
        baseTilingResult = elewiseBaseTiling.DoTiling<IsFiniteDag<half>::OpDag>(*tiling_, ASCEND_API_BUFFER);
    } else if (this->inputDtype == ge::DT_BF16) {
        dType = TPL_BF16;
        baseTilingResult = elewiseBaseTiling.DoTiling<IsFiniteDag<bfloat16_t>::OpDag>(*tiling_, ASCEND_API_BUFFER);
    } else if (this->inputDtype == ge::DT_FLOAT) {
        dType = TPL_FP32;
        baseTilingResult = elewiseBaseTiling.DoTiling<IsFiniteDag<float>::OpDag>(*tiling_, ASCEND_API_BUFFER);
    } else {
        OP_LOGE(
            tilingContext, "Input x dtype not supported, only support [DT_FLOAT, DT_FLOAT16, DT_BF16], got %s",
            ge::TypeUtils::DataTypeToSerialString(this->inputDtype).c_str());
        return ge::GRAPH_FAILED;
    }
    OP_CHECK_IF(
        baseTilingResult == ge::GRAPH_FAILED, OP_LOGE(tilingContext, "ElewiseBaseTiling failed"),
        return ge::GRAPH_FAILED);

    return SetTilingData();
}
} // namespace optiling