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
 * \file ceil_tiling_arch35.cpp
 * \brief
 */
#include "ceil_tiling_arch35.h"
#include <graph/utils/type_utils.h>
#include "tiling/platform/platform_ascendc.h"
#include "register/op_impl_registry.h"
#include "register/tilingdata_base.h"
#include "op_host/tiling_base_util.h"
#include "math/ceil/op_kernel/arch35/ceil_dag.h"
#include "math/ceil/op_kernel/arch35/ceil_struct.h"
#include "log/log.h"

#include <iostream>

namespace optiling {
const size_t ASCEND_WORKSPACE = 0;

ge::graphStatus CeilTiling::SetTilingData(const ElewiseBaseTiling& elewiseBaseTiling)
{
    size_t* currentWorkspace = tilingContext->GetWorkspaceSizes(1);
    OP_CHECK_NULL_WITH_CONTEXT(tilingContext, currentWorkspace);
    currentWorkspace[0] = ASCEND_WORKSPACE;

    const uint64_t tilingKey = GET_TPL_TILING_KEY(1, dType);
    OP_LOGD(tilingContext->GetNodeName(), "[TilingData] : tilingKey=%lu", tilingKey);
    tilingContext->SetTilingKey(tilingKey);
    tilingContext->SetBlockDim(elewiseBaseTiling.GetBlockDim());
    return ge::GRAPH_SUCCESS;
}

ge::graphStatus CeilTiling::CalcInputDtype()
{
    auto inputDesc = tilingContext->GetInputDesc(0);
    OP_CHECK_NULL_WITH_CONTEXT(tilingContext, inputDesc);
    this->inputDtype = inputDesc->GetDataType();
    OP_CHECK_IF(
        this->inputDtype != ge::DT_FLOAT16 && this->inputDtype != ge::DT_BF16 && this->inputDtype != ge::DT_FLOAT,
        OP_LOGE_FOR_INVALID_DTYPE(tilingContext->GetNodeName(), "inputDtype",
                                  ge::TypeUtils::DataTypeToSerialString(this->inputDtype), "FLOAT16, BF16, FLOAT"),
        return ge::GRAPH_FAILED);
    return ge::GRAPH_SUCCESS;
}

ge::graphStatus CeilTiling::CheckShape() const
{
    auto inputStorageShape = tilingContext->GetInputShape(0);
    OP_CHECK_NULL_WITH_CONTEXT(tilingContext, inputStorageShape);
    const gert::Shape& inputYShape = Ops::Base::EnsureNotScalar(inputStorageShape->GetStorageShape());

    auto outputStorageShape = tilingContext->GetOutputShape(0);
    OP_CHECK_NULL_WITH_CONTEXT(tilingContext, outputStorageShape);
    const gert::Shape& outputZShape = Ops::Base::EnsureNotScalar(outputStorageShape->GetStorageShape());

    OP_CHECK_IF(inputYShape != outputZShape,
                OP_LOGE_FOR_INVALID_SHAPES_WITH_REASON(
                    tilingContext->GetNodeName(), "x, y",
                    (Ops::Base::ToString(inputYShape) + ", " + Ops::Base::ToString(outputZShape)).c_str(),
                    "The shapes of x and y must be the same"),
                return ge::GRAPH_FAILED);
    return ge::GRAPH_SUCCESS;
}

ge::graphStatus CeilTiling::CalcOutputDtype()
{
    auto outputDesc = tilingContext->GetOutputDesc(0);
    OP_CHECK_NULL_WITH_CONTEXT(tilingContext, outputDesc);
    this->outputDtype = outputDesc->GetDataType();
    OP_CHECK_IF(this->outputDtype != this->inputDtype,
                OP_LOGE_FOR_INVALID_DTYPES_WITH_REASON(
                    tilingContext->GetNodeName(), "x, y",
                    std::string(ge::TypeUtils::DataTypeToSerialString(this->outputDtype)) + ", " +
                        std::string(ge::TypeUtils::DataTypeToSerialString(this->inputDtype)),
                    "The dtypes of x and y must be the same"),
                return ge::GRAPH_FAILED);
    return ge::GRAPH_SUCCESS;
}

ge::graphStatus CeilTiling::RunTiling()
{
    ElewiseBaseTiling elewiseBaseTiling(tilingContext);
    OP_CHECK_IF(CalcInputDtype() == ge::GRAPH_FAILED, OP_LOGE(tilingContext->GetNodeName(), "get input dtype failed"),
                return ge::GRAPH_FAILED);
    OP_CHECK_IF(CalcOutputDtype() == ge::GRAPH_FAILED, OP_LOGE(tilingContext->GetNodeName(), "get output dtype failed"),
                return ge::GRAPH_FAILED);
    OP_CHECK_IF(CheckShape() == ge::GRAPH_FAILED, OP_LOGE(tilingContext->GetNodeName(), "check shape failed"),
                return ge::GRAPH_FAILED);

    ge::graphStatus baseTilingResult = ge::GRAPH_FAILED;
    auto tiling = tilingContext->GetTilingData<EleBaseTilingData16B>();
    if (this->outputDtype == ge::DT_FLOAT16) {
        dType = TPL_FP16;
        baseTilingResult = elewiseBaseTiling.DoTiling<CeilOp::CeilDAG<half>::OpDag>(*tiling);
    } else if (this->outputDtype == ge::DT_BF16) {
        dType = TPL_BF16;
        baseTilingResult = elewiseBaseTiling.DoTiling<CeilOp::CeilDAG<bfloat16_t>::OpDag>(*tiling);
    } else if (this->outputDtype == ge::DT_FLOAT) {
        dType = TPL_FP32;
        baseTilingResult = elewiseBaseTiling.DoTiling<CeilOp::CeilDAG<float>::OpDag>(*tiling);
    } else {
        OP_LOGE_FOR_INVALID_DTYPE(tilingContext->GetNodeName(), "y",
                                  ge::TypeUtils::DataTypeToSerialString(this->outputDtype), "FLOAT16, BF16, FLOAT");
        return ge::GRAPH_FAILED;
    }
    OP_CHECK_IF(baseTilingResult == ge::GRAPH_FAILED,
                OP_LOGE(tilingContext->GetNodeName(), "elewiseBaseTiling failed, output dtype: %s.",
                        ge::TypeUtils::DataTypeToSerialString(this->outputDtype).c_str()),
                return ge::GRAPH_FAILED);

    return SetTilingData(elewiseBaseTiling);
}

static ge::graphStatus Tiling4Ceil(gert::TilingContext* tilingContextGen)
{
    OP_LOGD(tilingContextGen->GetNodeName(), "Tiling4Ceil rt2.0 is running.");
    auto compileInfo = tilingContextGen->GetCompileInfo<CeilCompileInfo>();
    OP_CHECK_NULL_WITH_CONTEXT(tilingContextGen, compileInfo);
    CeilTiling baseOpTiling(tilingContextGen);
    return baseOpTiling.RunTiling();
}

static ge::graphStatus TilingPrepareForCeil([[maybe_unused]] gert::TilingParseContext* context)
{
    return ge::GRAPH_SUCCESS;
}

IMPL_OP_OPTILING(Ceil).Tiling(Tiling4Ceil).TilingParse<CeilCompileInfo>(TilingPrepareForCeil);
} // namespace optiling
