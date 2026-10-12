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
 * \file nan_to_num_tiling_arch35.cpp
 * \brief nan_to_num_tiling source file
 */

#include "nan_to_num_tiling_arch35.h"
#include <graph/utils/type_utils.h>
#include <iostream>
#include "tiling/platform/platform_ascendc.h"
#include "register/op_impl_registry.h"
#include "register/tilingdata_base.h"
#include "op_host/tiling_base_util.h"
#include "atvoss/elewise/elewise_tiling.h"
#include "log/log.h"
#include "util/math_util.h"
#include "platform/platform_info.h"
#include "math/nan_to_num/op_kernel/arch35/nan_to_num_dag.h"
#include "math/nan_to_num/op_kernel/arch35/nan_to_num_struct.h"

using namespace ge;
using namespace NanToNumOp;

namespace optiling {
constexpr int64_t ASCEND_API_BUFFER = 122880;
const int64_t ASCEND_WORKSPACE = 16 * 1024 * 1024;

ge::graphStatus NanToNumTiling::CalcInputDtype()
{
    OP_LOGD(tilingContext->GetNodeName(), "NanToNumTiling CalcInputDtype enter.");
    auto inputDesc = tilingContext->GetInputDesc(0);
    OP_CHECK_NULL_WITH_CONTEXT(tilingContext, inputDesc);
    this->inputDtype = inputDesc->GetDataType();
    OP_CHECK_IF(
        this->inputDtype != ge::DT_FLOAT16 && this->inputDtype != ge::DT_BF16 && this->inputDtype != ge::DT_FLOAT,
        OP_LOGE_FOR_INVALID_DTYPE(tilingContext->GetNodeName(), "x", ge::TypeUtils::DataTypeToSerialString(this->inputDtype), "FLOAT16, BF16, FLOAT"),
        return ge::GRAPH_FAILED);
    return ge::GRAPH_SUCCESS;
}

ge::graphStatus NanToNumTiling::CalcOutputDtype()
{
    OP_LOGD(tilingContext->GetNodeName(), "NanToNumTiling CalcOutputDtype enter.");
    auto outputDesc = tilingContext->GetOutputDesc(0);
    OP_CHECK_NULL_WITH_CONTEXT(tilingContext, outputDesc);
    this->outputDtype = outputDesc->GetDataType();
    OP_CHECK_IF(
        this->outputDtype != this->inputDtype,
        OP_LOGE_FOR_INVALID_DTYPES_WITH_REASON(tilingContext->GetNodeName(), "x, y", std::string(ge::TypeUtils::DataTypeToSerialString(this->outputDtype)) + ", " + std::string(ge::TypeUtils::DataTypeToSerialString(this->inputDtype)), "The dtypes of x and y must be the same"), return ge::GRAPH_FAILED);
    return ge::GRAPH_SUCCESS;
}

ge::graphStatus NanToNumTiling::CheckShape()
{
    OP_LOGD(tilingContext->GetNodeName(), "NanToNumTiling CheckShape enter.");
    auto inputStorageShape = tilingContext->GetInputShape(0);
    OP_CHECK_NULL_WITH_CONTEXT(tilingContext, inputStorageShape);
    const gert::Shape& inputXShape = Ops::Base::EnsureNotScalar(inputStorageShape->GetStorageShape());

    auto outputStorageShape = tilingContext->GetOutputShape(0);
    OP_CHECK_NULL_WITH_CONTEXT(tilingContext, outputStorageShape);
    const gert::Shape& outputYShape = Ops::Base::EnsureNotScalar(outputStorageShape->GetStorageShape());

    OP_CHECK_IF(
        inputXShape != outputYShape, OP_LOGE_FOR_INVALID_SHAPES_WITH_REASON(tilingContext->GetNodeName(), "x, y", (Ops::Base::ToString(inputXShape) + ", " + Ops::Base::ToString(outputYShape)).c_str(), "The shapes of x and y must be the same"),
        return ge::GRAPH_FAILED);
    return ge::GRAPH_SUCCESS;
}

ge::graphStatus NanToNumTiling::SetAttr()
{
    OP_LOGD(tilingContext->GetNodeName(), "NanToNumTiling SetAttr enter.");
    auto attrs = tilingContext->GetAttrs();
    OP_CHECK_NULL_WITH_CONTEXT(tilingContext, attrs);
    const float* nanValueAttr = attrs->GetAttrPointer<float>(NanToNumOp::PLACEHOLDER_INDEX_0);
    const float* posinfValueAttr = attrs->GetAttrPointer<float>(NanToNumOp::PLACEHOLDER_INDEX_1);
    const float* neginfValueAttr = attrs->GetAttrPointer<float>(NanToNumOp::PLACEHOLDER_INDEX_2);

    OP_CHECK_IF(
        nanValueAttr == nullptr, OP_LOGE_FOR_INVALID_VALUE(tilingContext->GetNodeName(), "nan", "nullptr", "not nullptr"),
        return ge::GRAPH_FAILED);
    OP_CHECK_IF(
        posinfValueAttr == nullptr, OP_LOGE_FOR_INVALID_VALUE(tilingContext->GetNodeName(), "posinf", "nullptr", "not nullptr"),
        return ge::GRAPH_FAILED);
    OP_CHECK_IF(
        neginfValueAttr == nullptr, OP_LOGE_FOR_INVALID_VALUE(tilingContext->GetNodeName(), "neginf", "nullptr", "not nullptr"),
        return ge::GRAPH_FAILED);

    tiling->nan = *nanValueAttr;
    tiling->posinf = *posinfValueAttr;
    tiling->neginf = *neginfValueAttr;
    return ge::GRAPH_SUCCESS;
}

ge::graphStatus NanToNumTiling::SetTilingData()
{
    OP_LOGD(tilingContext->GetNodeName(), "NanToNumTiling SetTilingData enter.");

    size_t* currentWorkspace = tilingContext->GetWorkspaceSizes(1);
    currentWorkspace[0] = ASCEND_WORKSPACE;

    const uint64_t tilingKey = GET_TPL_TILING_KEY(static_cast<uint64_t>(tiling->baseTiling.scheMode), dType);
    OP_LOGD(tilingContext->GetNodeName(), "[TilingData] : tilingKey=%lu", tilingKey);
    tilingContext->SetTilingKey(tilingKey);
    tilingContext->SetBlockDim(tiling->baseTiling.blockNum);
    return ge::GRAPH_SUCCESS;
}

ge::graphStatus NanToNumTiling::RunTiling()
{
    OP_LOGD(tilingContext->GetNodeName(), "NanToNumTiling RunTiling enter.");
    ElewiseBaseTiling elewiseBaseTiling(tilingContext);
    tiling = tilingContext->GetTilingData<NanToNumTilingData>();

    OP_CHECK_IF(
        CalcInputDtype() == ge::GRAPH_FAILED, OP_LOGE(tilingContext->GetNodeName(), "get input dtype failed"),
        return ge::GRAPH_FAILED);
    OP_CHECK_IF(
        CalcOutputDtype() == ge::GRAPH_FAILED, OP_LOGE(tilingContext->GetNodeName(), "get output dtype failed"),
        return ge::GRAPH_FAILED);
    OP_CHECK_IF(
        CheckShape() == ge::GRAPH_FAILED, OP_LOGE(tilingContext->GetNodeName(), "check shape failed"), return ge::GRAPH_FAILED);
    OP_CHECK_IF(SetAttr() == ge::GRAPH_FAILED, OP_LOGE(tilingContext->GetNodeName(), "set Attr failed"), return ge::GRAPH_FAILED);

    ge::graphStatus baseTilingResult = ge::GRAPH_FAILED;
    if (this->outputDtype == ge::DT_FLOAT16) {
        dType = TPL_FP16;
        baseTilingResult = elewiseBaseTiling.DoTiling<NanToNumOp::NanToNumDAG<half>::OpDag>(tiling->baseTiling);
    } else if (this->outputDtype == ge::DT_BF16) {
        dType = TPL_BF16;
        baseTilingResult = elewiseBaseTiling.DoTiling<NanToNumOp::NanToNumDAG<bfloat16_t>::OpDag>(tiling->baseTiling);
    } else if (this->outputDtype == ge::DT_FLOAT) {
        dType = TPL_FP32;
        baseTilingResult = elewiseBaseTiling.DoTiling<NanToNumOp::NanToNumDAG<float>::OpDag>(tiling->baseTiling);
    } else {
        OP_LOGE_FOR_INVALID_DTYPE(
            tilingContext->GetNodeName(), "y",
            ge::TypeUtils::DataTypeToSerialString(this->outputDtype), "FLOAT16, BF16, FLOAT");
        return ge::GRAPH_FAILED;
    }
    OP_CHECK_IF(
        baseTilingResult == ge::GRAPH_FAILED, OP_LOGE(tilingContext->GetNodeName(), "elewiseBaseTiling failed"),
        return ge::GRAPH_FAILED);

    return SetTilingData();
}

static ge::graphStatus Tiling4NanToNum(gert::TilingContext* context)
{
    OP_LOGD(context->GetNodeName(), "Tiling4NanToNum rt2.0 is running.");
    auto compileInfo = reinterpret_cast<const ElewiseCompileInfo*>(context->GetCompileInfo());
    OP_CHECK_NULL_WITH_CONTEXT(context, compileInfo);
    NanToNumTiling baseOpTiling(context);
    return baseOpTiling.RunTiling();
}

static ge::graphStatus TilingPrepareForNanToNum([[maybe_unused]] gert::TilingParseContext* context)
{
    return ge::GRAPH_SUCCESS;
}

IMPL_OP_OPTILING(NanToNum).Tiling(Tiling4NanToNum).TilingParse<ElewiseCompileInfo>(TilingPrepareForNanToNum);

} // namespace optiling
