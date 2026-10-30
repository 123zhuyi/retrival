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
 * \file muls_tiling.cc
 * \brief
 */
#include "muls_tiling_arch35.h"
#include <graph/utils/type_utils.h>
#include "platform/platform_ascendc.h"
#include "log/log.h"
#include "atvoss/elewise/elewise_tiling.h"
#include "math/muls/op_kernel/arch35/muls_dag.h"
#include "op_host/tiling_base_util.h"

namespace optiling
{
using namespace ge;
using namespace MulsDag;

constexpr uint64_t WORKSPACE_RESERVE_BYTE = 16777216;  // 16 * 1024 * 1024

ge::graphStatus MulsTiling::SetTilingData()
{
    size_t* currentWorkspace = tilingContext->GetWorkspaceSizes(1);
    currentWorkspace[0] = WORKSPACE_RESERVE_BYTE;
    tilingContext->SetBlockDim(tiling->baseTiling.blockNum);
    return ge::GRAPH_SUCCESS;
}

ge::graphStatus MulsTiling::CalcOutputDtype()
{
    auto inputDesc = tilingContext->GetInputDesc(0);
    OP_CHECK_NULL_WITH_CONTEXT(tilingContext, inputDesc);
    ge::DataType inputDtype = inputDesc->GetDataType();

    auto outputDesc = tilingContext->GetOutputDesc(0);
    OP_CHECK_NULL_WITH_CONTEXT(tilingContext, outputDesc);
    this->outputDtype = outputDesc->GetDataType();

    OP_CHECK_IF(inputDtype != this->outputDtype,
        OP_LOGE_FOR_INVALID_DTYPES_WITH_REASON(tilingContext->GetNodeName(), "x and y",
            (Ops::Base::ToString(inputDtype) + " and " + Ops::Base::ToString(this->outputDtype)).c_str(),
            "The dtypes of x and y must be the same"),
        return ge::GRAPH_FAILED);
    return ge::GRAPH_SUCCESS;
}

ge::graphStatus MulsTiling::CheckShape()
{
    OP_LOGD(tilingContext->GetNodeName(), "MulsTiling CheckShape enter.");
    auto inputStorageShape = tilingContext->GetInputShape(0);
    OP_CHECK_NULL_WITH_CONTEXT(tilingContext, inputStorageShape);
    const gert::Shape& inputShape = Ops::Base::EnsureNotScalar(inputStorageShape->GetStorageShape());

    auto outputStorageShape = tilingContext->GetOutputShape(0);
    OP_CHECK_NULL_WITH_CONTEXT(tilingContext, outputStorageShape);
    const gert::Shape& outputShape = Ops::Base::EnsureNotScalar(outputStorageShape->GetStorageShape());

    if (inputShape != outputShape) {
        std::string inputShapeStr = Ops::Base::ToString(inputShape);
        std::string outputShapeStr = Ops::Base::ToString(outputShape);
        std::string shapesStr = inputShapeStr + " and " + outputShapeStr;
        OP_LOGE_FOR_INVALID_SHAPES_WITH_REASON(tilingContext->GetNodeName(), "x and y",
            shapesStr.c_str(), "The shapes of x and y must be the same");
        return ge::GRAPH_FAILED;
    }

    return ge::GRAPH_SUCCESS;
}

ge::graphStatus MulsTiling::RunTiling()
{
    ElewiseBaseTiling elewiseBaseTiling(tilingContext);
    OP_CHECK_IF(CalcOutputDtype() == ge::GRAPH_FAILED,
               OP_LOGE(tilingContext, "get output dtype failed"), return ge::GRAPH_FAILED);
    OP_CHECK_IF(CheckShape() == ge::GRAPH_FAILED, OP_LOGE(tilingContext, "check shape failed"),
               return ge::GRAPH_FAILED);

    // get tilingdata address in context
    tiling = tilingContext->GetTilingData<MulsTilingData>();
    OP_CHECK_NULL_WITH_CONTEXT(tilingContext, tiling);

    ge::graphStatus res = ge::GRAPH_FAILED;
    if (this->outputDtype == ge::DT_FLOAT16) {
        res = elewiseBaseTiling.DoTiling<MulsOp<half, CAST_MODE_NONE, CAST_MODE_RINT>::OpDag>(tiling->baseTiling);
    } else if (this->outputDtype == ge::DT_FLOAT) {
        res = elewiseBaseTiling.DoTiling<MulsOp<float, CAST_MODE_NONE, CAST_MODE_RINT>::OpDag>(tiling->baseTiling);
    } else if (this->outputDtype == ge::DT_BF16) {
        res = elewiseBaseTiling.DoTiling<MulsOp<bfloat16_t, CAST_MODE_NONE, CAST_MODE_RINT>::OpDag>(tiling->baseTiling);
    } else if (this->outputDtype == ge::DT_INT32) {
        res = elewiseBaseTiling.DoTiling<MulsOp<int32_t, CAST_MODE_ROUND, CAST_MODE_TRUNC>::OpDag>(tiling->baseTiling);
    } else if (this->outputDtype == ge::DT_INT16) {
        res = elewiseBaseTiling.DoTiling<MulsInt16Op::OpDag>(tiling->baseTiling);
    } else if (this->outputDtype == ge::DT_INT64) {
        res = elewiseBaseTiling.DoTiling<MulsOp<int64_t, CAST_MODE_ROUND, CAST_MODE_ROUND>::OpDag>(tiling->baseTiling);
    } else if (this->outputDtype == ge::DT_COMPLEX32) {
        res = elewiseBaseTiling.DoTiling<MulsComplex32Op<int32_t, int64_t>::OpDag>(tiling->baseTiling);
    } else if (this->outputDtype == ge::DT_COMPLEX64) {
        res = elewiseBaseTiling.DoTiling<MulsComplex64Op<int64_t>::OpDag>(tiling->baseTiling);
    } else {
        OP_LOGE_FOR_INVALID_DTYPE(tilingContext->GetNodeName(), "y",
            Ops::Base::ToString(this->outputDtype).c_str(),
            "float16, float32, bfloat16, int32, int16, int64, complex32 or complex64");
        return ge::GRAPH_FAILED;
    }

    OP_CHECK_IF(res != ge::GRAPH_SUCCESS, OP_LOGE(tilingContext, "DoTiling failed"),
               return ge::GRAPH_FAILED);

    auto runtimeAttrs = tilingContext->GetAttrs();
    OP_CHECK_NULL_WITH_CONTEXT(tilingContext, runtimeAttrs);
    tiling->value = 1;
    const float* valuePtr = runtimeAttrs->GetAttrPointer<float>(0);
    if (valuePtr != nullptr) {
        tiling->value = *valuePtr;
    }

    ge::graphStatus result = SetTilingData();
    return result;
}

static ge::graphStatus TilingForMuls(gert::TilingContext* context)
{
    OP_LOGD("MulsTiling", "Enter TilingForMuls");
    OP_CHECK_IF(context == nullptr, OP_LOGE(context, "Tiling context is null"),
               return ge::GRAPH_FAILED);

    OP_LOGD("MulsTiling", "Enter new MulsTiling");
    MulsTiling mulsTiling(context);
    return mulsTiling.RunTiling();
}

ge::graphStatus TilingPrepareForMuls([[maybe_unused]] gert::TilingParseContext *context) {
    return ge::GRAPH_SUCCESS;
}

IMPL_OP_OPTILING(Muls).Tiling(TilingForMuls).TilingParse<ElewiseCompileInfo>(TilingPrepareForMuls);
}  // namespace optiling