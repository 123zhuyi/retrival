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
 * \file coordinates_1d_to_2d_infershape.cpp
 * \brief infershape func of Coordinates1DTo2D
 */
#include "log/log.h"
#include "register/op_impl_registry.h"

using namespace ge;
namespace ops {

static constexpr size_t COORDINATES_1D_TO_2D_IDX_IN_X = 0;
static constexpr size_t COORDINATES_1D_TO_2D_IDX_OUT_ROW = 0;
static constexpr size_t COORDINATES_1D_TO_2D_IDX_OUT_COL = 1;
static constexpr size_t COORDINATES_1D_TO_2D_IDX_OUT_N = 2;

static ge::graphStatus Coordinates1DTo2DInferShape(gert::InferShapeContext* context)
{
    OP_LOGD(context->GetNodeName(), "Begin to do Coordinates1DTo2DInferShape");
    const auto* xShape = context->GetInputShape(COORDINATES_1D_TO_2D_IDX_IN_X);
    OP_CHECK_NULL_WITH_CONTEXT(context, xShape);
    auto* rowShape = context->GetOutputShape(COORDINATES_1D_TO_2D_IDX_OUT_ROW);
    OP_CHECK_NULL_WITH_CONTEXT(context, rowShape);
    auto* colShape = context->GetOutputShape(COORDINATES_1D_TO_2D_IDX_OUT_COL);
    OP_CHECK_NULL_WITH_CONTEXT(context, colShape);
    auto* nShape = context->GetOutputShape(COORDINATES_1D_TO_2D_IDX_OUT_N);
    OP_CHECK_NULL_WITH_CONTEXT(context, nShape);

    *rowShape = *xShape;
    *colShape = *xShape;
    *nShape = *xShape;
    OP_LOGD(context->GetNodeName(), "End to do Coordinates1DTo2DInferShape");
    return ge::GRAPH_SUCCESS;
}

static ge::graphStatus Coordinates1DTo2DInferDataType(gert::InferDataTypeContext* context)
{
    OP_LOGD(context->GetNodeName(), "Begin to do Coordinates1DTo2DInferDataType");
    auto xDtype = context->GetInputDataType(COORDINATES_1D_TO_2D_IDX_IN_X);
    context->SetOutputDataType(COORDINATES_1D_TO_2D_IDX_OUT_ROW, xDtype);
    context->SetOutputDataType(COORDINATES_1D_TO_2D_IDX_OUT_COL, xDtype);
    context->SetOutputDataType(COORDINATES_1D_TO_2D_IDX_OUT_N, xDtype);
    OP_LOGD(context->GetNodeName(), "End to do Coordinates1DTo2DInferDataType");
    return ge::GRAPH_SUCCESS;
}

static ge::graphStatus Coordinates1DTo2DInferShapeRange(gert::InferShapeRangeContext* context)
{
    OP_LOGD(context->GetNodeName(), "Begin to do Coordinates1DTo2DInferShapeRange");
    const auto* xRange = context->GetInputShapeRange(COORDINATES_1D_TO_2D_IDX_IN_X);
    OP_CHECK_NULL_WITH_CONTEXT(context, xRange);
    const auto* xRangeMax = xRange->GetMax();
    const auto* xRangeMin = xRange->GetMin();
    OP_CHECK_NULL_WITH_CONTEXT(context, xRangeMax);
    OP_CHECK_NULL_WITH_CONTEXT(context, xRangeMin);

    for (size_t i = 0; i < 3; i++) {
        auto* outRange = context->GetOutputShapeRange(i);
        OP_CHECK_NULL_WITH_CONTEXT(context, outRange);
        auto* outRangeMax = outRange->GetMax();
        auto* outRangeMin = outRange->GetMin();
        OP_CHECK_NULL_WITH_CONTEXT(context, outRangeMax);
        OP_CHECK_NULL_WITH_CONTEXT(context, outRangeMin);
        *outRangeMax = *xRangeMax;
        *outRangeMin = *xRangeMin;
    }
    OP_LOGD(context->GetNodeName(), "End to do Coordinates1DTo2DInferShapeRange");
    return ge::GRAPH_SUCCESS;
}

IMPL_OP_INFERSHAPE(Coordinates1DTo2D)
    .InferShape(Coordinates1DTo2DInferShape)
    .InferDataType(Coordinates1DTo2DInferDataType)
    .InferShapeRange(Coordinates1DTo2DInferShapeRange);
} // namespace ops
