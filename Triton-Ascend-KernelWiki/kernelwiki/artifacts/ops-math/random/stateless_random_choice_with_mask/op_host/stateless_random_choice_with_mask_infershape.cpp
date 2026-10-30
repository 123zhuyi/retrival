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
 * \file stateless_random_choice_with_mask_infershape.cpp
 * \brief
 */
#include "register/op_impl_registry.h"
#include "log/log.h"
#include "util/shape_util.h"

using namespace ge;
using namespace Ops::Base;

namespace {
constexpr size_t IN_IDX_X = 0;
constexpr size_t OUT_IDX_Y = 0;
constexpr size_t OUT_IDX_MASK = 1;
constexpr size_t OUTPUT_Y_RANK = 2;
constexpr size_t OUTPUT_MASK_RANK = 1;
} // namespace
namespace ops {
graphStatus InferShapeForStatelessRandomChoiceWithMask(gert::InferShapeContext* context)
{
    OP_LOGI(context->GetNodeName(), "Start InferShapeForStatelessRandomChoiceWithMask");
    const gert::Shape* xShape = context->GetInputShape(IN_IDX_X);
    OP_CHECK_NULL_WITH_CONTEXT(context, xShape);
    gert::Shape* yShape = context->GetOutputShape(OUT_IDX_Y);
    OP_CHECK_NULL_WITH_CONTEXT(context, yShape);
    gert::Shape* maskShape = context->GetOutputShape(OUT_IDX_MASK);
    OP_CHECK_NULL_WITH_CONTEXT(context, maskShape);

    if (IsUnknownRank(*xShape)) {
        SetUnknownRank(*yShape);
        SetUnknownRank(*maskShape);
        OP_LOGI(context->GetNodeName(), "End InferShapeForStatelessRandomChoiceWithMask (UNKNOWN RANK)");
        return ge::GRAPH_SUCCESS;
    }
    if (IsUnknownShape(*xShape)) {
        SetUnknownShape(OUTPUT_Y_RANK, *yShape);
        SetUnknownShape(OUTPUT_MASK_RANK, *maskShape);
        OP_LOGI(context->GetNodeName(), "End InferShapeForStatelessRandomChoiceWithMask (UNKNOWN SHAPE)");
        return ge::GRAPH_SUCCESS;
    }
    yShape->SetDimNum(OUTPUT_Y_RANK);
    yShape->SetDim(0, UNKNOWN_DIM);
    yShape->SetDim(1, static_cast<int64_t>(xShape->GetDimNum()));
    maskShape->SetDimNum(OUTPUT_MASK_RANK);
    maskShape->SetDim(0, UNKNOWN_DIM);

    OP_LOGI(context->GetNodeName(), "End InferShapeForStatelessRandomChoiceWithMask");
    return GRAPH_SUCCESS;
}

ge::graphStatus InferShapeRangeForStatelessRandomChoiceWithMask(gert::InferShapeRangeContext* context)
{
    OP_LOGI(context->GetNodeName(), "Start InferShapeRangeForStatelessRandomChoiceWithMask");
    auto xRange = context->GetInputShapeRange(IN_IDX_X);
    OP_CHECK_NULL_WITH_CONTEXT(context, xRange);
    auto yRange = context->GetOutputShapeRange(OUT_IDX_Y);
    OP_CHECK_NULL_WITH_CONTEXT(context, yRange);
    auto maskRange = context->GetOutputShapeRange(OUT_IDX_MASK);
    OP_CHECK_NULL_WITH_CONTEXT(context, maskRange);

    OP_CHECK_NULL_WITH_CONTEXT(context, xRange->GetMax());
    OP_CHECK_NULL_WITH_CONTEXT(context, yRange->GetMax());
    OP_CHECK_NULL_WITH_CONTEXT(context, yRange->GetMin());
    OP_CHECK_NULL_WITH_CONTEXT(context, maskRange->GetMax());
    OP_CHECK_NULL_WITH_CONTEXT(context, maskRange->GetMin());

    auto countTensorRange = context->GetInputTensorRange(1U);
    auto countTensor = countTensorRange->GetMax();
    auto countData = *(countTensor->GetData<int32_t>());

    int64_t outputLengthMax = xRange->GetMax()->GetShapeSize();
    outputLengthMax = outputLengthMax > countData ? outputLengthMax : countData;
    int64_t xDimNum = xRange->GetMax()->GetDimNum();

    yRange->GetMax()->SetDimNum(OUTPUT_Y_RANK);
    yRange->GetMax()->SetDim(0, outputLengthMax);
    yRange->GetMax()->SetDim(1, xDimNum);
    yRange->GetMin()->SetDimNum(OUTPUT_Y_RANK);
    yRange->GetMin()->SetDim(0, 0);
    yRange->GetMin()->SetDim(1, xDimNum);
    maskRange->GetMax()->SetDimNum(OUTPUT_MASK_RANK);
    maskRange->GetMax()->SetDim(0, outputLengthMax);
    maskRange->GetMin()->SetDimNum(OUTPUT_MASK_RANK);
    maskRange->GetMin()->SetDim(0, 0);

    OP_LOGD(context->GetNodeName(), "Get output length MAX %s.", ToString(*(yRange->GetMax())).c_str());
    OP_LOGD(context->GetNodeName(), "Get output length MIN %s.", ToString(*(yRange->GetMin())).c_str());
    OP_LOGI(context->GetNodeName(), "End InferShapeRangeForStatelessRandomChoiceWithMask");
    return ge::GRAPH_SUCCESS;
}

ge::graphStatus InferDataTypeForStatelessRandomChoiceWithMask(gert::InferDataTypeContext* context)
{
    OP_LOGI(context->GetNodeName(), "Start InferDataTypeForStatelessRandomChoiceWithMask");
    context->SetOutputDataType(OUT_IDX_Y, DataType::DT_INT32);
    context->SetOutputDataType(OUT_IDX_MASK, DataType::DT_BOOL);
    OP_LOGD(context->GetNodeName(), "Set output y dtype: %s", Ops::Base::ToString(DataType::DT_INT32).c_str());
    OP_LOGD(context->GetNodeName(), "Set output mask dtype: %s", Ops::Base::ToString(DataType::DT_BOOL).c_str());
    OP_LOGI(context->GetNodeName(), "End InferDataTypeForStatelessRandomChoiceWithMask");
    return ge::GRAPH_SUCCESS;
}

IMPL_OP(StatelessRandomChoiceWithMask).InferDataType(InferDataTypeForStatelessRandomChoiceWithMask);

IMPL_OP_INFERSHAPE(StatelessRandomChoiceWithMask)
    .InputsDataDependency({1})
    .InferShape(InferShapeForStatelessRandomChoiceWithMask)
    .InferShapeRange(InferShapeRangeForStatelessRandomChoiceWithMask)
    .InferDataType(InferDataTypeForStatelessRandomChoiceWithMask);
} // namespace ops
