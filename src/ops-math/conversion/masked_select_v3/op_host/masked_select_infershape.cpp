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
 * \file masked_select_infershape.cpp
 * \brief
 */
#include "log/log.h"
#include "register/op_impl_registry.h"

using namespace ge;

namespace {
constexpr size_t OUTPUT_SHAPE_RANK = 1;
constexpr int64_t UNKNOWN_DIM_VALUE_ = -1;
} // namespace
namespace ops {
static ge::graphStatus InferShape4MaskedSelect(gert::InferShapeContext* context)
{
    OP_LOGD(context, "InferShape4MaskedSelect running begin");
    auto number_shape = context->GetInputShape(0);
    OP_CHECK_NULL_WITH_CONTEXT(context, number_shape);
    auto out_shape = context->GetOutputShape(0);
    OP_CHECK_NULL_WITH_CONTEXT(context, out_shape);
    out_shape->SetDimNum(OUTPUT_SHAPE_RANK);
    out_shape->SetDim(0, -1);
    OP_LOGD(context, "InferShape4MaskedSelect run success");
    return GRAPH_SUCCESS;
}

static ge::graphStatus InferShapeRange4MaskedSelect(gert::InferShapeRangeContext* context)
{
    OP_LOGD(context, "InferShapeRange4MaskedSelect running begin");
    auto number_range = context->GetInputShapeRange(0);
    OP_CHECK_NULL_WITH_CONTEXT(context, number_range);
    auto out_range = context->GetOutputShapeRange(0);
    OP_CHECK_NULL_WITH_CONTEXT(context, out_range);

    OP_CHECK_NULL_WITH_CONTEXT(context, out_range->GetMax());
    OP_CHECK_NULL_WITH_CONTEXT(context, out_range->GetMin());
    OP_CHECK_NULL_WITH_CONTEXT(context, number_range->GetMax());

    int64_t total_num = number_range->GetMax()->GetShapeSize();
    out_range->GetMax()->SetDimNum(OUTPUT_SHAPE_RANK);
    out_range->GetMax()->SetDim(0, total_num);
    out_range->GetMin()->SetDimNum(OUTPUT_SHAPE_RANK);
    out_range->GetMin()->SetDim(0, 1);
    OP_LOGD(context, "InferShapeRange4MaskedSelect running success");

    return GRAPH_SUCCESS;
}

IMPL_OP_INFERSHAPE(MaskedSelect).InferShape(InferShape4MaskedSelect).InferShapeRange(InferShapeRange4MaskedSelect);
} // namespace ops