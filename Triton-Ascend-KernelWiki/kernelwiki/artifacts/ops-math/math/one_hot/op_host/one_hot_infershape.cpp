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
 * \file one_hot_infershape.cpp
 * \brief
 */
#include "infershape_broadcast_util.h"
#include "register/op_impl_registry.h"
#include "log/log.h"
#include "op_host/util/const_util.h"

using namespace ge;

namespace {
const int32_t INDEX_INPUT_X = 0;
const int32_t INDEX_INPUT_DEPTH = 1;
const int32_t INDEX_INPUT_ON_VALUE = 2;
const int32_t INDEX_INPUT_OFF_VALUE = 3;
const int32_t INDEX_OUTPUT_Y = 0;
const int32_t INDEX_ATTR_AXIS = 0;
} // namespace

namespace ops {
static ge::graphStatus InferShape4OneHot(gert::InferShapeContext* context)
{
    const auto x_shape_ptr = context->GetInputShape(INDEX_INPUT_X);
    OP_CHECK_NULL_WITH_CONTEXT(context, x_shape_ptr);
    const auto depth_shape_ptr = context->GetInputShape(INDEX_INPUT_DEPTH);
    OP_CHECK_NULL_WITH_CONTEXT(context, depth_shape_ptr);
    const auto on_value_shape_ptr = context->GetInputShape(INDEX_INPUT_ON_VALUE);
    OP_CHECK_NULL_WITH_CONTEXT(context, on_value_shape_ptr);
    const auto off_value_shape_ptr = context->GetInputShape(INDEX_INPUT_OFF_VALUE);
    OP_CHECK_NULL_WITH_CONTEXT(context, off_value_shape_ptr);
    auto y_shape_ptr = context->GetOutputShape(INDEX_OUTPUT_Y);
    OP_CHECK_NULL_WITH_CONTEXT(context, y_shape_ptr);
    const auto attrs = context->GetAttrs();
    OP_CHECK_NULL_WITH_CONTEXT(context, attrs);
    const auto axis_ptr = attrs->GetAttrPointer<int64_t>(INDEX_ATTR_AXIS);
    OP_CHECK_NULL_WITH_CONTEXT(context, axis_ptr);

    int64_t axis = *axis_ptr;
    OP_CHECK_IF(axis < -1, OP_LOGE(context->GetNodeName(), "attr axis(%ld) must be >= -1", axis), return GRAPH_FAILED);

    int64_t dimnum = static_cast<int64_t>(x_shape_ptr->GetDimNum());
    OP_CHECK_IF(
        axis > dimnum, OP_LOGE(context->GetNodeName(), "attr axis(%ld) must be < %ld", axis, dimnum),
        return GRAPH_FAILED);

    int64_t depth_value = -1;
    OP_CHECK_IF(
        !Ops::Base::GetConstInt(context, INDEX_INPUT_DEPTH, depth_value),
        OP_LOGE(context->GetNodeName(), "Get depth const tensor failed"), return ge::GRAPH_FAILED);

    // update output shape
    y_shape_ptr->SetDimNum(dimnum + 1);
    if (axis == -1) {
        for (int32_t i = 0; i < dimnum; i++) {
            y_shape_ptr->SetDim(i, x_shape_ptr->GetDim(i));
        }
        y_shape_ptr->SetDim(static_cast<size_t>(dimnum), depth_value);
    } else {
        while (dimnum > axis) {
            y_shape_ptr->SetDim(dimnum, x_shape_ptr->GetDim(dimnum - 1));
            dimnum--;
        }
        y_shape_ptr->SetDim(axis, depth_value);
        for (int32_t i = 0; i < axis; i++) {
            y_shape_ptr->SetDim(i, x_shape_ptr->GetDim(i));
        }
    }

    return ge::GRAPH_SUCCESS;
}

IMPL_OP_INFERSHAPE(OneHot).InferShape(InferShape4OneHot).InputsDataDependency({INDEX_INPUT_DEPTH});
} // namespace ops