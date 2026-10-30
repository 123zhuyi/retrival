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
 * \file feeds_repeat_infershape.cpp
 * \brief
 */
#include "log/log.h"
#include "register/op_impl_registry.h"

using namespace ge;


namespace ops {
static constexpr size_t IDX_0 = 0;
static constexpr size_t IDX_1 = 1;
constexpr int64_t UNKNOWN_RANK_DIM_VALUE_ = -2;

inline bool IsUnknownRankForFeeds(const gert::Shape* checkShape)
{
    return checkShape->GetDimNum() == 1 && checkShape->GetDim(0) == UNKNOWN_RANK_DIM_VALUE_;
}

inline ge::graphStatus SetUnknownRankForFeeds(gert::Shape* outShape)
{
    outShape->SetDimNum(0);
    outShape->AppendDim(UNKNOWN_RANK_DIM_VALUE_);
    return ge::GRAPH_SUCCESS;
}

static ge::graphStatus InferShape4FeedsRepeat(gert::InferShapeContext* context)
{
    OP_LOGD(context, "Begin to do InferShape4FeedsRepeat");
    const gert::Shape* feeds_shape = context->GetInputShape(IDX_0);
    OP_CHECK_IF(
        feeds_shape == nullptr, OP_LOGE(context->GetNodeName(), "get feedsrepeat feeds_shape failed"),
        return ge::GRAPH_FAILED);
    OP_CHECK_NULL_WITH_CONTEXT(context, feeds_shape);
    const gert::Shape* repeat_times_shape = context->GetInputShape(IDX_1);
    OP_CHECK_IF(
        repeat_times_shape == nullptr, OP_LOGE(context->GetNodeName(), "get feedsrepeat repeat_times_shape failed"),
        return ge::GRAPH_FAILED);
    OP_CHECK_NULL_WITH_CONTEXT(context, repeat_times_shape);
    gert::Shape* y_shape = context->GetOutputShape(IDX_0);
    OP_CHECK_IF(
        y_shape == nullptr, OP_LOGE(context->GetNodeName(), "get feedsrepeat y_shape failed"), return ge::GRAPH_FAILED);
    OP_CHECK_NULL_WITH_CONTEXT(context, y_shape);
    const gert::RuntimeAttrs* attrs = context->GetAttrs();
    OP_CHECK_IF(
        attrs == nullptr, OP_LOGE(context->GetNodeName(), "get feedsrepeat runtime attrs failed"),
        return ge::GRAPH_FAILED);
    OP_CHECK_NULL_WITH_CONTEXT(context, attrs);

    if (IsUnknownRankForFeeds(y_shape)) { // [-2]输入
        OP_LOGD(context, "Input shape is -2, set output shape to -2");
        return SetUnknownRankForFeeds(y_shape);
    } else {
        *y_shape = *feeds_shape;
        const int64_t* output_feeds_size = attrs->GetAttrPointer<int64_t>(0);
        y_shape->SetDim(0, *output_feeds_size);
    }
    OP_LOGD(context, "End to do InferShape4FeedsRepeat");
    return GRAPH_SUCCESS;
}

static ge::graphStatus InferDataType4FeedsRepeat(gert::InferDataTypeContext* context)
{
    OP_LOGD(context, "Begin to do InferDataType4FeedsRepeat");
    OP_LOGD(context, "input feeds dtype: %s", Ops::Base::ToString(context->GetInputDataType(IDX_0)).c_str());
    OP_LOGD(
        context, "input feeds_repeat_times dtype: %s", Ops::Base::ToString(context->GetInputDataType(IDX_1)).c_str());
    context->SetOutputDataType(IDX_0, context->GetInputDataType(IDX_0));
    OP_LOGD(context, "End to do InferDataType4FeedsRepeat");
    return GRAPH_SUCCESS;
}

IMPL_OP_INFERSHAPE(FeedsRepeat).InferShape(InferShape4FeedsRepeat).InferDataType(InferDataType4FeedsRepeat);
} // namespace ops