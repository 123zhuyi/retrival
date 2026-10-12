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
 * \file pad_v3_grad.cc
 * \brief
 */
#include "log/log.h"
#include "register/op_impl_registry.h"
#include "util/shape_util.h"
#include "op_api/op_util.h"

using namespace ge;

namespace {
constexpr size_t INDEX_X = 0;
constexpr size_t INDEX_PADDINGS = 1;
constexpr size_t INDEX_Y = 0;
constexpr size_t INDEX_PADDINGS_CONTIGUOUS = 1;
constexpr size_t PAIR = 2;
static constexpr int64_t UNKNOWN_DIM_VALUE_ = -1L;
} // namespace

namespace ops {
template <typename T>
static ge::graphStatus PadV3GradInfershape(const gert::InferShapeContext* context, const gert::Shape* x_shape,
                                           const gert::Tensor* paddings_tensor, gert::Shape* y_shape)
{
    const T* paddings_value = paddings_tensor->GetData<T>();
    const size_t paddings_num = static_cast<size_t>(paddings_tensor->GetShapeSize());
    auto attrs = context->GetAttrs();
    OP_CHECK_NULL_WITH_CONTEXT(context, attrs);
    const bool* paddings_contiguous = attrs->GetAttrPointer<bool>(INDEX_PADDINGS_CONTIGUOUS);
    OP_CHECK_NULL_WITH_CONTEXT(context, paddings_contiguous);
    OP_LOGD(context->GetNodeName(), "Begin to do PadV3GradInfershape");
    OP_LOGD(context->GetNodeName(), "input x = %s", Ops::Base::ToString(*x_shape).c_str());
    // input shape check
    size_t input_dim_size = x_shape->GetDimNum();
    OP_CHECK_IF(input_dim_size == 0,
                OP_LOGE_FOR_INVALID_SHAPE_WITH_REASON(
                    context->GetNodeName(), "x", Ops::Base::ToString(*x_shape).c_str(), "x cannot be an empty tensor"),
                return ge::GRAPH_FAILED);
    // pad size check
    if (input_dim_size * PAIR != paddings_num) {
        OP_LOGE_FOR_INVALID_SHAPESIZE_WITH_REASON(
            context->GetNodeName(), "paddings", std::to_string(paddings_num).c_str(),
            "The shape size of paddings should be equal to twice of the input x rank");
        return ge::GRAPH_FAILED;
    }
    // infer by paddings_contiguous
    y_shape->SetDimNum(input_dim_size);
    int64_t index_cof = 1;
    int64_t index_offset = input_dim_size;
    if (*paddings_contiguous) {
        index_cof = PAIR;
        index_offset = 1;
    }
    for (size_t i = 0; i < input_dim_size; ++i) {
        auto pad_front = paddings_value[index_cof * i];
        auto pad_end = paddings_value[index_cof * i + index_offset];

        int64_t dim_value = x_shape->GetDim(i) == UNKNOWN_DIM_VALUE_ ? UNKNOWN_DIM_VALUE_ :
                                                                       (x_shape->GetDim(i) - pad_front - pad_end);
        if (x_shape->GetDim(i) != UNKNOWN_DIM_VALUE_ && dim_value < 0) {
            OP_LOGE_FOR_INVALID_SHAPEDIM_WITH_REASON(context->GetNodeName(), "y", std::to_string(dim_value).c_str(),
                                                     "The shape dim of y must be greater than or equal to 0");
            return ge::GRAPH_FAILED;
        }
        y_shape->SetDim(i, dim_value);
    }
    OP_LOGD(context->GetNodeName(), "output y = %s", Ops::Base::ToString(*y_shape).c_str());
    OP_LOGD(context->GetNodeName(), "End to do PadV3GradInfershape");
    return ge::GRAPH_SUCCESS;
}

static ge::graphStatus SetAllUnknownDim(const int64_t rank, gert::Shape* output_shape)
{
    output_shape->SetDimNum(rank);
    for (int64_t i = 0; i < rank; ++i) {
        output_shape->SetDim(i, UNKNOWN_DIM_VALUE_);
    }
    OP_LOGD("SetAllUnknownDim", "set all dim = -1, output = %s", Ops::Base::ToString(*output_shape).c_str());

    return ge::GRAPH_SUCCESS;
}

static ge::graphStatus InferShape4PadV3Grad(gert::InferShapeContext* context)
{
    const gert::Shape* x_shape = context->GetInputShape(INDEX_X);
    OP_CHECK_NULL_WITH_CONTEXT(context, x_shape);
    gert::Shape* y_shape = context->GetOutputShape(INDEX_Y);
    OP_CHECK_NULL_WITH_CONTEXT(context, y_shape);

    if (Ops::Base::IsUnknownRank(*x_shape)) {
        Ops::Base::SetUnknownRank(*y_shape);
        return GRAPH_SUCCESS;
    }

    const gert::Tensor* paddings_tensor = context->GetInputTensor(INDEX_PADDINGS);
    OP_CHECK_NULL_WITH_CONTEXT(context, paddings_tensor);

    if (!IsConstTensor(paddings_tensor)) {
        return SetAllUnknownDim(x_shape->GetDimNum(), y_shape);
    }

    ge::DataType paddings_dtype = paddings_tensor->GetDataType();
    switch (paddings_dtype) {
        case ge::DT_INT32: {
            return PadV3GradInfershape<int32_t>(context, x_shape, paddings_tensor, y_shape);
        }
        case ge::DT_INT64: {
            return PadV3GradInfershape<int64_t>(context, x_shape, paddings_tensor, y_shape);
        }
        default:
            OP_LOGE_FOR_INVALID_DTYPE(context->GetNodeName(), "paddings", Ops::Base::ToString(paddings_dtype).c_str(),
                                      "int32 or int64");
            return ge::GRAPH_FAILED;
    }
    return ge::GRAPH_FAILED;
}

IMPL_OP_INFERSHAPE(PadV3Grad).InferShape(InferShape4PadV3Grad).InputsDataDependency({INDEX_PADDINGS});
} // namespace ops
