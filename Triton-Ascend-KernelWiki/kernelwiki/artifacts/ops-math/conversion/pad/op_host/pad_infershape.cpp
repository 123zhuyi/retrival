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
 * \file pad_infershape.cpp
 * \brief
 */

#include "register/op_impl_registry.h"
#include "op_api/op_util.h"
#include "log/log.h"
#include "util/shape_util.h"

using namespace ge;
using namespace Ops::Base;

namespace ops {
static constexpr size_t PAD_IN_IDX_X = 0;
static constexpr size_t PAD_IN_IDX_PADDINGS = 1;
static constexpr size_t PAD_OUT_IDX_Y = 0;
static constexpr size_t INT_DATA_2 = 2;
static constexpr int64_t UNKNOWN_DIM_VALUE_ = -1L;

template <typename T>
static bool PadInfershape(const gert::InferShapeContext* context, const gert::Shape* x_shape, const T* paddings_value,
                          const size_t paddings_size, gert::Shape* y_shape)
{
    OP_LOGD(context->GetNodeName(), "Begin to do PadInfershape");
    OP_LOGD(context->GetNodeName(), "input x = %s", ToString(*x_shape).c_str());
    size_t input_dim_size = x_shape->GetDimNum();
    OP_CHECK_IF(input_dim_size == 0, OP_LOGE(context->GetNodeName(), "input shape cannot empty"), return false);
    if (input_dim_size * INT_DATA_2 != paddings_size) {
        OP_LOGE(context->GetNodeName(),
                "the paddings num must be twice of the input x rank."
                "but paddings num is %zu, input x rank is %zu",
                paddings_size, input_dim_size);
        return false;
    }
    y_shape->SetDimNum(input_dim_size);
    int64_t dim_value = UNKNOWN_DIM_VALUE_;
    for (size_t i = 0; i < input_dim_size; ++i) {
        dim_value = x_shape->GetDim(i) == UNKNOWN_DIM_VALUE_ ?
                        UNKNOWN_DIM_VALUE_ :
                        x_shape->GetDim(i) + paddings_value[INT_DATA_2 * i] + paddings_value[INT_DATA_2 * i + 1];
        if (x_shape->GetDim(i) != UNKNOWN_DIM_VALUE_ && dim_value < 0) {
            OP_LOGE(context->GetNodeName(),
                    "The output shape at index %zu is %ld, but output shape CANNOT contain negative values. x_shape at "
                    "index %zu: %ld, corresponding pad_front: %ld, corresponding pad_end: %ld.",
                    i, dim_value, i, x_shape->GetDim(i), static_cast<int64_t>(paddings_value[INT_DATA_2 * i]),
                    static_cast<int64_t>(paddings_value[INT_DATA_2 * i + 1]));
            return false;
        }
        y_shape->SetDim(i, dim_value);
    }
    OP_LOGD(context->GetNodeName(), "output y = %s", ToString(*y_shape).c_str());
    OP_LOGD(context->GetNodeName(), "End to do PadInfershape");

    return true;
}

template <typename T>
ge::graphStatus PadInfershapeWithTensor(const gert::InferShapeContext* context, const gert::Shape* x_shape,
                                        const gert::Tensor* paddings_tensor, gert::Shape* y_shape)
{
    const T* paddings_value = paddings_tensor->GetData<T>();
    const size_t paddings_num = paddings_tensor->GetShapeSize();
    OP_CHECK_IF(!PadInfershape<T>(context, x_shape, paddings_value, paddings_num, y_shape),
                OP_LOGE(context->GetNodeName(), "do PadInfershape failed"), return ge::GRAPH_FAILED);

    return ge::GRAPH_SUCCESS;
}

static ge::graphStatus SetAllUnknownDim(const int64_t rank, gert::Shape* output_shape)
{
    OP_CHECK_IF(output_shape == nullptr, OP_LOGD("SetAllUnknownDim", "the output_shape is nullptr, return unsuccess"),
                return ge::GRAPH_FAILED);

    output_shape->SetDimNum(rank);
    for (int64_t i = 0; i < rank; ++i) {
        output_shape->SetDim(i, UNKNOWN_DIM_VALUE_);
    }
    OP_LOGD("SetAllUnknownDim", "set all dim = -1, output = %s", Ops::Base::ToString(*output_shape).c_str());

    return ge::GRAPH_SUCCESS;
}

static ge::graphStatus InferShapeForPad(gert::InferShapeContext* context)
{
    const gert::Shape* x_shape = context->GetInputShape(PAD_IN_IDX_X);
    OP_CHECK_NULL_WITH_CONTEXT(context, x_shape);
    gert::Shape* y_shape = context->GetOutputShape(PAD_OUT_IDX_Y);
    OP_CHECK_NULL_WITH_CONTEXT(context, y_shape);
    const gert::Tensor* paddings_tensor = context->GetInputTensor(PAD_IN_IDX_PADDINGS);
    OP_CHECK_NULL_WITH_CONTEXT(context, paddings_tensor);

    // if x_shape is unknown rank [-2] that means cannot know how many ranks,
    // which make output unknown rank.
    if (IsUnknownRank(*x_shape)) {
        SetUnknownRank(*y_shape);
        return ge::GRAPH_SUCCESS;
    }

    if (IsConstTensor(paddings_tensor)) {
        ge::DataType paddings_dtype = paddings_tensor->GetDataType();
        switch (paddings_dtype) {
            case ge::DT_INT32: {
                return PadInfershapeWithTensor<int32_t>(context, x_shape, paddings_tensor, y_shape);
            }
            case ge::DT_INT64: {
                return PadInfershapeWithTensor<int64_t>(context, x_shape, paddings_tensor, y_shape);
            }
            default:
                OP_LOGE_FOR_INVALID_DTYPE(context->GetNodeName(), "paddings",
                                          Ops::Base::ToString(paddings_dtype).c_str(), "int32 or int64");
                return ge::GRAPH_FAILED;
        }
    } else {
        return SetAllUnknownDim(x_shape->GetDimNum(), y_shape);
    }
}

IMPL_OP_INFERSHAPE(Pad).InferShape(InferShapeForPad).InputsDataDependency({PAD_IN_IDX_PADDINGS});
} // namespace ops
