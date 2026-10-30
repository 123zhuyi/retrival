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
 * \file top_k_v2_infershape.cpp
 * \brief
 */
#include "register/op_impl_registry.h"
#include "util/shape_util.h"
#include "log/log.h"

#include <string>
static constexpr int OUTPUT_VALUES_INDEX = 0;
static constexpr int OUTPUT_INDICES_INDEX = 1;
static constexpr int INPUT_X_INDEX = 0;
using namespace ge;
namespace ops {
static bool InferShapeForTopKCommon(gert::InferShapeContext* context, int64_t k, const int64_t* dim)
{
    const gert::Shape* input_x_shape = context->GetInputShape(INPUT_X_INDEX);
    OP_CHECK_NULL_WITH_CONTEXT(context, input_x_shape);
    size_t dim_size = input_x_shape->GetDimNum();
    if (dim_size <= 0) {
        OP_LOGE_FOR_INVALID_SHAPEDIM_WITH_REASON(context->GetNodeName(), "x", (std::to_string(dim_size) + "D").c_str(),
                                                 "The shape dim of x must be greater than 0.");
        return GRAPH_FAILED;
    }
    int64_t sorted_axis = dim_size - 1;

    if (dim != nullptr) {
        sorted_axis = *dim;
        if (sorted_axis < 0) {
            sorted_axis += dim_size;
        }
        if (sorted_axis >= static_cast<int64_t>(dim_size)) {
            OP_LOGE_WITH_INVALID_ATTR(
                context->GetNodeName(), "dim", std::to_string(*dim).c_str(),
                ("[-" + std::to_string(dim_size) + ", " + std::to_string(static_cast<int64_t>(dim_size) - 1) + "]")
                    .c_str());
            return GRAPH_FAILED;
        }
    }

    gert::Shape* output_values_shape = context->GetOutputShape(OUTPUT_VALUES_INDEX);
    OP_CHECK_NULL_WITH_CONTEXT(context, output_values_shape);
    gert::Shape* output_indices_shape = context->GetOutputShape(OUTPUT_INDICES_INDEX);
    OP_CHECK_NULL_WITH_CONTEXT(context, output_indices_shape);

    output_values_shape->SetDimNum(dim_size);
    output_indices_shape->SetDimNum(dim_size);
    for (size_t i = 0; i < dim_size; i++) {
        if (static_cast<int64_t>(i) == sorted_axis) {
            output_values_shape->SetDim(i, k);
            output_indices_shape->SetDim(i, k);
            continue;
        }
        output_values_shape->SetDim(i, input_x_shape->GetDim(i));
        output_indices_shape->SetDim(i, input_x_shape->GetDim(i));
    }
    return GRAPH_SUCCESS;
}

static graphStatus InferShapeForTopKV2D(gert::InferShapeContext* context)
{
    OP_LOGD(context->GetNodeName(), "Begin to do TopKV2DInferShape");
    const gert::RuntimeAttrs* attrs = context->GetAttrs();
    OP_CHECK_NULL_WITH_CONTEXT(context, attrs);
    const int64_t* dim = attrs->GetInt(1);
    const gert::Tensor* input_k_tensor = context->GetInputTensor(1);
    OP_CHECK_NULL_WITH_CONTEXT(context, input_k_tensor);
    DataType input_k_dtype = input_k_tensor->GetDataType();
    if (input_k_dtype == DT_INT32) {
        const int32_t* k = input_k_tensor->GetData<int32_t>();
        OP_CHECK_NULL_WITH_CONTEXT(context, k);
        return InferShapeForTopKCommon(context, *k, dim);
    } else if (input_k_dtype == DT_INT64) {
        const int64_t* k = input_k_tensor->GetData<int64_t>();
        OP_CHECK_NULL_WITH_CONTEXT(context, k);
        return InferShapeForTopKCommon(context, *k, dim);
    } else {
        OP_LOGE_FOR_INVALID_DTYPE(context->GetNodeName(), "k", Ops::Base::ToString(input_k_dtype).c_str(),
                                  "INT32 or INT64");
        return GRAPH_FAILED;
    }
}

IMPL_OP_INFERSHAPE(TopKV2).InferShape(InferShapeForTopKV2D).InputsDataDependency({1});
} // namespace ops
