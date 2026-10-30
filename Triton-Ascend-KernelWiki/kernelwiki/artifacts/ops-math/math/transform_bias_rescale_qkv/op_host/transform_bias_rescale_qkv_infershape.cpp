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
 * \file transform_bias_rescale_qkv_infershape.cpp
 * \brief transform_bias_rescale_qkv operater graph infer resource
 */

#include "log/log.h"
#include "register/op_impl_registry.h"
#include "util/shape_util.h"

static constexpr int INPUT_NODE_NUM = 2;
static constexpr int OUTPUT_NODE_NUM = 3;
static constexpr int INDEX_ATTR_NUM_HEADS = 0;
static constexpr int INDEX_INPUT_QKV = 0;
static constexpr int INDEX_INPUT_QKV_BIAS = 1;
static constexpr int INDEX_OUTPUT_Q = 0;
static constexpr int INDEX_OUTPUT_K = 1;
static constexpr int INDEX_OUTPUT_V = 2;

static constexpr int QKV_NUM = 3;

static constexpr int INPUT_QKV_DIM0 = 0;
static constexpr int INPUT_QKV_DIM1 = 1;
static constexpr int INPUT_QKV_DIM2 = 2;

static constexpr int OUTPUT_QKV_DIM_NUM = 4;
static constexpr int OUTPUT_QKV_DIM0 = 0;
static constexpr int OUTPUT_QKV_DIM1 = 1;
static constexpr int OUTPUT_QKV_DIM2 = 2;
static constexpr int OUTPUT_QKV_DIM3 = 3;

using namespace ge;
namespace ops {

static ge::graphStatus InferShape4TransformBiasRescaleQkv(gert::InferShapeContext *context)
{
    if (context == nullptr) {
        OP_LOGE("InferShape4TransformBiasRescaleQkv", "Context is nullptr, check failed.");
        return GRAPH_FAILED;
    }
    if (context->GetComputeNodeInputNum() != INPUT_NODE_NUM ||
        context->GetComputeNodeOutputNum() != OUTPUT_NODE_NUM) {
        return GRAPH_FAILED;
    }

    auto* attrs = context->GetAttrs();
    OP_CHECK_NULL_WITH_CONTEXT(context, attrs);
    auto* num_heads_ptr = attrs->GetAttrPointer<int>(INDEX_ATTR_NUM_HEADS);
    OP_CHECK_NULL_WITH_CONTEXT(context, num_heads_ptr);
    auto num_heads = *num_heads_ptr;

    const gert::Shape* qkv_shape = context->GetInputShape(INDEX_INPUT_QKV);
    OP_CHECK_NULL_WITH_CONTEXT(context, qkv_shape);

    gert::Shape* q_shape = context->GetOutputShape(INDEX_OUTPUT_Q);
    OP_CHECK_NULL_WITH_CONTEXT(context, q_shape);
    gert::Shape* k_shape = context->GetOutputShape(INDEX_OUTPUT_K);
    OP_CHECK_NULL_WITH_CONTEXT(context, k_shape);
    gert::Shape* v_shape = context->GetOutputShape(INDEX_OUTPUT_V);
    OP_CHECK_NULL_WITH_CONTEXT(context, v_shape);

    q_shape->SetDimNum(OUTPUT_QKV_DIM_NUM);
    q_shape->SetDim(OUTPUT_QKV_DIM0, qkv_shape->GetDim(INPUT_QKV_DIM0));
    q_shape->SetDim(OUTPUT_QKV_DIM1, static_cast<int64_t>(num_heads));
    q_shape->SetDim(OUTPUT_QKV_DIM2, qkv_shape->GetDim(INPUT_QKV_DIM1));
    q_shape->SetDim(OUTPUT_QKV_DIM3, qkv_shape->GetDim(INPUT_QKV_DIM2) / QKV_NUM / num_heads);

    *k_shape = *q_shape;
    *v_shape = *q_shape;

    return ge::GRAPH_SUCCESS;
}

static ge::graphStatus InferDataType4TransformBiasRescaleQkv(gert::InferDataTypeContext* context)
{
    const ge::DataType qkv_data_type = context->GetInputDataType(0);
    context->SetOutputDataType(INDEX_OUTPUT_Q, qkv_data_type);
    context->SetOutputDataType(INDEX_OUTPUT_K, qkv_data_type);
    context->SetOutputDataType(INDEX_OUTPUT_V, qkv_data_type);
    return ge::GRAPH_SUCCESS;
}

IMPL_OP_INFERSHAPE(TransformBiasRescaleQkv)
    .InferShape(InferShape4TransformBiasRescaleQkv)
    .InferDataType(InferDataType4TransformBiasRescaleQkv);

} // namespace ops