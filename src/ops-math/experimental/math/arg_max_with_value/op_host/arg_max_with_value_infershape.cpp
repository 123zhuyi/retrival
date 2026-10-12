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
 * \file arg_max_with_value_infershape.cpp
 * \brief self-contained infershape for ArgMaxWithValue (dual output: indice + values)
 */
#include "register/op_impl_registry.h"
#include "log/log.h"
#include "infershape_reduce_util.h"

using namespace ge;

namespace ops {
static constexpr size_t ARG_IDX_X = 0;
static constexpr size_t ARG_OUT_INDICE = 0;
static constexpr size_t ARG_OUT_VALUE = 1;
static constexpr size_t ARG_ATTR_DIM = 0;
static constexpr size_t ARG_ATTR_KEEPDIMS = 1;

static ge::graphStatus InferShapeArgMaxWithValue(gert::InferShapeContext* context)
{
    OP_LOGD(context->GetNodeName(), "InferShapeArgMaxWithValue begin");
    const gert::Shape* xShape = context->GetInputShape(ARG_IDX_X);
    OP_CHECK_NULL_WITH_CONTEXT(context, xShape);
    gert::Shape* indiceShape = context->GetOutputShape(ARG_OUT_INDICE);
    OP_CHECK_NULL_WITH_CONTEXT(context, indiceShape);
    gert::Shape* valueShape = context->GetOutputShape(ARG_OUT_VALUE);
    OP_CHECK_NULL_WITH_CONTEXT(context, valueShape);

    auto attrs = context->GetAttrs();
    OP_CHECK_NULL_WITH_CONTEXT(context, attrs);
    const int64_t* dimPtr = attrs->GetAttrPointer<int64_t>(ARG_ATTR_DIM);
    OP_CHECK_NULL_WITH_CONTEXT(context, dimPtr);
    const bool* keepPtr = attrs->GetAttrPointer<bool>(ARG_ATTR_KEEPDIMS);
    bool keepDims = (keepPtr != nullptr) ? *keepPtr : false;

    // scalar input -> scalar outputs
    if (xShape->GetDimNum() == 0) {
        *valueShape = *xShape;
        *indiceShape = *xShape;
        return GRAPH_SUCCESS;
    }

    int64_t dim = *dimPtr;
    ge::graphStatus ret;
    if (keepDims) {
        ret = Ops::Base::ReduceDimsWithKeepDims<int64_t>(xShape, &dim, 1, valueShape);
    } else {
        ret = Ops::Base::ReduceDimsWithoutKeepDims<int64_t>(xShape, &dim, 1, valueShape);
    }
    OP_CHECK_IF(ret != GRAPH_SUCCESS, OP_LOGE(context->GetNodeName(), "reduce dims failed"), return ge::GRAPH_FAILED);
    *indiceShape = *valueShape;
    OP_LOGD(context->GetNodeName(), "InferShapeArgMaxWithValue end");
    return GRAPH_SUCCESS;
}

IMPL_OP_INFERSHAPE(ArgMaxWithValue).InferShape(InferShapeArgMaxWithValue);
} // namespace ops
