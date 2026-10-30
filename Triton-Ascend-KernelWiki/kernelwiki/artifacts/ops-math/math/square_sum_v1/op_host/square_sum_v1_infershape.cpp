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
 * \file square_sum_v1_infershape.cpp
 * \brief
 */
#include "op_host/infershape_reduce_util.h"
#include "log/log.h"
#include "register/op_impl_registry.h"

using namespace ge;
namespace ops {

static ge::graphStatus InferShape4SquareSumV1(gert::InferShapeContext* context)
{
    auto attrs = context->GetAttrs();
    OP_CHECK_NULL_WITH_CONTEXT(context, attrs);

    auto axis = attrs->GetAttrPointer<gert::ContinuousVector>(0);
    OP_CHECK_NULL_WITH_CONTEXT(context, axis);
    auto axes_size = static_cast<int32_t>(axis->GetSize());
    OP_CHECK_IF(
        axes_size < 1, OP_LOGE(context->GetNodeName(), "axes size must greater than 0!"), return ge::GRAPH_FAILED);
    auto axes_dims = static_cast<const int64_t*>(axis->GetData());

    const bool* keep_dims = attrs->GetAttrPointer<bool>(1);
    OP_CHECK_NULL_WITH_CONTEXT(context, keep_dims);

    auto in_shape = context->GetInputShape(0);
    OP_CHECK_NULL_WITH_CONTEXT(context, in_shape);
    auto out_shape = context->GetOutputShape(0);
    OP_CHECK_NULL_WITH_CONTEXT(context, out_shape);
    if (*keep_dims) {
        return Ops::Base::ReduceDimsWithKeepDims<int64_t>(in_shape, axes_dims, axes_size, out_shape);
    }
    return Ops::Base::ReduceDimsWithoutKeepDims<int64_t>(in_shape, axes_dims, axes_size, out_shape);
}

IMPL_OP_INFERSHAPE(SquareSumV1).InferShape(InferShape4SquareSumV1);
} // namespace ops