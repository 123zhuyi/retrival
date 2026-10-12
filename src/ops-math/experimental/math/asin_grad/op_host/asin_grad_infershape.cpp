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
 * \file asin_grad_infershape.cpp
 * \brief AsinGrad shape inference (z.shape = y.shape).
 */

#include "exe_graph/runtime/infer_shape_context.h"
#include "register/op_impl_registry.h"
#include "log/log.h"

namespace ops {

static ge::graphStatus InferShape4AsinGrad(gert::InferShapeContext* context)
{
    OP_CHECK_IF(context == nullptr, OP_LOGE(context, "context is nullptr"), return ge::GRAPH_FAILED);
    const gert::Shape* yShape = context->GetInputShape(0);
    const gert::Shape* dyShape = context->GetInputShape(1);
    gert::Shape* zShape = context->GetOutputShape(0);
    OP_CHECK_NULL_WITH_CONTEXT(context, yShape);
    OP_CHECK_NULL_WITH_CONTEXT(context, dyShape);
    OP_CHECK_NULL_WITH_CONTEXT(context, zShape);
    *zShape = *yShape;
    return ge::GRAPH_SUCCESS;
}

IMPL_OP_INFERSHAPE(AsinGrad).InferShape(InferShape4AsinGrad);

} // namespace ops
