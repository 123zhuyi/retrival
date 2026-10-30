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
 * \file right_shift_infershape.cpp
 * \brief RightShift infershape
 */

#include "register/op_impl_registry.h"
#include "op_host/infershape_broadcast_util.h"
#include "log/log.h"

using namespace ge;
using namespace Ops::Base;

namespace ops {
static ge::graphStatus InferShapeForRightShift(gert::InferShapeContext* context)
{
    OP_LOGI(context->GetNodeName(), "Begin to do InferShapeForRightShift");
    const gert::Shape* x_shape = context->GetInputShape(0);
    OP_CHECK_NULL_WITH_CONTEXT(context, x_shape);
    const gert::Shape* y_shape = context->GetInputShape(1);
    OP_CHECK_NULL_WITH_CONTEXT(context, y_shape);
    gert::Shape* z_shape = context->GetOutputShape(0);
    OP_CHECK_NULL_WITH_CONTEXT(context, z_shape);
    OP_CHECK_IF(
        !BroadcastShape(x_shape, y_shape, z_shape),
        OP_LOGE_FOR_INVALID_SHAPES_WITH_REASON(
            context->GetNodeName(), "x, y",
            (Ops::Base::ToString(*x_shape) + ", " + Ops::Base::ToString(*y_shape)),
            "The shapes of x and y must be broadcastable"),
        return ge::GRAPH_FAILED);
    return GRAPH_SUCCESS;
}

IMPL_OP_INFERSHAPE(RightShift).InferShape(InferShapeForRightShift);
} // namespace ops
