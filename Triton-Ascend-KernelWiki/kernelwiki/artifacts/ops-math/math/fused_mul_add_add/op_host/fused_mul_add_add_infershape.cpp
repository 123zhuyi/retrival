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
 * \file fused_mul_add_add_infershape.cpp
 * \brief
 */
#include "register/op_impl_registry.h"
#include "op_common/op_host/util/shape_util.h"
#include "log/log.h"

using namespace ge;
namespace ops {
static ge::graphStatus InferShape4FusedMulAddAdd(gert::InferShapeContext* context)
{
    OP_LOGD(context->GetNodeName(), "Begin to do InferShape4FusedMulAddAdd in ops-math");
    const gert::Shape* x1Shape = context->GetInputShape(0);
    OP_CHECK_NULL_WITH_CONTEXT(context, x1Shape);
    const gert::Shape* x2Shape = context->GetInputShape(1);
    OP_CHECK_NULL_WITH_CONTEXT(context, x2Shape);
    const gert::Shape* x3Shape = context->GetInputShape(2);
    OP_CHECK_NULL_WITH_CONTEXT(context, x3Shape);
    const gert::Shape* x4Shape = context->GetInputShape(3);
    OP_CHECK_NULL_WITH_CONTEXT(context, x4Shape);
    gert::Shape* yShape = context->GetOutputShape(0);
    OP_CHECK_NULL_WITH_CONTEXT(context, yShape);

    // Align with op_proto: dynamic shape is not supported.
    if (Ops::Base::IsUnknownRank(*x1Shape) || Ops::Base::IsUnknownShape(*x1Shape) ||
        Ops::Base::IsUnknownRank(*x2Shape) || Ops::Base::IsUnknownShape(*x2Shape) ||
        Ops::Base::IsUnknownRank(*x3Shape) || Ops::Base::IsUnknownShape(*x3Shape) ||
        Ops::Base::IsUnknownRank(*x4Shape) || Ops::Base::IsUnknownShape(*x4Shape)) {
        OP_LOGW(context->GetNodeName(), "Inputs do not support dynamic shape!");
        return ge::GRAPH_FAILED;
    }

    // Align with op_proto: output shape equals x1 shape (no broadcast).
    *yShape = *x1Shape;
    return ge::GRAPH_SUCCESS;
}
IMPL_OP_INFERSHAPE(FusedMulAddAdd).InferShape(InferShape4FusedMulAddAdd);
} // namespace ops
