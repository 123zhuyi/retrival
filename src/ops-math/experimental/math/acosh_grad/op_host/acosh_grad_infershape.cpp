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
 * \file acosh_grad_infershape.cpp
 * \brief AcoshGrad InferShape — output z has same shape/dtype as y
 */

#include "register/op_impl_registry.h"
#include "exe_graph/runtime/infer_shape_context.h"
#include "log/log.h"

using namespace ge;

namespace ops {

static bool IsSameShape(const gert::Shape& lhs, const gert::Shape& rhs)
{
    if (lhs.GetDimNum() != rhs.GetDimNum()) {
        return false;
    }
    for (size_t i = 0; i < lhs.GetDimNum(); ++i) {
        if (lhs.GetDim(i) != rhs.GetDim(i)) {
            return false;
        }
    }
    return true;
}

static ge::graphStatus InferShapeAcoshGrad(gert::InferShapeContext* ctx)
{
    OP_CHECK_IF(ctx == nullptr, OP_LOGE(ctx, "context is nullptr"), return ge::GRAPH_FAILED);

    const gert::Shape* yShape = ctx->GetInputShape(0);
    if (yShape == nullptr) {
        OP_LOGE(ctx, "input y shape is nullptr");
        return ge::GRAPH_FAILED;
    }
    const gert::Shape* dyShape = ctx->GetInputShape(1);
    if (dyShape == nullptr) {
        OP_LOGE(ctx, "input dy shape is nullptr");
        return ge::GRAPH_FAILED;
    }
    if (!IsSameShape(*yShape, *dyShape)) {
        OP_LOGE(ctx, "shape of y and dy must be same");
        return ge::GRAPH_FAILED;
    }
    gert::Shape* zShape = ctx->GetOutputShape(0);
    if (zShape == nullptr) {
        OP_LOGE(ctx, "output dx shape is nullptr");
        return ge::GRAPH_FAILED;
    }
    *zShape = *yShape;
    return ge::GRAPH_SUCCESS;
}

IMPL_OP_INFERSHAPE(AcoshGrad).InferShape(InferShapeAcoshGrad);

} // namespace ops
