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
 * \file softshrink_grad_infershape.cpp
 * \brief SoftShrinkGrad 算子形状推导实现
 */

#include "register/op_impl_registry.h"
#include "exe_graph/runtime/infer_shape_context.h"
#include "op_common/log/log.h"

using namespace ge;

namespace ops {

static ge::graphStatus InferShapeSoftShrinkGrad(gert::InferShapeContext* context)
{
    const gert::Shape* inputShape = context->GetInputShape(0);
    if (inputShape == nullptr) {
        OP_LOGE("SoftshrinkGrad", "GetInputShape(0) returns nullptr");
        return ge::GRAPH_FAILED;
    }

    const gert::Shape* selfShape = context->GetInputShape(1);
    if (selfShape == nullptr) {
        OP_LOGE("SoftshrinkGrad", "GetInputShape(1) returns nullptr");
        return ge::GRAPH_FAILED;
    }

    gert::Shape* outputShape = context->GetOutputShape(0);
    if (outputShape == nullptr) {
        OP_LOGE("SoftshrinkGrad", "GetOutputShape(0) returns nullptr");
        return ge::GRAPH_FAILED;
    }

    *outputShape = *inputShape;
    return ge::GRAPH_SUCCESS;
}

IMPL_OP_INFERSHAPE(SoftshrinkGrad).InferShape(InferShapeSoftShrinkGrad);

} // namespace ops
