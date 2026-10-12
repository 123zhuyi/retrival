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
 * \file tanh_grad_infer.cpp
 * \brief
 */

#include "register/op_impl_registry.h"
#include "log/log.h"

using namespace ge;

namespace ops {
static constexpr int64_t IDX_0 = 0;
static constexpr int64_t IDX_1 = 1;

static ge::graphStatus InferShapeTanhGrad(gert::InferShapeContext* context)
{
    OP_CHECK_IF(context == nullptr, OP_LOGE(context, "context is nullptr"), return ge::GRAPH_FAILED);
    OP_LOGD(context->GetNodeName(), "Begin to do InferShapeTanhGrad");

    // get input shapes
    const gert::Shape* yShape = context->GetInputShape(IDX_0);
    OP_CHECK_NULL_WITH_CONTEXT(context, yShape);
    const gert::Shape* dyShape = context->GetInputShape(IDX_1);
    OP_CHECK_NULL_WITH_CONTEXT(context, dyShape);

    // get output shapes
    gert::Shape* dxShape = context->GetOutputShape(IDX_0);
    OP_CHECK_NULL_WITH_CONTEXT(context, dxShape);

    // 填充输出shape大小
    *dxShape = *yShape;
    OP_LOGD(context->GetNodeName(), "End to do InferShapeTanhGrad");
    return GRAPH_SUCCESS;
}

IMPL_OP_INFERSHAPE(TanhGrad).InferShape(InferShapeTanhGrad);
} // namespace ops