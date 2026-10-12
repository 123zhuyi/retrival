/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

/**
 * NOTE: Portions of this code were AI-generated and have been
 * technically reviewed for functional accuracy and security
 */

/**
 * \file inv_grad_infershape.cpp
 * \brief InvGrad shape inference: dx.shape = y.shape
 *
 * Shape rule: dx has the same shape as input y (== input dy).
 * Shape/dtype equality check between y and dy is performed during tiling.
 */

#include "register/op_impl_registry.h"
#include "exe_graph/runtime/infer_shape_context.h"

using namespace ge;

namespace ops {

static ge::graphStatus InferShape4InvGrad(gert::InferShapeContext* context)
{
    const gert::Shape* yShape = context->GetInputShape(0);
    if (yShape == nullptr) {
        return ge::GRAPH_FAILED;
    }

    gert::Shape* dxShape = context->GetOutputShape(0);
    if (dxShape == nullptr) {
        return ge::GRAPH_FAILED;
    }

    *dxShape = *yShape;

    return ge::GRAPH_SUCCESS;
}

IMPL_OP_INFERSHAPE(InvGrad).InferShape(InferShape4InvGrad);

} // namespace ops
