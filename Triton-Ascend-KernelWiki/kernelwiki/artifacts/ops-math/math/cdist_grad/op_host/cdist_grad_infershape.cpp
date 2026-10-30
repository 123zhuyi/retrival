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
 * \file cdist_grad_infershape.cpp
 * \brief InferShape for CdistGrad operator
 *
 * CdistGrad reduces along axis -2 (without keepdims).
 * Input shape:  [..., P, R, M] (after broadcast)
 * Output shape: [..., P, M]   (R dimension is reduced away)
 */

#include "log/log.h"
#include "register/op_impl_registry.h"

using namespace ge;

namespace ops {
constexpr int64_t NUMBER_TWO = 2;

static ge::graphStatus InferShape4CdistGrad(gert::InferShapeContext* context)
{
    OP_LOGD(context->GetNodeName(), "Begin to do InferShape4CdistGrad");
    const gert::Shape* gradShape = context->GetInputShape(0);
    OP_CHECK_NULL_WITH_CONTEXT(context, gradShape);
    const gert::Shape* x1Shape = context->GetInputShape(1);
    OP_CHECK_NULL_WITH_CONTEXT(context, x1Shape);
    const gert::Shape* cdistShape = context->GetInputShape(3);
    OP_CHECK_NULL_WITH_CONTEXT(context, cdistShape);

    int64_t dimNum = gradShape->GetDimNum();
    OP_CHECK_IF(dimNum < NUMBER_TWO,
                OP_LOGE(context->GetNodeName(),
                        "Only supports at least 2D tensors, got: %ld.", dimNum),
                return ge::GRAPH_FAILED);

    gert::Shape* yShape = context->GetOutputShape(0);
    OP_CHECK_NULL_WITH_CONTEXT(context, yShape);

    // CdistGrad reduces on axis -2: output has dimNum - 1 dimensions
    // Output dims: grad_shape[0..dimNum-3] + grad_shape[dimNum-1]
    yShape->SetDimNum(dimNum - 1);
    for (int64_t i = 0; i < dimNum - NUMBER_TWO; i++) {
        yShape->SetDim(i, gradShape->GetDim(i));
    }
    yShape->SetDim(dimNum - NUMBER_TWO, gradShape->GetDim(dimNum - 1));

    OP_LOGD(context->GetNodeName(), "End to do InferShape4CdistGrad");
    return GRAPH_SUCCESS;
}

IMPL_OP_INFERSHAPE(CdistGrad).InferShape(InferShape4CdistGrad);
} // namespace ops
