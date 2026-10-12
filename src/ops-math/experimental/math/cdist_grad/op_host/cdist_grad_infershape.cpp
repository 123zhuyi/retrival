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
 * 我们正常的版权申明，下面是我们的备注
 *
 * NOTE: Portions of this code were AI-generated and have been
 * technically reviewed for functional accuracy and security
 */

/*!
 * \file cdist_grad_infershape.cpp
 * \brief CdistGrad InferShape: grad_x1 shape = x1 shape
 */
#include "register/op_impl_registry.h"
#include "exe_graph/runtime/infer_shape_context.h"

using namespace ge;

namespace ops {

static ge::graphStatus InferShape4CdistGrad(gert::InferShapeContext* context)
{
    // x1 is the second input (index=1)
    const gert::Shape* x1Shape = context->GetInputShape(1);
    if (x1Shape == nullptr) {
        return ge::GRAPH_FAILED;
    }

    // grad_x1 is the first output (index=0)
    gert::Shape* outputShape = context->GetOutputShape(0);
    if (outputShape == nullptr) {
        return ge::GRAPH_FAILED;
    }

    // grad_x1 shape = x1 shape
    int64_t dimNum = static_cast<int64_t>(x1Shape->GetDimNum());
    outputShape->SetDimNum(dimNum);
    for (int64_t i = 0; i < dimNum; i++) {
        outputShape->SetDim(i, x1Shape->GetDim(i));
    }

    return ge::GRAPH_SUCCESS;
}

IMPL_OP_INFERSHAPE(CdistGrad).InferShape(InferShape4CdistGrad);

} // namespace ops
