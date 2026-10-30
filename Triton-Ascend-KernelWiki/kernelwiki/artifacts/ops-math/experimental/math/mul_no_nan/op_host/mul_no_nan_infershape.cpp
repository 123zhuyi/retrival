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
 *
 * NOTE: Portions of this code were AI-generated and have been
 * technically reviewed for functional accuracy and security
 */

/*!
 * \file mul_no_nan_infershape.cpp
 * \brief MulNoNan shape inference (broadcast rules)
 */

#include <algorithm>
#include "register/op_impl_registry.h"
#include "exe_graph/runtime/infer_shape_context.h"
#include "op_common/log/log.h"

using namespace ge;

namespace ops {

static ge::graphStatus InferShape4MulNoNan(gert::InferShapeContext* context)
{
    const gert::Shape* inputShapeX = context->GetInputShape(0);
    OP_CHECK_NULL_WITH_CONTEXT(context, inputShapeX);

    const gert::Shape* inputShapeY = context->GetInputShape(1);
    OP_CHECK_NULL_WITH_CONTEXT(context, inputShapeY);

    gert::Shape* outputShape = context->GetOutputShape(0);
    OP_CHECK_NULL_WITH_CONTEXT(context, outputShape);

    // Broadcast shape derivation: output shape = broadcast(x_shape, y_shape)
    // Following NumPy broadcast rules: align from the right, each dim must be equal or one of them is 1.
    size_t xDimNum = inputShapeX->GetDimNum();
    size_t yDimNum = inputShapeY->GetDimNum();
    size_t outDimNum = std::max(xDimNum, yDimNum);

    // Handle 0-dim scalar: output shape = the other input's shape
    if (xDimNum == 0) {
        *outputShape = *inputShapeY;
        return ge::GRAPH_SUCCESS;
    }
    if (yDimNum == 0) {
        *outputShape = *inputShapeX;
        return ge::GRAPH_SUCCESS;
    }

    outputShape->SetDimNum(outDimNum);

    for (size_t i = 0; i < outDimNum; ++i) {
        // Align from the right: index into x and y from their respective ends
        int64_t xDim = 1;
        int64_t yDim = 1;
        // 红线：数组访问前显式校验 xDimNum/yDimNum > 0，避免 (xDimNum - 1 - i) 减法下溢
        if (xDimNum > 0 && i < xDimNum) {
            xDim = inputShapeX->GetDim(xDimNum - 1 - i);
        }
        if (yDimNum > 0 && i < yDimNum) {
            yDim = inputShapeY->GetDim(yDimNum - 1 - i);
        }

        int64_t outDim = 0;
        if (xDim == yDim) {
            outDim = xDim;
        } else if (xDim == 1) {
            outDim = yDim;
        } else if (yDim == 1) {
            outDim = xDim;
        } else {
            OP_LOGE(
                context,
                "MulNoNan InferShape: shapes are not broadcast-compatible at dim %zu "
                "(from right): x_dim=%ld, y_dim=%ld",
                i, xDim, yDim);
            return ge::GRAPH_FAILED;
        }

        outputShape->SetDim(outDimNum - 1 - i, outDim);
    }

    return ge::GRAPH_SUCCESS;
}

IMPL_OP_INFERSHAPE(MulNoNan).InferShape(InferShape4MulNoNan);

} // namespace ops
