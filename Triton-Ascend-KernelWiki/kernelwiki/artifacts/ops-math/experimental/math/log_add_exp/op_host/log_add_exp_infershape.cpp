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
 * \file log_add_exp_infershape.cpp
 * \brief LogAddExp shape inference implementation
 *
 * Implements NumPy broadcast rules for LogAddExp operator.
 */

#include "register/op_impl_registry.h"
#include "exe_graph/runtime/infer_shape_context.h"
#include <algorithm>

using namespace ge;

namespace ops {

static ge::graphStatus InferShape4LogAddExp(gert::InferShapeContext* context)
{
    // Get input shapes
    const gert::Shape* inputShapeX = context->GetInputShape(0);
    if (inputShapeX == nullptr) {
        return ge::GRAPH_FAILED;
    }
    const gert::Shape* inputShapeY = context->GetInputShape(1);
    if (inputShapeY == nullptr) {
        return ge::GRAPH_FAILED;
    }

    // Get output shape
    gert::Shape* outputShape = context->GetOutputShape(0);
    if (outputShape == nullptr) {
        return ge::GRAPH_FAILED;
    }

    // Implement NumPy broadcast rules
    int64_t dimX = inputShapeX->GetDimNum();
    int64_t dimY = inputShapeY->GetDimNum();
    int64_t maxDim = std::max(dimX, dimY);

    // Handle empty tensors
    if (dimX == 0 && dimY == 0) {
        // Both are scalars, output is scalar
        outputShape->SetDimNum(0);
        return ge::GRAPH_SUCCESS;
    }

    // Prepare output shape
    std::vector<int64_t> outDims(maxDim);

    // Broadcast from right to left
    for (int64_t i = 0; i < maxDim; i++) {
        int64_t dimIdxX = dimX - 1 - i;
        int64_t dimIdxY = dimY - 1 - i;
        int64_t outIdx = maxDim - 1 - i;

        int64_t sizeX = (dimIdxX >= 0) ? inputShapeX->GetDim(dimIdxX) : 1;
        int64_t sizeY = (dimIdxY >= 0) ? inputShapeY->GetDim(dimIdxY) : 1;

        // Broadcast rule: dimensions must be equal or one of them is 1
        if (sizeX == sizeY) {
            outDims[outIdx] = sizeX;
        } else if (sizeX == 1) {
            outDims[outIdx] = sizeY;
        } else if (sizeY == 1) {
            outDims[outIdx] = sizeX;
        } else {
            // Incompatible shapes
            return ge::GRAPH_FAILED;
        }
    }

    // Set output shape
    outputShape->SetDimNum(maxDim);
    for (int64_t i = 0; i < maxDim; i++) {
        outputShape->SetDim(i, outDims[i]);
    }

    return ge::GRAPH_SUCCESS;
}

IMPL_OP_INFERSHAPE(LogAddExp).InferShape(InferShape4LogAddExp);

} // namespace ops
