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

/*!
 * \file complex_v3_infershape.cpp
 * \brief ComplexV3 operator shape inference implementation
 *
 * Output shape = broadcast(real.shape, imag.shape)
 * Output dtype (DT_COMPLEX64/DT_COMPLEX32) is determined by the DataType list in op_def.
 */

#include "register/op_impl_registry.h"
#include "exe_graph/runtime/infer_shape_context.h"

using namespace ge;

namespace ops {

static ge::graphStatus InferShape4ComplexV3(gert::InferShapeContext* context)
{
    // Get input shapes
    const gert::Shape* realShape = context->GetInputShape(0);
    const gert::Shape* imagShape = context->GetInputShape(1);
    if (realShape == nullptr || imagShape == nullptr) {
        return ge::GRAPH_FAILED;
    }

    // Get output shape
    gert::Shape* outShape = context->GetOutputShape(0);
    if (outShape == nullptr) {
        return ge::GRAPH_FAILED;
    }

    // NumPy broadcast rules: right-align dimensions, take max per dim
    size_t realDims = realShape->GetDimNum();
    size_t imagDims = imagShape->GetDimNum();
    size_t outDims = std::max(realDims, imagDims);
    outShape->SetDimNum(outDims);

    for (size_t i = 0; i < outDims; i++) {
        int64_t realDim = (i < outDims - realDims) ? 1 : realShape->GetDim(i - (outDims - realDims));
        int64_t imagDim = (i < outDims - imagDims) ? 1 : imagShape->GetDim(i - (outDims - imagDims));

        if (realDim != imagDim && realDim != 1 && imagDim != 1) {
            return ge::GRAPH_FAILED; // Does not satisfy broadcast condition
        }
        outShape->SetDim(i, std::max(realDim, imagDim));
    }

    return ge::GRAPH_SUCCESS;
}

IMPL_OP_INFERSHAPE(ComplexV3).InferShape(InferShape4ComplexV3);

} // namespace ops
