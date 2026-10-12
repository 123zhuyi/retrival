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
 * \file cdist_infer.cpp
 * \brief
 */
#include "register/op_impl_registry.h"
#include "log/log.h"

using namespace ge;

namespace ops {
constexpr int64_t MIN_DIM_LEN = 2;

static ge::graphStatus InferShapeCdist(gert::InferShapeContext* context)
{
    OP_LOGD(context->GetNodeName(), "Begin to do InferShapeCdist");
    const gert::Shape* x1Shape = context->GetInputShape(0);
    OP_CHECK_NULL_WITH_CONTEXT(context, x1Shape);
    const gert::Shape* x2Shape = context->GetInputShape(1);
    OP_CHECK_NULL_WITH_CONTEXT(context, x2Shape);
    int64_t x1DimNum = x1Shape->GetDimNum();
    int64_t x2DimNum = x2Shape->GetDimNum();
    OP_CHECK_IF(x1DimNum < MIN_DIM_LEN || x2DimNum < MIN_DIM_LEN,
                OP_LOGE(context->GetNodeName(),
                "Only supports at least 2D tensors, X1 got: %ld, X2 got: %ld.", x1DimNum, x2DimNum),
                return ge::GRAPH_FAILED);

    gert::Shape* yShape = context->GetOutputShape(0);
    OP_CHECK_NULL_WITH_CONTEXT(context, yShape);

    yShape->SetDimNum(x1DimNum);
    for (int64_t i = 0; i < x1DimNum - 1; i++) {
        int64_t dim = x1Shape->GetDim(i);
        yShape->SetDim(i, dim);
    }
    yShape->SetDim(x1DimNum - 1, x2Shape->GetDim(x2DimNum - MIN_DIM_LEN));
    OP_LOGD(context->GetNodeName(), "End to do InferShapeCdist");
    return GRAPH_SUCCESS;
}

IMPL_OP_INFERSHAPE(Cdist).InferShape(InferShapeCdist);
} // namespace ops