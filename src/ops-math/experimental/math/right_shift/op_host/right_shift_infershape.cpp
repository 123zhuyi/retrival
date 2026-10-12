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
 * \file right_shift_infershape.cpp
 * \brief RightShift infershape
 */

#include <algorithm>
#include <vector>
#include "register/op_impl_registry.h"
#include "log/log.h"

using namespace ge;

namespace ops {
namespace {
bool GetDimOrOne(const gert::Shape& shape, size_t dimIndex, int64_t& dim)
{
    if (shape.GetDimNum() == 0) {
        dim = 1;
        return true;
    }
    dim = shape.GetDim(dimIndex);
    return dim >= 0;
}

bool InferBroadcastDims(const gert::Shape& xShape, const gert::Shape& yShape, std::vector<int64_t>& outDims)
{
    size_t xRank = static_cast<size_t>(xShape.GetDimNum());
    size_t yRank = static_cast<size_t>(yShape.GetDimNum());
    size_t outRank = std::max(xRank, yRank);
    outDims.assign(outRank, 1);

    for (size_t i = 0; i < outRank; ++i) {
        int64_t xDim = 1;
        int64_t yDim = 1;
        if (i >= outRank - xRank) {
            size_t xIdx = i - (outRank - xRank);
            if (!GetDimOrOne(xShape, xIdx, xDim)) {
                return false;
            }
        }
        if (i >= outRank - yRank) {
            size_t yIdx = i - (outRank - yRank);
            if (!GetDimOrOne(yShape, yIdx, yDim)) {
                return false;
            }
        }
        if (xDim == yDim) {
            outDims[i] = xDim;
            continue;
        }
        if (xDim == 1) {
            outDims[i] = yDim;
            continue;
        }
        if (yDim == 1) {
            outDims[i] = xDim;
            continue;
        }
        return false;
    }
    return true;
}
} // namespace

static ge::graphStatus InferShapeForRightShift(gert::InferShapeContext* context)
{
    OP_CHECK_NULL_WITH_CONTEXT(context, context);
    const gert::Shape* xShape = context->GetInputShape(0);
    OP_CHECK_NULL_WITH_CONTEXT(context, xShape);
    const gert::Shape* yShape = context->GetInputShape(1);
    OP_CHECK_NULL_WITH_CONTEXT(context, yShape);
    gert::Shape* zShape = context->GetOutputShape(0);
    OP_CHECK_NULL_WITH_CONTEXT(context, zShape);

    std::vector<int64_t> outDims;
    OP_CHECK_IF(!InferBroadcastDims(*xShape, *yShape, outDims),
                OP_LOGE(context->GetNodeName(), "RightShift broadcast failed."), return ge::GRAPH_FAILED);

    // For scalar/scalar, keep scalar shape instead of [1].
    if (xShape->GetDimNum() == 0 && yShape->GetDimNum() == 0) {
        zShape->SetDimNum(0);
        return ge::GRAPH_SUCCESS;
    }

    zShape->SetDimNum(outDims.size());
    for (size_t i = 0; i < outDims.size(); ++i) {
        zShape->SetDim(i, outDims[i]);
    }
    return ge::GRAPH_SUCCESS;
}

IMPL_OP_INFERSHAPE(RightShift).InferShape(InferShapeForRightShift);
} // namespace ops
