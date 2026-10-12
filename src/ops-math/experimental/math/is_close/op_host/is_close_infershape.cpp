/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include <algorithm>

#include "log/log.h"
#include "register/op_impl_registry.h"
#include "../op_kernel/is_close_tiling_data.h"

using namespace ge;

namespace ops {
namespace {
constexpr size_t X1_IDX = 0;
constexpr size_t X2_IDX = 1;
constexpr size_t Y_IDX = 0;

bool ReadAlignedShape(const gert::Shape& shape, size_t rank, uint64_t (&dims)[IS_CLOSE_MAX_BROADCAST_DIM])
{
    if (shape.GetDimNum() > rank || rank > IS_CLOSE_MAX_BROADCAST_DIM) {
        return false;
    }
    for (size_t i = 0; i < rank; ++i) {
        dims[i] = 1;
    }
    size_t dimOffset = rank - shape.GetDimNum();
    for (size_t i = 0; i < shape.GetDimNum(); ++i) {
        int64_t dim = shape.GetDim(i);
        if (dim < 0) {
            return false;
        }
        dims[dimOffset + i] = static_cast<uint64_t>(dim);
    }
    return true;
}

bool InferBroadcastShape(
    const gert::Shape& x1Shape, const gert::Shape& x2Shape, uint64_t (&outDims)[IS_CLOSE_MAX_BROADCAST_DIM],
    size_t& rank)
{
    rank = std::max(x1Shape.GetDimNum(), x2Shape.GetDimNum());
    if (rank > IS_CLOSE_MAX_BROADCAST_DIM) {
        return false;
    }

    uint64_t x1Dims[IS_CLOSE_MAX_BROADCAST_DIM] = {};
    uint64_t x2Dims[IS_CLOSE_MAX_BROADCAST_DIM] = {};
    if (!ReadAlignedShape(x1Shape, rank, x1Dims) || !ReadAlignedShape(x2Shape, rank, x2Dims)) {
        return false;
    }

    for (size_t i = 0; i < rank; ++i) {
        uint64_t x1Dim = x1Dims[i];
        uint64_t x2Dim = x2Dims[i];
        if (x1Dim != x2Dim && x1Dim != 1 && x2Dim != 1) {
            return false;
        }
        outDims[i] = (x1Dim == 0 || x2Dim == 0) ? 0 : std::max(x1Dim, x2Dim);
    }
    return true;
}
} // namespace

static ge::graphStatus InferShape4IsClose(gert::InferShapeContext* context)
{
    OP_LOGD(context->GetNodeName(), "Begin to do InferShape4IsClose");
    const gert::Shape* x1Shape = context->GetInputShape(X1_IDX);
    OP_CHECK_NULL_WITH_CONTEXT(context, x1Shape);
    const gert::Shape* x2Shape = context->GetInputShape(X2_IDX);
    OP_CHECK_NULL_WITH_CONTEXT(context, x2Shape);

    uint64_t outDims[IS_CLOSE_MAX_BROADCAST_DIM] = {};
    size_t rank = 0;
    OP_CHECK_IF(
        !InferBroadcastShape(*x1Shape, *x2Shape, outDims, rank),
        OP_LOGE(context->GetNodeName(), "x1 and x2 shape should be broadcastable and rank should be <= 8."),
        return ge::GRAPH_FAILED);

    gert::Shape* yShape = context->GetOutputShape(Y_IDX);
    OP_CHECK_NULL_WITH_CONTEXT(context, yShape);
    yShape->SetDimNum(rank);
    for (size_t i = 0; i < rank; ++i) {
        yShape->SetDim(i, static_cast<int64_t>(outDims[i]));
    }
    OP_LOGD(context->GetNodeName(), "End to do InferShape4IsClose");
    return ge::GRAPH_SUCCESS;
}

static ge::graphStatus InferDataType4IsClose(gert::InferDataTypeContext* context)
{
    context->SetOutputDataType(Y_IDX, ge::DT_BOOL);
    return ge::GRAPH_SUCCESS;
}

IMPL_OP_INFERSHAPE(IsClose).InferShape(InferShape4IsClose).InferDataType(InferDataType4IsClose);
} // namespace ops
