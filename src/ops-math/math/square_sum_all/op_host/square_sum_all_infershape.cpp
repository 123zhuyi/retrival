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
 * \file square_sum_all_infershape.cpp
 * \brief SquareSumAll shape inference.
 */

#include <string>

#include "log/log.h"
#include "op_common/op_host/util/shape_util.h"
#include "register/op_impl_registry.h"

namespace ops {
namespace {
constexpr size_t X1_INDEX = 0;
constexpr size_t X2_INDEX = 1;
constexpr size_t Y1_INDEX = 0;
constexpr size_t Y2_INDEX = 1;
constexpr size_t MAX_RANK = 8;
} // namespace

static ge::graphStatus InferShapeForSquareSumAll(gert::InferShapeContext* context)
{
    const gert::Shape* x1Shape = context->GetInputShape(X1_INDEX);
    OP_CHECK_NULL_WITH_CONTEXT(context, x1Shape);
    const gert::Shape* x2Shape = context->GetInputShape(X2_INDEX);
    OP_CHECK_NULL_WITH_CONTEXT(context, x2Shape);
    gert::Shape* y1Shape = context->GetOutputShape(Y1_INDEX);
    OP_CHECK_NULL_WITH_CONTEXT(context, y1Shape);
    gert::Shape* y2Shape = context->GetOutputShape(Y2_INDEX);
    OP_CHECK_NULL_WITH_CONTEXT(context, y2Shape);

    const bool x1UnknownRank = Ops::Base::IsUnknownRank(*x1Shape);
    const bool x2UnknownRank = Ops::Base::IsUnknownRank(*x2Shape);
    if (!x1UnknownRank && !x2UnknownRank) {
        const size_t x1Rank = x1Shape->GetDimNum();
        const size_t x2Rank = x2Shape->GetDimNum();
        const std::string actualRanks = std::to_string(x1Rank) + ", " + std::to_string(x2Rank);
        OP_CHECK_IF(x1Rank != x2Rank,
                    OP_LOGE_FOR_INVALID_SHAPEDIMS_WITH_REASON(context->GetNodeName(), "x1, x2", actualRanks,
                                                              "input ranks must be equal"),
                    return ge::GRAPH_FAILED);
        OP_CHECK_IF(x1Rank > MAX_RANK,
                    OP_LOGE_FOR_INVALID_SHAPEDIM_WITH_REASON(context->GetNodeName(), "x1, x2", std::to_string(x1Rank),
                                                             "input rank must be in [0, 8]"),
                    return ge::GRAPH_FAILED);

        const std::string inputShapes = Ops::Base::ToString(*x1Shape) + ", " + Ops::Base::ToString(*x2Shape);

        for (size_t i = 0; i < x1Rank; ++i) {
            const int64_t x1Dim = x1Shape->GetDim(i);
            const int64_t x2Dim = x2Shape->GetDim(i);
            OP_CHECK_IF(x1Dim == 0 || x2Dim == 0,
                        OP_LOGE_FOR_INVALID_SHAPES_WITH_REASON(context->GetNodeName(), "x1, x2", inputShapes,
                                                               "empty input tensors are unsupported"),
                        return ge::GRAPH_FAILED);
            OP_CHECK_IF(x1Dim < ge::UNKNOWN_DIM || x2Dim < ge::UNKNOWN_DIM,
                        OP_LOGE_FOR_INVALID_SHAPES_WITH_REASON(context->GetNodeName(), "x1, x2", inputShapes,
                                                               "only -1 may represent an unknown dimension"),
                        return ge::GRAPH_FAILED);
            OP_CHECK_IF(x1Dim > 0 && x2Dim > 0 && x1Dim != x2Dim,
                        OP_LOGE_FOR_INVALID_SHAPES_WITH_REASON(context->GetNodeName(), "x1, x2", inputShapes,
                                                               "concrete input shapes must be identical"),
                        return ge::GRAPH_FAILED);
        }
    }

    y1Shape->SetDimNum(0);
    y2Shape->SetDimNum(0);
    return ge::GRAPH_SUCCESS;
}

IMPL_OP_INFERSHAPE(SquareSumAll).InferShape(InferShapeForSquareSumAll);
} // namespace ops
