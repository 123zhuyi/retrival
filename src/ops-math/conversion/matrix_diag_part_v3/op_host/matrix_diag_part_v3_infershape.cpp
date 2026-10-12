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
 * \file matrix_diag_part_v3_infershape.cpp
 * \brief InferShape for MatrixDiagPartV3. The output shape depends on the const value of k: the batch dims of x are
 *        kept, a num_diags dim is inserted when k selects a band of diagonals, and the last dim is the longest
 *        diagonal length derived from the last two dims of x. When k is not a compile-time constant, or x has an
 *        unknown rank, or k has a non fully defined shape, the output falls back to a 1-D unknown shape.
 */

#include <algorithm>
#include <cstdint>
#include <string>

#include "common/inc/op_host/matrix_diag_infershape_common.h"
#include "register/op_impl_registry.h"
#include "util/shape_util.h"
#include "log/log.h"

using namespace ge;

namespace ops {
namespace {
using namespace ops::matrix_diag;

constexpr size_t kInputXIdx = 0U;
constexpr size_t kInputKIdx = 1U;
constexpr size_t kInputPaddingValueIdx = 2U;
constexpr size_t kOutputYIdx = 0U;

constexpr size_t kXMinRank = 2U;

ge::graphStatus CheckInputRanks(gert::InferShapeContext* context, const gert::Shape& xShape, const gert::Shape& kShape)
{
    OP_CHECK_IF(IsRankInvalid(xShape, kXMinRank, true),
                OP_LOGE_FOR_INVALID_SHAPEDIM_WITH_REASON(context->GetNodeName(), "x",
                                                         std::to_string(xShape.GetDimNum()).c_str(),
                                                         "The rank of x must be at least 2."),
                return ge::GRAPH_FAILED);
    OP_CHECK_IF(IsRankAboveLimit(kShape, kKMaxRank),
                OP_LOGE_FOR_INVALID_SHAPEDIM_WITH_REASON(context->GetNodeName(), "k",
                                                         std::to_string(kShape.GetDimNum()).c_str(),
                                                         "The rank of k must be at most 1."),
                return ge::GRAPH_FAILED);
    return CheckScalarInput(context, kInputPaddingValueIdx, "padding_value");
}

// The longest diagonal length can only be computed when both matrix dims are known; otherwise it stays unknown.
ge::graphStatus GetMaxDiagLen(gert::InferShapeContext* context, const gert::Shape& xShape, int64_t lowerDiagIndex,
                              int64_t upperDiagIndex, int64_t& maxDiagLen)
{
    const size_t xRank = xShape.GetDimNum();
    const int64_t numRows = xShape.GetDim(xRank - kXMinRank);
    const int64_t numCols = xShape.GetDim(xRank - 1U);

    maxDiagLen = kUnknownDim;
    if (numRows == kUnknownDim || numCols == kUnknownDim) {
        return ge::GRAPH_SUCCESS;
    }

    OP_CHECK_IF(lowerDiagIndex != kZero && (-numRows >= lowerDiagIndex || lowerDiagIndex >= numCols),
                OP_LOGE_FOR_INVALID_VALUE_WITH_REASON(
                    context->GetNodeName(), "k", std::to_string(lowerDiagIndex).c_str(),
                    "The lower diagonal index must be 0 or fall in the open range (-num_rows, num_cols)."),
                return ge::GRAPH_FAILED);
    OP_CHECK_IF(upperDiagIndex != kZero && (-numRows >= upperDiagIndex || upperDiagIndex >= numCols),
                OP_LOGE_FOR_INVALID_VALUE_WITH_REASON(
                    context->GetNodeName(), "k", std::to_string(upperDiagIndex).c_str(),
                    "The upper diagonal index must be 0 or fall in the open range (-num_rows, num_cols)."),
                return ge::GRAPH_FAILED);

    maxDiagLen = std::min(numRows + std::min(upperDiagIndex, kZero), numCols - std::max(lowerDiagIndex, kZero));
    return ge::GRAPH_SUCCESS;
}

ge::graphStatus InferShape4MatrixDiagPartV3(gert::InferShapeContext* context)
{
    OP_LOGD(context->GetNodeName(), "Begin InferShape4MatrixDiagPartV3");

    const gert::Shape* xShape = context->GetInputShape(kInputXIdx);
    OP_CHECK_NULL_WITH_CONTEXT(context, xShape);
    const gert::Shape* kShape = context->GetInputShape(kInputKIdx);
    OP_CHECK_NULL_WITH_CONTEXT(context, kShape);
    gert::Shape* yShape = context->GetOutputShape(kOutputYIdx);
    OP_CHECK_NULL_WITH_CONTEXT(context, yShape);

    if (CheckInputRanks(context, *xShape, *kShape) != ge::GRAPH_SUCCESS) {
        return ge::GRAPH_FAILED;
    }

    // The diagonal band is selected by the const value of k. Without it, or with an unknown x rank, or with a k whose
    // own shape is not fully defined, no concrete output shape can be derived.
    const int32_t* kData = TryGetDiagIndexData(context, kInputKIdx);
    if (kData == nullptr || Ops::Base::IsUnknownRank(*xShape) || !IsShapeFullyDefined(*kShape)) {
        OP_LOGD(context->GetNodeName(), "k is not a compile-time constant or x/k shape is not fully defined, "
                                        "fall back to a 1-D unknown output shape");
        Ops::Base::SetUnknownShape(static_cast<int64_t>(kFallbackRank), *yShape);
        return ge::GRAPH_SUCCESS;
    }

    int64_t lowerDiagIndex = 0;
    int64_t upperDiagIndex = 0;
    if (GetDiagIndex(context, *kShape, kData, lowerDiagIndex, upperDiagIndex) != ge::GRAPH_SUCCESS) {
        return ge::GRAPH_FAILED;
    }

    int64_t maxDiagLen = kUnknownDim;
    if (GetMaxDiagLen(context, *xShape, lowerDiagIndex, upperDiagIndex, maxDiagLen) != ge::GRAPH_SUCCESS) {
        return ge::GRAPH_FAILED;
    }

    // Output layout: batch dims of x, then num_diags when k selects a band, then the longest diagonal length.
    const size_t xRank = xShape->GetDimNum();
    const bool isDiagBand = (lowerDiagIndex < upperDiagIndex);
    const size_t outRank = xRank - 1U + (isDiagBand ? 1U : 0U);
    yShape->SetDimNum(outRank);

    size_t outIdx = 0U;
    for (size_t i = 0U; i + kXMinRank < xRank; i++) {
        yShape->SetDim(outIdx, xShape->GetDim(i));
        outIdx++;
    }
    if (isDiagBand) {
        yShape->SetDim(outIdx, upperDiagIndex - lowerDiagIndex + 1);
        outIdx++;
    }
    yShape->SetDim(outIdx, maxDiagLen);

    OP_LOGD(context->GetNodeName(), "End InferShape4MatrixDiagPartV3");
    return ge::GRAPH_SUCCESS;
}
} // namespace

IMPL_OP_INFERSHAPE(MatrixDiagPartV3).InferShape(InferShape4MatrixDiagPartV3).InputsDataDependency({kInputKIdx});
} // namespace ops
