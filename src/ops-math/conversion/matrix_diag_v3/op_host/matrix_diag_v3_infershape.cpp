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
 * \file matrix_diag_v3_infershape.cpp
 * \brief InferShape for MatrixDiagV3. x holds the diagonal values and the output is the matrix built from them, so
 *        the output replaces the trailing dims of x with [num_rows, num_cols]. num_rows and num_cols come from the
 *        const values of the inputs with the same names; when either is absent they are derived from the longest
 *        diagonal length and the diagonal band selected by k. When k is not a compile-time constant, or x has an
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
constexpr size_t kInputNumRowsIdx = 2U;
constexpr size_t kInputNumColsIdx = 3U;
constexpr size_t kInputPaddingValueIdx = 4U;
constexpr size_t kOutputYIdx = 0U;

constexpr size_t kXMinRank = 1U;

ge::graphStatus CheckInputRanks(gert::InferShapeContext* context, const gert::Shape& xShape, const gert::Shape& kShape)
{
    OP_CHECK_IF(IsRankInvalid(xShape, kXMinRank, true),
                OP_LOGE_FOR_INVALID_SHAPEDIM_WITH_REASON(context->GetNodeName(), "x",
                                                         std::to_string(xShape.GetDimNum()).c_str(),
                                                         "The rank of x must be at least 1."),
                return ge::GRAPH_FAILED);
    OP_CHECK_IF(IsRankAboveLimit(kShape, kKMaxRank),
                OP_LOGE_FOR_INVALID_SHAPEDIM_WITH_REASON(context->GetNodeName(), "k",
                                                         std::to_string(kShape.GetDimNum()).c_str(),
                                                         "The rank of k must be at most 1."),
                return ge::GRAPH_FAILED);
    if (CheckScalarInput(context, kInputNumRowsIdx, "num_rows") != ge::GRAPH_SUCCESS) {
        return ge::GRAPH_FAILED;
    }
    if (CheckScalarInput(context, kInputNumColsIdx, "num_cols") != ge::GRAPH_SUCCESS) {
        return ge::GRAPH_FAILED;
    }
    return CheckScalarInput(context, kInputPaddingValueIdx, "padding_value");
}

// Reads a scalar int32 const input; leaves value untouched when the input is not a compile-time constant.
void TryGetConstScalar(gert::InferShapeContext* context, size_t inputIdx, int64_t& value)
{
    const int32_t* data = TryGetDiagIndexData(context, inputIdx);
    if (data == nullptr) {
        return;
    }
    value = static_cast<int64_t>(data[0]);
}

ge::graphStatus GetRowsAndCols(gert::InferShapeContext* context, const gert::Shape& xShape, int64_t lowerDiagIndex,
                               int64_t upperDiagIndex, int64_t& numRows, int64_t& numCols)
{
    const size_t xRank = xShape.GetDimNum();
    if (lowerDiagIndex < upperDiagIndex) {
        OP_CHECK_IF(xRank < kDiagBandMinRank,
                    OP_LOGE_FOR_INVALID_SHAPEDIM_WITH_REASON(context->GetNodeName(), "x", std::to_string(xRank).c_str(),
                                                             "The rank of x must be at least 2 when k selects a band "
                                                             "of diagonals."),
                    return ge::GRAPH_FAILED);
        const int64_t numDiags = xShape.GetDim(xRank - kDiagBandMinRank);
        OP_CHECK_IF(numDiags != upperDiagIndex - lowerDiagIndex + 1,
                    OP_LOGE_FOR_INVALID_SHAPEDIM_WITH_REASON(
                        context->GetNodeName(), "x", std::to_string(numDiags).c_str(),
                        "The number of diagonals held by x does not match the band implied by k."),
                    return ge::GRAPH_FAILED);
    }

    numRows = kUnknownDim;
    numCols = kUnknownDim;
    TryGetConstScalar(context, kInputNumRowsIdx, numRows);
    TryGetConstScalar(context, kInputNumColsIdx, numCols);

    const int64_t maxDiagLen = xShape.GetDim(xRank - 1U);
    const int64_t minNumRows = maxDiagLen - std::min(upperDiagIndex, kZero);
    const int64_t minNumCols = maxDiagLen + std::max(lowerDiagIndex, kZero);

    if (numRows == kUnknownDim && numCols == kUnknownDim) {
        numRows = std::max(minNumRows, minNumCols);
        numCols = numRows;
    }

    if (numRows == kUnknownDim) {
        numRows = minNumRows;
    } else {
        OP_CHECK_IF(numRows < minNumRows,
                    OP_LOGE_FOR_INVALID_VALUE_WITH_REASON(
                        context->GetNodeName(), "num_rows", std::to_string(numRows).c_str(),
                        ("The value of num_rows must not be less than " + std::to_string(minNumRows) + ".").c_str()),
                    return ge::GRAPH_FAILED);
    }

    if (numCols == kUnknownDim) {
        numCols = minNumCols;
    } else {
        OP_CHECK_IF(numCols < minNumCols,
                    OP_LOGE_FOR_INVALID_VALUE_WITH_REASON(
                        context->GetNodeName(), "num_cols", std::to_string(numCols).c_str(),
                        ("The value of num_cols must not be less than " + std::to_string(minNumCols) + ".").c_str()),
                    return ge::GRAPH_FAILED);
    }

    OP_CHECK_IF(
        numRows != minNumRows && numCols != minNumCols && minNumRows != kUnknownDim && minNumCols != kUnknownDim,
        OP_LOGE_FOR_INVALID_VALUE_WITH_REASON(
            context->GetNodeName(), "num_rows", (std::to_string(numRows) + ", " + std::to_string(numCols)).c_str(),
            ("num_rows and num_cols do not match the minimum sizes implied by x and k, which are " +
             std::to_string(minNumRows) + " and " + std::to_string(minNumCols) + ".")
                .c_str()),
        return ge::GRAPH_FAILED);
    return ge::GRAPH_SUCCESS;
}

ge::graphStatus InferShape4MatrixDiagV3(gert::InferShapeContext* context)
{
    OP_LOGD(context->GetNodeName(), "Begin InferShape4MatrixDiagV3");

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

    int64_t numRows = kUnknownDim;
    int64_t numCols = kUnknownDim;
    if (GetRowsAndCols(context, *xShape, lowerDiagIndex, upperDiagIndex, numRows, numCols) != ge::GRAPH_SUCCESS) {
        return ge::GRAPH_FAILED;
    }
    OP_LOGD(context->GetNodeName(), "num_rows: %ld, num_cols: %ld", numRows, numCols);

    // A single diagonal grows the rank by one (the diagonal dim becomes a [num_rows, num_cols] matrix); a band of
    // diagonals keeps the rank (the num_diags dim is consumed by num_rows).
    const size_t xRank = xShape->GetDimNum();
    if (lowerDiagIndex == upperDiagIndex) {
        yShape->SetDimNum(xRank + 1U);
        for (size_t i = 0U; i + 1U < xRank; i++) {
            yShape->SetDim(i, xShape->GetDim(i));
        }
        yShape->SetDim(xRank - 1U, numRows);
        yShape->SetDim(xRank, numCols);
    } else {
        yShape->SetDimNum(xRank);
        for (size_t i = 0U; i + kDiagBandMinRank < xRank; i++) {
            yShape->SetDim(i, xShape->GetDim(i));
        }
        yShape->SetDim(xRank - kDiagBandMinRank, numRows);
        yShape->SetDim(xRank - 1U, numCols);
    }

    OP_LOGD(context->GetNodeName(), "End InferShape4MatrixDiagV3");
    return ge::GRAPH_SUCCESS;
}
} // namespace

IMPL_OP_INFERSHAPE(MatrixDiagV3)
    .InferShape(InferShape4MatrixDiagV3)
    .InputsDataDependency({kInputKIdx, kInputNumRowsIdx, kInputNumColsIdx});
} // namespace ops
