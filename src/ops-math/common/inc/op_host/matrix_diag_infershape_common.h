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
 * \file matrix_diag_infershape_common.h
 * \brief Shared InferShape helpers for the MatrixDiag operator family (MatrixDiagPartV3 / MatrixDiagV3).
 *
 * The rank helpers reproduce the tolerance semantics of the source side WithRank / WithRankAtLeast /
 * WithRankAtMost / FullyDefined helpers: an unknown rank input is always accepted, so that a legal dynamic
 * graph is not rejected. Keep them here rather than duplicating per operator, otherwise a later fix to one
 * copy silently drifts from the other.
 */

#ifndef MATH_COMMON_OP_HOST_MATRIX_DIAG_INFERSHAPE_COMMON_H
#define MATH_COMMON_OP_HOST_MATRIX_DIAG_INFERSHAPE_COMMON_H

#include <cstdint>
#include <string>

#include "log/log.h"
#include "register/op_impl_registry.h"
#include "util/shape_util.h"

namespace ops {
namespace matrix_diag {
constexpr size_t kKMaxRank = 1U;
constexpr size_t kScalarRank = 0U;
constexpr size_t kFallbackRank = 1U;
constexpr size_t kDiagBandMinRank = 2U;

constexpr int64_t kUnknownDim = -1;
constexpr int64_t kDiagIndexNumOne = 1;
constexpr int64_t kDiagIndexNumTwo = 2;
constexpr int64_t kZero = 0;

// Mirrors WithRank / WithRankAtLeast: an unknown rank input is always accepted, otherwise the rank is checked.
inline bool IsRankInvalid(const gert::Shape& shape, size_t expectRank, bool atLeast)
{
    if (Ops::Base::IsUnknownRank(shape)) {
        return false;
    }
    return atLeast ? (shape.GetDimNum() < expectRank) : (shape.GetDimNum() != expectRank);
}

// Mirrors WithRankAtMost: an unknown rank input is always accepted, otherwise the rank must not exceed maxRank.
inline bool IsRankAboveLimit(const gert::Shape& shape, size_t maxRank)
{
    if (Ops::Base::IsUnknownRank(shape)) {
        return false;
    }
    return shape.GetDimNum() > maxRank;
}

// Mirrors FullyDefined: the rank is known and every dim is known.
inline bool IsShapeFullyDefined(const gert::Shape& shape)
{
    if (Ops::Base::IsUnknownRank(shape)) {
        return false;
    }
    for (size_t i = 0U; i < shape.GetDimNum(); i++) {
        if (shape.GetDim(i) < 0) {
            return false;
        }
    }
    return true;
}

// Checks that an input is a scalar, tolerating unknown rank.
inline ge::graphStatus CheckScalarInput(gert::InferShapeContext* context, size_t inputIdx, const char* paramName)
{
    const gert::Shape* shape = context->GetInputShape(inputIdx);
    OP_CHECK_NULL_WITH_CONTEXT(context, shape);
    OP_CHECK_IF(IsRankInvalid(*shape, kScalarRank, false),
                OP_LOGE_FOR_INVALID_SHAPEDIM_WITH_REASON(context->GetNodeName(), paramName,
                                                         std::to_string(shape->GetDimNum()).c_str(),
                                                         "The rank of this input must be 0."),
                return ge::GRAPH_FAILED);
    return ge::GRAPH_SUCCESS;
}

// k is either a scalar, or a vector holding one diagonal index (single diagonal) or two indices (a band).
inline ge::graphStatus GetDiagIndex(gert::InferShapeContext* context, const gert::Shape& kShape, const int32_t* kData,
                                    int64_t& lowerDiagIndex, int64_t& upperDiagIndex)
{
    if (kShape.GetDimNum() == 0U) {
        lowerDiagIndex = static_cast<int64_t>(kData[0]);
        upperDiagIndex = lowerDiagIndex;
    } else {
        const int64_t numElements = kShape.GetDim(0);
        if (numElements == kDiagIndexNumOne) {
            lowerDiagIndex = static_cast<int64_t>(kData[0]);
            upperDiagIndex = lowerDiagIndex;
        } else if (numElements == kDiagIndexNumTwo) {
            lowerDiagIndex = static_cast<int64_t>(kData[0]);
            upperDiagIndex = static_cast<int64_t>(kData[1]);
        } else {
            OP_LOGE_FOR_INVALID_SHAPESIZE_WITH_REASON(
                context->GetNodeName(), "k", std::to_string(numElements).c_str(),
                "The input k must be a scalar or a vector with one or two elements.");
            return ge::GRAPH_FAILED;
        }
    }

    OP_CHECK_IF(
        lowerDiagIndex > upperDiagIndex,
        OP_LOGE_FOR_INVALID_VALUE_WITH_REASON(context->GetNodeName(), "k", std::to_string(lowerDiagIndex).c_str(),
                                              "The lower diagonal index of k must not be greater than the "
                                              "upper diagonal index."),
        return ge::GRAPH_FAILED);
    return ge::GRAPH_SUCCESS;
}

// Reads the const value of k when it is available at compile time; returns nullptr otherwise.
inline const int32_t* TryGetDiagIndexData(gert::InferShapeContext* context, size_t inputIdx)
{
    const gert::Tensor* kTensor = context->GetInputTensor(inputIdx);
    return (kTensor == nullptr) ? nullptr : kTensor->GetData<int32_t>();
}
} // namespace matrix_diag
} // namespace ops

#endif // MATH_COMMON_OP_HOST_MATRIX_DIAG_INFERSHAPE_COMMON_H
