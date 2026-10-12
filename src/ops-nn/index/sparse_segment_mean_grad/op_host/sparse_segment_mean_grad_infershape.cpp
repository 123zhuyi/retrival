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
 * \file sparse_segment_mean_grad_infershape.cpp
 * \brief
 */

#include "register/op_impl_registry.h"
#include "log/log.h"
#include "util/const_util.h"
#include "util/shape_util.h"

using namespace ge;
using namespace Ops::Base;

namespace ops {
namespace {
constexpr size_t kXInputIdx = 0U;
constexpr size_t kIndicesInputIdx = 1U;
constexpr size_t kSegmentIdsInputIdx = 2U;
constexpr size_t kOutputDim0InputIdx = 3U;
constexpr size_t kYOutputIdx = 0U;
constexpr size_t kXMinRank = 1U;
constexpr size_t kIndicesRank = 1U;
constexpr size_t kOutputDim0Rank = 0U;
constexpr int64_t kUnknownDimValue = -1LL;

// Rank check that lets an unknown-rank (-2) shape pass, matching the WithRank / WithRankAtLeast
// helpers used by the source implementation.
bool IsRankInvalid(const gert::Shape& shape, size_t expectRank, bool atLeast)
{
    if (IsUnknownRank(shape)) {
        return false;
    }
    return atLeast ? (shape.GetDimNum() < expectRank) : (shape.GetDimNum() != expectRank);
}

// Merge semantics: two dims conflict only when both of them are known and differ; an unknown
// dim (-1) merges with anything.
bool IsDimConflict(int64_t lhs, int64_t rhs)
{
    return lhs != kUnknownDimValue && rhs != kUnknownDimValue && lhs != rhs;
}
} // namespace

static ge::graphStatus CheckRanksForSparseSegmentMeanGrad(const gert::InferShapeContext* context)
{
    // x is already known to have a known rank here: the caller returns early on unknown rank.
    const gert::Shape* xShape = context->GetInputShape(kXInputIdx);
    OP_CHECK_NULL_WITH_CONTEXT(context, xShape);
    if (IsRankInvalid(*xShape, kXMinRank, true)) {
        const std::string xDim = std::to_string(xShape->GetDimNum()) + "D";
        OP_LOGE_FOR_INVALID_SHAPEDIM(context->GetNodeName(), "x", xDim.c_str(), "at least 1D");
        return GRAPH_FAILED;
    }

    const gert::Shape* indicesShape = context->GetInputShape(kIndicesInputIdx);
    OP_CHECK_NULL_WITH_CONTEXT(context, indicesShape);
    if (IsRankInvalid(*indicesShape, kIndicesRank, false)) {
        const std::string indicesDim = std::to_string(indicesShape->GetDimNum()) + "D";
        OP_LOGE_FOR_INVALID_SHAPEDIM(context->GetNodeName(), "indices", indicesDim.c_str(), "1D");
        return GRAPH_FAILED;
    }

    const gert::Shape* segmentIdsShape = context->GetInputShape(kSegmentIdsInputIdx);
    OP_CHECK_NULL_WITH_CONTEXT(context, segmentIdsShape);
    if (IsRankInvalid(*segmentIdsShape, kIndicesRank, false)) {
        const std::string segmentIdsDim = std::to_string(segmentIdsShape->GetDimNum()) + "D";
        OP_LOGE_FOR_INVALID_SHAPEDIM(context->GetNodeName(), "segment_ids", segmentIdsDim.c_str(), "1D");
        return GRAPH_FAILED;
    }

    const gert::Shape* outputDim0Shape = context->GetInputShape(kOutputDim0InputIdx);
    OP_CHECK_NULL_WITH_CONTEXT(context, outputDim0Shape);
    if (IsRankInvalid(*outputDim0Shape, kOutputDim0Rank, false)) {
        const std::string outputDim0Dim = std::to_string(outputDim0Shape->GetDimNum()) + "D";
        OP_LOGE_FOR_INVALID_SHAPEDIM(context->GetNodeName(), "output_dim0", outputDim0Dim.c_str(), "0D");
        return GRAPH_FAILED;
    }
    return GRAPH_SUCCESS;
}

static ge::graphStatus CheckInputShapesForSparseSegmentMeanGrad(const gert::InferShapeContext* context)
{
    if (CheckRanksForSparseSegmentMeanGrad(context) != GRAPH_SUCCESS) {
        return GRAPH_FAILED;
    }

    // indices and segment_ids must describe the same number of elements. Only compare when both
    // ranks are known, so that dynamic shapes are not rejected here.
    const gert::Shape* indicesShape = context->GetInputShape(kIndicesInputIdx);
    const gert::Shape* segmentIdsShape = context->GetInputShape(kSegmentIdsInputIdx);
    if (IsUnknownRank(*indicesShape) || IsUnknownRank(*segmentIdsShape)) {
        return GRAPH_SUCCESS;
    }
    if (IsDimConflict(indicesShape->GetDim(0), segmentIdsShape->GetDim(0))) {
        OP_LOGE(context->GetNodeName(), "segment_ids dim(0)[%ld] must be equal to indices dim(0)[%ld].",
                segmentIdsShape->GetDim(0), indicesShape->GetDim(0));
        return GRAPH_FAILED;
    }
    return GRAPH_SUCCESS;
}

static ge::graphStatus InferShapeForSparseSegmentMeanGrad(gert::InferShapeContext* context)
{
    OP_LOGD(context->GetNodeName(), "Begin InferShapeForSparseSegmentMeanGrad");

    const gert::Shape* xShape = context->GetInputShape(kXInputIdx);
    OP_CHECK_NULL_WITH_CONTEXT(context, xShape);
    gert::Shape* yShape = context->GetOutputShape(kYOutputIdx);
    OP_CHECK_NULL_WITH_CONTEXT(context, yShape);

    // dynamic -2: nothing can be derived, keep the unknown rank.
    if (IsUnknownRank(*xShape)) {
        OP_LOGD(context->GetNodeName(), "x shape is UnknownRank, set y shape to -2");
        *yShape = *xShape;
        return GRAPH_SUCCESS;
    }

    if (CheckInputShapesForSparseSegmentMeanGrad(context) != GRAPH_SUCCESS) {
        return GRAPH_FAILED;
    }

    // y is x with its first dim replaced by output_dim0; output_dim0 falls back to -1 when it is not a
    // compile-time const.
    int64_t outputDim0 = kUnknownDimValue;
    if (!GetConstInt(context, kOutputDim0InputIdx, outputDim0)) {
        OP_LOGD(context->GetNodeName(), "output_dim0 is not a compile-time const, fall back to -1");
    }
    *yShape = *xShape;
    yShape->SetDim(0, outputDim0);

    OP_LOGD(context->GetNodeName(), "End InferShapeForSparseSegmentMeanGrad");
    return GRAPH_SUCCESS;
}

static ge::graphStatus InferDataTypeForSparseSegmentMeanGrad(gert::InferDataTypeContext* context)
{
    OP_LOGD(context->GetNodeName(), "Begin InferDataTypeForSparseSegmentMeanGrad");
    context->SetOutputDataType(kYOutputIdx, context->GetInputDataType(kXInputIdx));
    OP_LOGD(context->GetNodeName(), "End InferDataTypeForSparseSegmentMeanGrad");
    return GRAPH_SUCCESS;
}

IMPL_OP_INFERSHAPE(SparseSegmentMeanGrad)
    .InputsDataDependency({kOutputDim0InputIdx})
    .InferShape(InferShapeForSparseSegmentMeanGrad)
    .InferDataType(InferDataTypeForSparseSegmentMeanGrad);
} // namespace ops
