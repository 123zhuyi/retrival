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
 * \file data_compare_infershape.cpp
 * \brief DataCompare 算子 InferShape
 *
 * All Reduce：输出固定为标量（0 维 float32）。
 * 输入 x1.shape == x2.shape（不支持 broadcast），rank ≤ 8。
 */
#include "register/op_impl_registry.h"
#include "op_common/log/log.h"

using namespace ge;

namespace ops {

namespace {
constexpr size_t kInputX1Idx = 0;
constexpr size_t kInputX2Idx = 1;
constexpr size_t kOutputNumIdx = 0;
constexpr size_t kMaxRank = 8;
} // namespace

static ge::graphStatus InferShape4DataCompare(gert::InferShapeContext* context)
{
    OP_LOGD(context->GetNodeName(), "Begin InferShape4DataCompare");

    const gert::Shape* x1Shape = context->GetInputShape(kInputX1Idx);
    OP_CHECK_NULL_WITH_CONTEXT(context, x1Shape);
    const gert::Shape* x2Shape = context->GetInputShape(kInputX2Idx);
    OP_CHECK_NULL_WITH_CONTEXT(context, x2Shape);
    gert::Shape* numShape = context->GetOutputShape(kOutputNumIdx);
    OP_CHECK_NULL_WITH_CONTEXT(context, numShape);

    // unknown rank 透传（检查 dimNum == 0 且 shape 未定义）
    const int64_t x1Rank = static_cast<int64_t>(x1Shape->GetDimNum());
    const int64_t x2Rank = static_cast<int64_t>(x2Shape->GetDimNum());

    // rank 校验 ≤ 8
    OP_CHECK_IF(
        x1Rank > static_cast<int64_t>(kMaxRank),
        OP_LOGE(context->GetNodeName(), "x1 rank %ld exceeds max %zu", x1Rank, kMaxRank), return GRAPH_FAILED);
    OP_CHECK_IF(
        x2Rank > static_cast<int64_t>(kMaxRank),
        OP_LOGE(context->GetNodeName(), "x2 rank %ld exceeds max %zu", x2Rank, kMaxRank), return GRAPH_FAILED);

    // shape 一致性校验
    OP_CHECK_IF(
        x1Rank != x2Rank, OP_LOGE(context->GetNodeName(), "x1 rank %ld != x2 rank %ld", x1Rank, x2Rank),
        return GRAPH_FAILED);
    for (int64_t i = 0; i < x1Rank; ++i) {
        OP_CHECK_IF(
            x1Shape->GetDim(static_cast<size_t>(i)) != x2Shape->GetDim(static_cast<size_t>(i)),
            OP_LOGE(
                context->GetNodeName(), "x1 dim[%ld]=%ld != x2 dim[%ld]=%ld", i,
                x1Shape->GetDim(static_cast<size_t>(i)), i, x2Shape->GetDim(static_cast<size_t>(i))),
            return GRAPH_FAILED);
    }

    // 输出固定为标量（0 维）
    *numShape = gert::Shape();

    OP_LOGD(context->GetNodeName(), "End InferShape: x1Rank=%ld, output=scalar", x1Rank);
    return GRAPH_SUCCESS;
}

static ge::graphStatus InferDataType4DataCompare(gert::InferDataTypeContext* context)
{
    // 输出固定为 float32
    context->SetOutputDataType(kOutputNumIdx, ge::DT_FLOAT);
    return GRAPH_SUCCESS;
}

IMPL_OP_INFERSHAPE(DataCompare).InferShape(InferShape4DataCompare).InferDataType(InferDataType4DataCompare);

} // namespace ops
