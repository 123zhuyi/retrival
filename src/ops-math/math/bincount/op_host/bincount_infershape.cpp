/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include "util/const_util.h"
#include "log/log.h"
#include "register/op_impl_registry.h"
#include "util/shape_util.h"
using namespace ge;
namespace ops {
static constexpr int64_t UNKNOWN_DIM_VALUE_ = -1LL;
static constexpr size_t kInputIndex0 = 0U;
static constexpr size_t kInputIndex1 = 1U;
static constexpr size_t kOutputIndex0 = 0U;
static constexpr size_t kInputIndex2 = 2U;
static constexpr int64_t kRank0 = 0U;
static constexpr int64_t kRank1 = 1U;

static ge::graphStatus SetAllUnknownDim(const int64_t rank, gert::Shape* output_shape)
{
    OP_CHECK_IF(
        output_shape == nullptr, OP_LOGD("SetAllUnknownDim", "the output_shape is nullptr, return unsuccess"),
        return ge::GRAPH_FAILED);

    output_shape->SetDimNum(rank);
    for (int64_t i = 0; i < rank; ++i) {
        output_shape->SetDim(i, UNKNOWN_DIM_VALUE_);
    }
    OP_LOGD("SetAllUnknownDim", "set all dim = -1, output = %s", Ops::Base::ToString(*output_shape).c_str());

    return ge::GRAPH_SUCCESS;
}

static ge::graphStatus WithRank(
    const gert::Shape* tensor, int64_t rank, gert::Shape* out_shape, const std::string opName)
{
    if (rank > INT32_MAX) {
        OP_LOGE(opName, "rank[%ld] cannot exceed kint32max", rank);
        return ge::GRAPH_FAILED;
    }
    int64_t existing = static_cast<int64_t>(tensor->GetDimNum());
    if (Ops::Base::IsUnknownRank(*tensor)) {
        SetAllUnknownDim(rank, out_shape);
        return ge::GRAPH_SUCCESS;
    }
    if (existing != rank) {
        OP_LOGE(opName, "rank[%ld] must be [%ld]", existing, rank);
        return ge::GRAPH_FAILED;
    }
    *out_shape = *tensor;
    return ge::GRAPH_SUCCESS;
}

static ge::graphStatus InferShapeForBincount(gert::InferShapeContext* context)
{
    OP_LOGI(context->GetNodeName(), "Begin to do InferShapeForBincount");
    gert::Shape output_unused;

    const gert::Shape* array_shape = context->GetInputShape(kInputIndex0);
    OP_CHECK_NULL_WITH_CONTEXT(context, array_shape);
    const gert::Shape* size_shape = context->GetInputShape(kInputIndex1);
    OP_CHECK_NULL_WITH_CONTEXT(context, size_shape);
    const gert::Tensor* size_tensor = context->GetInputTensor(kInputIndex1);
    OP_CHECK_NULL_WITH_CONTEXT(context, size_tensor);

    OP_CHECK_IF(
        WithRank(size_shape, kRank0, &output_unused, context->GetNodeName()) != GRAPH_SUCCESS,
        OP_LOGE(context->GetNodeName(), "call WithRankAtLeast failed."), return ge::GRAPH_FAILED);
    gert::Shape* bins_shape = context->GetOutputShape(kOutputIndex0);
    OP_CHECK_NULL_WITH_CONTEXT(context, bins_shape);
    int64_t bins = 0;
    gert::Shape const_shape;
    if (!Ops::Base::GetConstIntToShape<gert::InferShapeContext>(context, kInputIndex1, const_shape)) {
        bins = UNKNOWN_DIM;
    }
    if (bins != UNKNOWN_DIM) {
        const int32_t* size_data = size_tensor->GetData<int32_t>();
        bins = size_data[kInputIndex0];
    }
    bins_shape->SetDimNum(kRank1);
    bins_shape->SetDim(kOutputIndex0, bins);
    return ge::GRAPH_SUCCESS;
}

IMPL_OP_INFERSHAPE(Bincount).InputsDataDependency({1}).InferShape(InferShapeForBincount);

} // namespace ops
