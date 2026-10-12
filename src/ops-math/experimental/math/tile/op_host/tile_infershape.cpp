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
 * \file tile_infershape.cpp
 * \brief
 */
#include <vector>

#include "log/log.h"
#include "register/op_impl_registry.h"

using namespace ge;

namespace ops {
static constexpr size_t IDX_X = 0;
static constexpr size_t IDX_MULTIPLES = 1;
static constexpr size_t IDX_Y = 0;

static ge::graphStatus SetUnknownShape(gert::InferShapeContext* context, gert::Shape* yShape, size_t outputDims)
{
    OP_CHECK_NULL_WITH_CONTEXT(context, yShape);
    yShape->SetDimNum(outputDims);
    for (size_t idx = 0; idx < outputDims; ++idx) {
        yShape->SetDim(idx, -1);
    }
    return GRAPH_SUCCESS;
}

template <typename T>
static ge::graphStatus GetMultiples(
    gert::InferShapeContext* context, const gert::Tensor* multiplesTensor, size_t multiplesSize,
    std::vector<int64_t>& multiples)
{
    const T* multiplesData = multiplesTensor->GetData<T>();
    OP_CHECK_NULL_WITH_CONTEXT(context, multiplesData);
    multiples.reserve(multiplesSize);
    for (size_t idx = 0; idx < multiplesSize; ++idx) {
        multiples.push_back(static_cast<int64_t>(multiplesData[idx]));
    }
    return GRAPH_SUCCESS;
}

static ge::graphStatus InferShapeTile(gert::InferShapeContext* context)
{
    OP_CHECK_IF(context == nullptr, OP_LOGE(context, "context is nullptr"), return ge::GRAPH_FAILED);

    const gert::Shape* xShape = context->GetInputShape(IDX_X);
    OP_CHECK_NULL_WITH_CONTEXT(context, xShape);
    gert::Shape* yShape = context->GetOutputShape(IDX_Y);
    OP_CHECK_NULL_WITH_CONTEXT(context, yShape);

    const gert::Shape* multiplesShape = context->GetInputShape(IDX_MULTIPLES);
    OP_CHECK_NULL_WITH_CONTEXT(context, multiplesShape);
    const gert::Tensor* multiplesTensor = context->GetInputTensor(IDX_MULTIPLES);

    size_t xDims = xShape->GetDimNum();
    size_t multiplesSize = 0;
    if (multiplesShape->GetDimNum() > 0) {
        multiplesSize = static_cast<size_t>(multiplesShape->GetDim(0));
    }
    if (multiplesSize == 0) {
        *yShape = *xShape;
        return GRAPH_SUCCESS;
    }
    size_t outputDims = (xDims > multiplesSize) ? xDims : multiplesSize;
    if (multiplesTensor == nullptr) {
        return SetUnknownShape(context, yShape, outputDims);
    }

    ge::DataType multiplesDtype = multiplesTensor->GetDataType();
    bool hasData = false;
    if (multiplesDtype == ge::DT_INT32) {
        hasData = (multiplesTensor->GetData<int32_t>() != nullptr);
    } else if (multiplesDtype == ge::DT_INT64) {
        hasData = (multiplesTensor->GetData<int64_t>() != nullptr);
    } else {
        OP_LOGE(context, "multiples dtype must be int32 or int64");
        return GRAPH_FAILED;
    }
    if (!hasData) {
        return SetUnknownShape(context, yShape, outputDims);
    }

    std::vector<int64_t> multiples;
    if (multiplesDtype == ge::DT_INT32) {
        OP_CHECK_IF(
            GetMultiples<int32_t>(context, multiplesTensor, multiplesSize, multiples) != GRAPH_SUCCESS,
            OP_LOGE(context, "get int32 multiples failed"), return GRAPH_FAILED);
    } else {
        OP_CHECK_IF(
            GetMultiples<int64_t>(context, multiplesTensor, multiplesSize, multiples) != GRAPH_SUCCESS,
            OP_LOGE(context, "get int64 multiples failed"), return GRAPH_FAILED);
    }

    size_t xOffset = outputDims - xDims;
    size_t multiplesOffset = outputDims - multiples.size();
    yShape->SetDimNum(outputDims);
    for (size_t idx = 0; idx < outputDims; ++idx) {
        int64_t xDim = (idx < xOffset) ? 1 : xShape->GetDim(idx - xOffset);
        int64_t repeat = (idx < multiplesOffset) ? 1 : multiples[idx - multiplesOffset];
        OP_CHECK_IF(
            repeat < 0, OP_LOGE(context, "multiples[%zu] must be >= 0, but got %ld", idx - multiplesOffset, repeat),
            return GRAPH_FAILED);
        yShape->SetDim(idx, xDim * repeat);
    }

    return GRAPH_SUCCESS;
}

IMPL_OP_INFERSHAPE(Tile).InferShape(InferShapeTile).InputsDataDependency({IDX_MULTIPLES});
} // namespace ops
