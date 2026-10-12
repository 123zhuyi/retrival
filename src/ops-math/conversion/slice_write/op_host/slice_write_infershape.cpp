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
 * \file slice_write_infershape.cpp
 * \brief infershape func of SliceWrite
 */
#include "log/log.h"
#include "register/op_impl_registry.h"
#include "op_api/op_util.h"

using namespace ge;
using namespace Ops::Base;
namespace ops {

static constexpr size_t SLICE_WRITE_IDX_IN_X = 0;
static constexpr size_t SLICE_WRITE_IDX_IN_VALUE = 2;
static constexpr size_t SLICE_WRITE_IDX_OUT_X = 0;

static ge::graphStatus SliceWriteInferShape(gert::InferShapeContext* context)
{
    OP_LOGD(context->GetNodeName(), "Begin to do SliceWriteInferShape");
    const auto* xShape = context->GetInputShape(SLICE_WRITE_IDX_IN_X);
    OP_CHECK_NULL_WITH_CONTEXT(context, xShape);
    auto* outShape = context->GetOutputShape(SLICE_WRITE_IDX_OUT_X);
    OP_CHECK_NULL_WITH_CONTEXT(context, outShape);

    const auto* xTensor = context->GetInputTensor(SLICE_WRITE_IDX_IN_X);
    OP_CHECK_NULL_WITH_CONTEXT(context, xTensor);
    const auto* valueTensor = context->GetInputTensor(SLICE_WRITE_IDX_IN_VALUE);
    OP_CHECK_NULL_WITH_CONTEXT(context, valueTensor);
    OP_CHECK_IF(xTensor->GetDataType() != valueTensor->GetDataType(),
                OP_LOGE(context->GetNodeName(), "dtype of x [%s] must be same as value [%s]",
                        Ops::Base::ToString(xTensor->GetDataType()).c_str(),
                        Ops::Base::ToString(valueTensor->GetDataType()).c_str()),
                return ge::GRAPH_FAILED);

    *outShape = *xShape;
    OP_LOGD(context->GetNodeName(), "End to do SliceWriteInferShape");
    return ge::GRAPH_SUCCESS;
}

static ge::graphStatus SliceWriteInferDataType(gert::InferDataTypeContext* context)
{
    OP_LOGD(context->GetNodeName(), "Begin to do SliceWriteInferDataType");
    auto xDtype = context->GetInputDataType(SLICE_WRITE_IDX_IN_X);
    context->SetOutputDataType(SLICE_WRITE_IDX_OUT_X, xDtype);
    OP_LOGD(context->GetNodeName(), "End to do SliceWriteInferDataType");
    return ge::GRAPH_SUCCESS;
}

static ge::graphStatus SliceWriteInferShapeRange(gert::InferShapeRangeContext* context)
{
    OP_LOGD(context->GetNodeName(), "Begin to do SliceWriteInferShapeRange");
    const auto* xRange = context->GetInputShapeRange(SLICE_WRITE_IDX_IN_X);
    OP_CHECK_NULL_WITH_CONTEXT(context, xRange);
    const auto* xRangeMax = xRange->GetMax();
    const auto* xRangeMin = xRange->GetMin();
    OP_CHECK_NULL_WITH_CONTEXT(context, xRangeMax);
    OP_CHECK_NULL_WITH_CONTEXT(context, xRangeMin);

    auto* outRange = context->GetOutputShapeRange(SLICE_WRITE_IDX_OUT_X);
    OP_CHECK_NULL_WITH_CONTEXT(context, outRange);
    auto* outRangeMax = outRange->GetMax();
    auto* outRangeMin = outRange->GetMin();
    OP_CHECK_NULL_WITH_CONTEXT(context, outRangeMax);
    OP_CHECK_NULL_WITH_CONTEXT(context, outRangeMin);
    *outRangeMax = *xRangeMax;
    *outRangeMin = *xRangeMin;
    OP_LOGD(context->GetNodeName(), "End to do SliceWriteInferShapeRange");
    return ge::GRAPH_SUCCESS;
}

IMPL_OP_INFERSHAPE(SliceWrite)
    .InferShape(SliceWriteInferShape)
    .InferDataType(SliceWriteInferDataType)
    .InferShapeRange(SliceWriteInferShapeRange);
} // namespace ops
