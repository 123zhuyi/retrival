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
 * \file square_sum_v1_infershape.cpp
 * \brief
 */
#include "log/log.h"
#include "register/op_impl_registry.h"
#include "common/inc/op_host/infershape_reduce_util.h"

using namespace ge;
using namespace Ops::Math;
namespace ops {

static ge::graphStatus InferShape4SquareSumV1(gert::InferShapeContext* context)
{
    auto attrs = context->GetAttrs();
    OP_CHECK_NULL_WITH_CONTEXT(context, attrs);

    auto axis = attrs->GetAttrPointer<gert::ContinuousVector>(0);
    OP_CHECK_NULL_WITH_CONTEXT(context, axis);
    auto axesSize = static_cast<int32_t>(axis->GetSize());
    OP_CHECK_IF(axesSize < 0, OP_LOGE(context->GetNodeName(), "axes size must be >= 0!"), return ge::GRAPH_FAILED);
    auto axesData = const_cast<int64_t*>(static_cast<const int64_t*>(axis->GetData()));

    const bool* keepDims = attrs->GetAttrPointer<bool>(1);
    OP_CHECK_NULL_WITH_CONTEXT(context, keepDims);

    const bool* noopWithEmptyAxes = attrs->GetAttrPointer<bool>(2);
    OP_CHECK_NULL_WITH_CONTEXT(context, noopWithEmptyAxes);

    auto inShape = context->GetInputShape(0);
    OP_CHECK_NULL_WITH_CONTEXT(context, inShape);
    auto outShape = context->GetOutputShape(0);
    OP_CHECK_NULL_WITH_CONTEXT(context, outShape);

    OP_LOGI(context->GetNodeName(), "inShape = %s, keepDims = %s, noopWithEmptyAxes = %s, axesSize = %d",
            ToString(*inShape).c_str(), *keepDims ? "true" : "false", *noopWithEmptyAxes ? "true" : "false", axesSize);

    if (IsUnknownRank(*inShape)) {
        OP_LOGI(context->GetNodeName(), "outShape = {-2}");
        SetUnknownRank(*outShape);
        return ge::GRAPH_SUCCESS;
    }

    gert::StorageShape axesShape({axesSize}, {axesSize});
    gert::StorageFormat axesFormat(ge::FORMAT_ND, ge::FORMAT_ND, gert::ExpandDimsType());
    gert::Tensor axesTensor(axesShape, axesFormat, gert::kOnHost, ge::DT_INT64, axesData);

    auto ret = DoInferShapeReduce(context, inShape, outShape, &axesTensor, *keepDims, *noopWithEmptyAxes);
    OP_LOGI(context->GetNodeName(), "outShape = %s", ToString(*outShape).c_str());
    return ret;
}

static ge::graphStatus InferShapeRange4SquareSumV1(gert::InferShapeRangeContext* context)
{
    return InferShapeRange4ReduceCommon(context, "InferShapeRange4SquareSumV1");
}

static ge::graphStatus InferDataType4SquareSumV1(gert::InferDataTypeContext* context)
{
    return InferDataType4ReduceCommon(context, "InferDataType4SquareSumV1");
}

IMPL_OP_INFERSHAPE(SquareSumV1)
    .InferShape(InferShape4SquareSumV1)
    .InferShapeRange(InferShapeRange4SquareSumV1)
    .InferDataType(InferDataType4SquareSumV1);
} // namespace ops
