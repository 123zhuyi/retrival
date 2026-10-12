/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include "log/log.h"
#include "register/op_impl_registry.h"

using namespace ge;

namespace ops {
constexpr int64_t INPUT_X_INDEX = 0;
constexpr int64_t OUTPUT_Y_INDEX = 0;

static ge::graphStatus InferShapeCumulativeLogsumexp(gert::InferShapeContext* context)
{
    const gert::Shape* xShape = context->GetInputShape(INPUT_X_INDEX);
    OP_CHECK_NULL_WITH_CONTEXT(context, xShape);
    gert::Shape* yShape = context->GetOutputShape(OUTPUT_Y_INDEX);
    OP_CHECK_NULL_WITH_CONTEXT(context, yShape);

    yShape->SetDimNum(xShape->GetDimNum());
    for (size_t i = 0; i < xShape->GetDimNum(); ++i) {
        yShape->SetDim(i, xShape->GetDim(i));
    }
    return GRAPH_SUCCESS;
}

static ge::graphStatus InferDataTypeCumulativeLogsumexp(gert::InferDataTypeContext* context)
{
    context->SetOutputDataType(OUTPUT_Y_INDEX, context->GetInputDataType(INPUT_X_INDEX));
    return GRAPH_SUCCESS;
}

IMPL_OP_INFERSHAPE(CumulativeLogsumexp)
    .InferShape(InferShapeCumulativeLogsumexp)
    .InferDataType(InferDataTypeCumulativeLogsumexp);
} // namespace ops
