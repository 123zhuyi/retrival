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
 * \file diag_part_infershape.cpp
 * \brief
 */
#include "register/op_impl_registry.h"
#include "log/log.h"

using namespace ge;

namespace ops {
static constexpr int64_t IDX_0 = 0;
static constexpr int64_t IDX_1 = 1;

static ge::graphStatus InferShapeDiagPart(gert::InferShapeContext* context)
{
    OP_CHECK_IF(context == nullptr, OP_LOGE(context, "context is nullptr"), return ge::GRAPH_FAILED);
    OP_LOGD(context->GetNodeName(), "Begin to do InferShapeDiagPart");

    const gert::Shape* xShape = context->GetInputShape(IDX_0);
    OP_CHECK_NULL_WITH_CONTEXT(context, xShape);

    gert::Shape* yShape = context->GetOutputShape(IDX_0);
    OP_CHECK_NULL_WITH_CONTEXT(context, yShape);

    auto xDimNum = xShape->GetDimNum();
    if (xDimNum % 2 != 0) {
        OP_LOGE(context->GetNodeName(), "Input x dimNum must be even, but got %lu", xDimNum);
        return ge::GRAPH_FAILED;
    }

    int64_t outputDimNum = xDimNum / 2;
    yShape->SetDimNum(outputDimNum);
    for (int64_t i = 0; i < outputDimNum; i++) {
        yShape->SetDim(i, xShape->GetDim(i));
    }

    OP_LOGD(context->GetNodeName(), "End to do InferShapeDiagPart");
    return GRAPH_SUCCESS;
}

IMPL_OP_INFERSHAPE(DiagPart).InferShape(InferShapeDiagPart);
} // namespace ops