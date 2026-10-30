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
 * \file trace_infershape.cpp
 * \brief InferShape implementation for trace operator
 *
 * Output shape: scalar (0-dim tensor)
 * Output dtype: float types -> same as input; integer/bool -> int64
 * Input must be 2D matrix
 */
#include "register/op_impl_registry.h"
#include "log/log.h"

using namespace ge;

namespace ops {
static constexpr int64_t IDX_0 = 0;

static ge::graphStatus InferShapeTrace(gert::InferShapeContext* context)
{
    OP_LOGD(context->GetNodeName(), "Begin to do InferShapeTrace");

    const gert::Shape* xShape = context->GetInputShape(IDX_0);
    OP_CHECK_NULL_WITH_CONTEXT(context, xShape);

    // Dimension check: must be 2D
    if (xShape->GetDimNum() != 2) {
        OP_LOGE_FOR_INVALID_SHAPEDIM_WITH_REASON(
            context->GetNodeName(), "x",
            std::to_string(xShape->GetDimNum()).c_str(),
            "trace expects a 2D matrix");
        return GRAPH_FAILED;
    }

    // Output shape is scalar (0 dimensions)
    gert::Shape* yShape = context->GetOutputShape(IDX_0);
    OP_CHECK_NULL_WITH_CONTEXT(context, yShape);
    yShape->SetDimNum(0);

    OP_LOGD(context->GetNodeName(), "End to do InferShapeTrace");
    return GRAPH_SUCCESS;
}

IMPL_OP_INFERSHAPE(Trace).InferShape(InferShapeTrace);
}  // namespace ops
