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
 * \file angle_v2_infer.cpp
 * \brief
 */
#include "register/op_impl_registry.h"
#include "log/log.h"

using namespace ge;
namespace ops {
static constexpr int64_t X_IDX = 0;
static constexpr int64_t Y_IDX = 0;

static ge::graphStatus AngleV2InferShapeFunc(gert::InferShapeContext* context)
{
    OP_LOGD(context, "Begin to do AngleV2InferShapeFunc");
    // get input shapes
    const gert::Shape* xShape = context->GetInputShape(X_IDX);
    OP_CHECK_NULL_WITH_CONTEXT(context, xShape);

    // get output shapes
    gert::Shape* yShape = context->GetOutputShape(Y_IDX);
    OP_CHECK_NULL_WITH_CONTEXT(context, yShape);

    *yShape = *xShape;
    OP_LOGD(context, "End to do AngleV2InferShapeFunc");
    return ge::GRAPH_SUCCESS;
}

IMPL_OP_INFERSHAPE(AngleV2).InferShape(AngleV2InferShapeFunc);
} // namespace ops
