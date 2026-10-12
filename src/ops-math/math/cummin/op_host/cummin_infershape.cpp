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
 * \file cummin_infershape.cpp
 * \brief Cummin InferShape and InferDataType.
 *        y 与 argmin 的 shape 均与输入 x 一致；
 *        y.dtype 与 x 一致，argmin.dtype 固定为 int32。
 */
#include "register/op_impl_registry.h"
#include "log/log.h"

using namespace ge;

namespace ops {
constexpr size_t INPUT_X_INDEX = 0;
constexpr size_t OUTPUT_Y_INDEX = 0;
constexpr size_t OUTPUT_ARGMIN_INDEX = 1;

static ge::graphStatus InferShapeForCummin(gert::InferShapeContext* context)
{
    OP_LOGD(context->GetNodeName(), "Begin to do InferShapeForCummin");
    const auto* xShape = context->GetInputShape(INPUT_X_INDEX);
    OP_CHECK_NULL_WITH_CONTEXT(context, xShape);
    gert::Shape* yShape = context->GetOutputShape(OUTPUT_Y_INDEX);
    OP_CHECK_NULL_WITH_CONTEXT(context, yShape);
    gert::Shape* argminShape = context->GetOutputShape(OUTPUT_ARGMIN_INDEX);
    OP_CHECK_NULL_WITH_CONTEXT(context, argminShape);
    // y 与 argmin 的 shape 与输入 x 一致
    *yShape = *xShape;
    *argminShape = *xShape;
    OP_LOGD(context->GetNodeName(), "End to do InferShapeForCummin");
    return GRAPH_SUCCESS;
}

static ge::graphStatus InferDataTypeForCummin(gert::InferDataTypeContext* context)
{
    // y.dtype -> x.dtype
    context->SetOutputDataType(OUTPUT_Y_INDEX, context->GetInputDataType(INPUT_X_INDEX));
    // argmin.dtype 固定为 int32
    context->SetOutputDataType(OUTPUT_ARGMIN_INDEX, ge::DT_INT32);
    return GRAPH_SUCCESS;
}

IMPL_OP_INFERSHAPE(Cummin).InferShape(InferShapeForCummin).InferDataType(InferDataTypeForCummin);
} // namespace ops
