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
 * \file interleave_rope_infer.cpp
 * \brief
 */
#include <sstream>
#include <string>
#include <vector>
#include "register/op_def_registry.h"
#include "log/log.h"
#include "util/math_util.h"
#include "util/shape_util.h"

using namespace ge;
namespace ops {
constexpr size_t INPUT_IDX_X = 0;
constexpr size_t INPUT_IDX_COS = 1;
constexpr size_t INPUT_IDX_SIN = 2;
constexpr size_t OUTPUT_IDX_Y = 0;

graphStatus InferShape4InterleaveRope(gert::InferShapeContext* context)
{
    OP_LOGI(context, "Begin to do InferShape4InterleaveRope.");

    const gert::Shape* xInputShape = context->GetInputShape(INPUT_IDX_X);
    OP_CHECK_NULL_WITH_CONTEXT(context, xInputShape);
    gert::Shape* yShape = context->GetOutputShape(OUTPUT_IDX_Y);
    OP_CHECK_NULL_WITH_CONTEXT(context, yShape);
    *yShape = *xInputShape;

    OP_LOGD(context, "End to do InferShape4InterleaveRope.");
    return GRAPH_SUCCESS;
}

graphStatus InferDtype4InterleaveRope(gert::InferDataTypeContext* context)
{
    OP_LOGD(context, "InferDtype4InterleaveRope enter");

    auto x_dtype = context->GetInputDataType(INPUT_IDX_X);
    context->SetOutputDataType(OUTPUT_IDX_Y, x_dtype);

    OP_LOGD(context, "InferDtype4InterleaveRope end");

    return GRAPH_SUCCESS;
}

IMPL_OP_INFERSHAPE(InterleaveRope).InferShape(InferShape4InterleaveRope).InferDataType(InferDtype4InterleaveRope);
} // namespace ops
