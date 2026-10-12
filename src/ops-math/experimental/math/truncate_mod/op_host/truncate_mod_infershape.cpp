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
 * \file truncate_mod_infershape.cpp
 * \brief TruncateMod shape / dtype 推导。输出 y 与输入 x1 保持一致。
 */
#include "register/op_impl_registry.h"
#include "log/log.h"
#include "exe_graph/runtime/infer_shape_context.h"

using namespace ge;

namespace ops {
static constexpr size_t INPUT_X1_IDX = 0;
static constexpr size_t OUTPUT_Y_IDX = 0;

static ge::graphStatus InferShapeTruncateMod(gert::InferShapeContext* context)
{
    const gert::Shape* x1Shape = context->GetInputShape(INPUT_X1_IDX);
    OP_CHECK_NULL_WITH_CONTEXT(context, x1Shape);
    gert::Shape* outputShape = context->GetOutputShape(OUTPUT_Y_IDX);
    OP_CHECK_NULL_WITH_CONTEXT(context, outputShape);
    *outputShape = *x1Shape;
    return ge::GRAPH_SUCCESS;
}

static ge::graphStatus InferDataTypeTruncateMod(gert::InferDataTypeContext* context)
{
    context->SetOutputDataType(OUTPUT_Y_IDX, context->GetInputDataType(INPUT_X1_IDX));
    return ge::GRAPH_SUCCESS;
}

IMPL_OP_INFERSHAPE(TruncateMod).InferShape(InferShapeTruncateMod).InferDataType(InferDataTypeTruncateMod);
} // namespace ops
