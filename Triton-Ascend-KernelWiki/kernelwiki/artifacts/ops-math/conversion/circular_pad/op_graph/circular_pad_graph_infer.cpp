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
 * \file circular_pad_graph_infer.cpp
 * \brief circular_pad operator graph infer resource
 */

#include "log/log.h"
#include "register/op_impl_registry.h"
#include "util/shape_util.h"

static constexpr int INPUT_NODE_NUM = 2;
static constexpr int OUTPUT_NODE_NUM = 1;

using namespace ge;
namespace ops {

static ge::graphStatus InferShape4CircularPad(gert::InferShapeContext *context)
{
    if (context == nullptr) {
        OP_LOGE("InferShape4CircularPad", "Context is nullptr, check failed.");
        return GRAPH_FAILED;
    }
    if (context->GetComputeNodeInputNum() != INPUT_NODE_NUM ||
        context->GetComputeNodeOutputNum() != OUTPUT_NODE_NUM) {
        OP_LOGE("InferShape4CircularPad", "input or output num check failed.");
        return GRAPH_FAILED;
    }

    return ge::GRAPH_SUCCESS;
}

static ge::graphStatus InferDataType4CircularPad(gert::InferDataTypeContext* context)
{
    auto input_x_dtype = context->GetInputDataType(0);
    context->SetOutputDataType(0, input_x_dtype);
    return ge::GRAPH_SUCCESS;
}

IMPL_OP_INFERSHAPE(CircularPad)
    .InferShape(InferShape4CircularPad)
    .InferDataType(InferDataType4CircularPad);

} // namespace ops
