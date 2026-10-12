/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

/**
 *
 * NOTE: Portions of this code were AI-generated and have been
 * technically reviewed for functional accuracy and security
 */

/*!
 * \file asin_with_agent_infershape.cpp
 * \brief AsinWithAgent 算子形状推导实现
 *
 * 输出 shape 与输入 shape 相同（逐元素算子）
 * 注意：整数/BOOL 类型输入时，输出 dtype 为 FLOAT32（由 L0 API 处理）
 */

#include "register/op_impl_registry.h"
#include "exe_graph/runtime/infer_shape_context.h"

using namespace ge;

namespace ops {

static ge::graphStatus InferShape4AsinWithAgent(gert::InferShapeContext* context)
{
    // 获取输入形状
    const gert::Shape* input_shape = context->GetInputShape(0);
    if (input_shape == nullptr) {
        return ge::GRAPH_FAILED;
    }

    // 获取输出形状
    gert::Shape* output_shape = context->GetOutputShape(0);
    if (output_shape == nullptr) {
        return ge::GRAPH_FAILED;
    }

    // 输出 shape 与输入 shape 相同（逐元素算子）
    *output_shape = *input_shape;

    return ge::GRAPH_SUCCESS;
}

IMPL_OP_INFERSHAPE(AsinWithAgent).InferShape(InferShape4AsinWithAgent);

} // namespace ops
