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
 * \file acos_grad_v2_infershape.cpp
 * \brief AcosGradV2 形状/类型推导（z 的 shape 与 dtype 均等于 y）
 */

#include "register/op_impl_registry.h"
#include "exe_graph/runtime/infer_shape_context.h"

using namespace ge;

namespace ops {

static ge::graphStatus InferShape4AcosGradV2(gert::InferShapeContext* context)
{
    const gert::Shape* yShape = context->GetInputShape(0);
    if (yShape == nullptr) {
        return ge::GRAPH_FAILED;
    }

    const gert::Shape* dyShape = context->GetInputShape(1);
    if (dyShape == nullptr) {
        return ge::GRAPH_FAILED;
    }

    gert::Shape* outputShape = context->GetOutputShape(0);
    if (outputShape == nullptr) {
        return ge::GRAPH_FAILED;
    }

    // 输出 z 的 shape 与 y 一致
    *outputShape = *yShape;

    return ge::GRAPH_SUCCESS;
}

IMPL_OP_INFERSHAPE(AcosGradV2).InferShape(InferShape4AcosGradV2);

} // namespace ops
