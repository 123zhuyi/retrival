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
 * \file fused_mul_add_nl2loss_infershape.cpp
 * \brief y1 shape = x1 shape；y2 = 0 维标量（空 dims）
 *        与 canndev built-in FusedMulAddNL2lossInferShape（elewise_calculation_ops.cc）
 *        功能完全一致，gert infershape 2.0 写法
 */
#include "register/op_impl_registry.h"

namespace ops {
static ge::graphStatus FusedMulAddNL2lossInferShape(gert::InferShapeContext* context)
{
    const gert::Shape* x1Shape = context->GetInputShape(0);
    gert::Shape* y1Shape = context->GetOutputShape(0);
    gert::Shape* y2Shape = context->GetOutputShape(1);

    // y1 shape = x1 shape
    size_t dimNum = x1Shape->GetDimNum();
    y1Shape->SetDimNum(dimNum);
    for (size_t i = 0; i < dimNum; i++) {
        y1Shape->SetDim(i, x1Shape->GetDim(i));
    }

    // y2 = 0 维标量（空 dims），与 910b common inferfunc 一致
    y2Shape->SetDimNum(0);

    return ge::GRAPH_SUCCESS;
}

IMPL_OP_INFERSHAPE(FusedMulAddNL2loss).InferShape(FusedMulAddNL2lossInferShape);
} // namespace ops
