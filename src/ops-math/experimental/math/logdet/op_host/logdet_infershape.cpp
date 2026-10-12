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
 * \file logdet_infershape.cpp
 * \brief Logdet shape/type inference:
 *        - output shape = self.shape[:-2] (batch shape).
 *          self=[n,n] => scalar [] (dimNum=0); self=[b,n,n] => [b].
 *        - output dtype = self.dtype (fp32).
 */

#include "register/op_impl_registry.h"
#include "exe_graph/runtime/infer_shape_context.h"

using namespace ge;

namespace ops {

static ge::graphStatus InferShape4Logdet(gert::InferShapeContext* context)
{
    const gert::Shape* inputShape = context->GetInputShape(0);
    if (inputShape == nullptr) {
        return ge::GRAPH_FAILED;
    }
    gert::Shape* outShape = context->GetOutputShape(0);
    if (outShape == nullptr) {
        return ge::GRAPH_FAILED;
    }

    const size_t inputRank = inputShape->GetDimNum();
    // self 必须 >= 2 维（方阵 batch），否则非法
    if (inputRank < 2) {
        return ge::GRAPH_FAILED;
    }

    // 输出 batch 形状 = self.shape[:-2]
    const size_t batchRank = inputRank - 2;
    outShape->SetDimNum(batchRank);
    for (size_t i = 0; i < batchRank; i++) {
        outShape->SetDim(i, inputShape->GetDim(i));
    }

    return ge::GRAPH_SUCCESS;
}

static ge::graphStatus InferDataType4Logdet(gert::InferDataTypeContext* context)
{
    const ge::DataType selfDtype = context->GetInputDataType(0);
    context->SetOutputDataType(0, selfDtype);
    return ge::GRAPH_SUCCESS;
}

IMPL_OP_INFERSHAPE(Logdet)
    .InferShape(InferShape4Logdet)
    .InferDataType(InferDataType4Logdet);

} // namespace ops
