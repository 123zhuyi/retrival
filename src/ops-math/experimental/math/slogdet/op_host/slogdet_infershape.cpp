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
 * 我们正常的版权申明，下面是我们的备注
 *
 * NOTE: Portions of this code were AI-generated and have been
 * technically reviewed for functional accuracy and security
 */

/*!
 * \file slogdet_infershape.cpp
 * \brief Slogdet 形状/数据类型推导：
 *        - 两个输出 signOut/logOut 的 shape = self.shape[:-2]（batch 形状）。
 *          self=[n,n] => 标量 []（dimNum=0）；self=[b,n,n] => [b]。
 *        - 两个输出 dtype = self.dtype（fp32）。
 */

#include "register/op_impl_registry.h"
#include "exe_graph/runtime/infer_shape_context.h"

using namespace ge;

namespace ops {

static ge::graphStatus InferShape4Slogdet(gert::InferShapeContext* context)
{
    const gert::Shape* inputShape = context->GetInputShape(0);
    if (inputShape == nullptr) {
        return ge::GRAPH_FAILED;
    }
    gert::Shape* signShape = context->GetOutputShape(0);
    gert::Shape* logShape = context->GetOutputShape(1);
    if (signShape == nullptr || logShape == nullptr) {
        return ge::GRAPH_FAILED;
    }

    const size_t inputRank = inputShape->GetDimNum();
    // self 必须 >= 2 维（方阵 batch），否则非法
    if (inputRank < 2) {
        return ge::GRAPH_FAILED;
    }

    // 输出 batch 形状 = self.shape[:-2]
    const size_t batchRank = inputRank - 2;
    signShape->SetDimNum(batchRank);
    logShape->SetDimNum(batchRank);
    for (size_t i = 0; i < batchRank; i++) {
        const int64_t dimVal = inputShape->GetDim(i);
        signShape->SetDim(i, dimVal);
        logShape->SetDim(i, dimVal);
    }

    return ge::GRAPH_SUCCESS;
}

static ge::graphStatus InferDataType4Slogdet(gert::InferDataTypeContext* context)
{
    const ge::DataType selfDtype = context->GetInputDataType(0);
    context->SetOutputDataType(0, selfDtype);
    context->SetOutputDataType(1, selfDtype);
    return ge::GRAPH_SUCCESS;
}

IMPL_OP_INFERSHAPE(Slogdet)
    .InferShape(InferShape4Slogdet)
    .InferDataType(InferDataType4Slogdet);

} // namespace ops
