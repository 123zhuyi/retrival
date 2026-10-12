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
 * \file exp_segsum_grad_infershape.cpp
 * \brief ExpSegsumGrad shape and dtype inference.
 */

#include "register/op_impl_registry.h"
#include "log/log.h"
#include <string>

using namespace ge;

namespace {
constexpr int64_t GRAD_OUTPUT_IDX = 0;
constexpr int64_t GRAD_INPUT_IDX = 0;
constexpr int64_t MIN_INPUT_RANK = 2;
constexpr int64_t UNKNOWN_RANK_DIM_VALUE = -2;
} // namespace

namespace ops {
static ge::graphStatus InferShape4ExpSegsumGrad(gert::InferShapeContext* context)
{
    auto gradOutputShape = context->GetInputShape(GRAD_OUTPUT_IDX);
    OP_CHECK_NULL_WITH_CONTEXT(context, gradOutputShape);
    auto gradInputShape = context->GetOutputShape(GRAD_INPUT_IDX);
    OP_CHECK_NULL_WITH_CONTEXT(context, gradInputShape);

    const int64_t dimNum = static_cast<int64_t>(gradOutputShape->GetDimNum());
    if (dimNum == 1 && gradOutputShape->GetDim(0) == UNKNOWN_RANK_DIM_VALUE) {
        gradInputShape->SetDimNum(0);
        gradInputShape->AppendDim(UNKNOWN_RANK_DIM_VALUE);
        return ge::GRAPH_SUCCESS;
    }

    OP_CHECK_IF(dimNum < MIN_INPUT_RANK,
                OP_LOGE_FOR_INVALID_SHAPEDIM(context->GetNodeName(), "grad_output", std::to_string(dimNum).c_str(),
                                             "greater than or equal to 2"),
                return ge::GRAPH_FAILED);

    gradInputShape->SetDimNum(dimNum - 1);
    for (int64_t i = 0; i < dimNum - 1; ++i) {
        gradInputShape->SetDim(i, gradOutputShape->GetDim(i));
    }
    return ge::GRAPH_SUCCESS;
}

static ge::graphStatus InferDataType4ExpSegsumGrad(gert::InferDataTypeContext* context)
{
    context->SetOutputDataType(GRAD_INPUT_IDX, context->GetInputDataType(GRAD_OUTPUT_IDX));
    return ge::GRAPH_SUCCESS;
}

IMPL_OP_INFERSHAPE(ExpSegsumGrad).InferShape(InferShape4ExpSegsumGrad).InferDataType(InferDataType4ExpSegsumGrad);
} // namespace ops
