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
 * \file dot_infershape.cpp
 * \brief
 */
#include "log/log.h"
#include "register/op_impl_registry.h"
#include "op_host/util/shape_util.h"

namespace ops
{
using namespace Ops::Base;

const size_t SUPPORTED_DIM_NUM = 1;

static ge::graphStatus InferShapeForDot(gert::InferShapeContext* context)
{
    auto inputXShape = context->GetInputShape(0);
    OP_CHECK_NULL_WITH_CONTEXT(context, inputXShape);

    auto inputYShape = context->GetInputShape(1);
    OP_CHECK_NULL_WITH_CONTEXT(context, inputYShape);

    if (!IsUnknownRank(*inputXShape)) {
        // -2 no need check dim num
        OP_CHECK_IF(inputXShape->GetDimNum() != SUPPORTED_DIM_NUM,
                 OP_LOGE(context->GetNodeName(), "The dim of input_x must be 1."),
                 return ge::GRAPH_FAILED);
    }
    if (!IsUnknownRank(*inputYShape)) {
        // -2 no need check dim num
        OP_CHECK_IF(inputYShape->GetDimNum() != SUPPORTED_DIM_NUM,
                 OP_LOGE(context->GetNodeName(), "The dim of input_y must be 1."),
                 return ge::GRAPH_FAILED);
    }
    bool isDynamic = (IsUnknownRank(*inputXShape) || IsUnknownRank(*inputYShape) || IsUnknownShape(*inputXShape) ||
                      IsUnknownShape(*inputYShape));
    if (!isDynamic) {
        OP_CHECK_IF(
            inputXShape->GetDim(0) != inputYShape->GetDim(0),
            OP_LOGE(context->GetNodeName(), "The 0-dim of input_x and input_y not match."),
            return ge::GRAPH_FAILED);
    }

    auto dotOutput = context->GetOutputShape(0);
    OP_CHECK_NULL_WITH_CONTEXT(context, dotOutput);
    dotOutput->SetDimNum(0);
    return ge::GRAPH_SUCCESS;
}

IMPL_OP_INFERSHAPE(Dot).InferShape(InferShapeForDot);
}  // namespace ops