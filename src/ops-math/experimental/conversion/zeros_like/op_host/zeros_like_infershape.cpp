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
 * \file zeros_like_infershape.cpp
 * \brief experimental 自包含 ZerosLike InferShape / InferDataType（输出与输入同形、同 dtype）。
 */
#include "register/op_impl_registry.h"
#include "log/log.h"

using namespace ge;

namespace ops {
static constexpr int64_t IDX_0 = 0;

static ge::graphStatus InferShapeZerosLike(gert::InferShapeContext* context)
{
    OP_LOGD(context->GetNodeName(), "Begin to do InferShapeZerosLike");

    // get input shape
    const gert::Shape* xShape = context->GetInputShape(IDX_0);
    OP_CHECK_NULL_WITH_CONTEXT(context, xShape);

    // get output shape
    gert::Shape* yShape = context->GetOutputShape(IDX_0);
    OP_CHECK_NULL_WITH_CONTEXT(context, yShape);

    // 输出与输入同形
    auto xShapeSize = xShape->GetDimNum();
    yShape->SetDimNum(xShapeSize);
    for (size_t i = 0; i < xShapeSize; i++) {
        int64_t dim = xShape->GetDim(i);
        yShape->SetDim(i, dim);
    }

    OP_LOGD(context->GetNodeName(), "End to do InferShapeZerosLike");
    return GRAPH_SUCCESS;
}

static ge::graphStatus InferDataTypeZerosLike(gert::InferDataTypeContext* context)
{
    OP_LOGD(context->GetNodeName(), "Begin to do InferDataTypeZerosLike");
    auto inputDtype = context->GetInputDataType(IDX_0);
    context->SetOutputDataType(IDX_0, inputDtype);
    OP_LOGD(context->GetNodeName(), "End to do InferDataTypeZerosLike");
    return GRAPH_SUCCESS;
}

IMPL_OP_INFERSHAPE(ZerosLike).InferShape(InferShapeZerosLike).InferDataType(InferDataTypeZerosLike);
} // namespace ops
