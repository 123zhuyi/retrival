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
 * \file matrix_diag_infershape.cpp
 * \brief infershape func of MatrixDiag
 */
#include <cmath>
#include <climits>
#include "log/log.h"
#include "register/op_impl_registry.h"

using namespace ge;
namespace ops {

static graphStatus InferShape4MatrixDiag(gert::InferShapeContext* context)
{
    OP_LOGD(context->GetNodeName(), "InferShape4MatrixDiag start");
    const gert::Shape* xShape = context->GetInputShape(0);
    OP_CHECK_NULL_WITH_CONTEXT(context, xShape);
    gert::Shape* yShape = context->GetOutputShape(0);
    OP_CHECK_NULL_WITH_CONTEXT(context, yShape);

    int64_t allDim = xShape->GetDimNum();
    int64_t xDimNum = xShape->GetDimNum() - 1;
    int64_t xDimNum1 = xShape->GetDim(xDimNum);
    yShape->SetDimNum(allDim + 1);
    for (int32_t i = 0; i < allDim; i++) {
        int64_t xDim = xShape->GetDim(i);
        yShape->SetDim(i, xDim);
    }
    yShape->SetDim(allDim, xDimNum1); // 设置输出shape
    OP_LOGD(context->GetNodeName(), "InferShape4MatrixDiag end");
    return GRAPH_SUCCESS;
}

IMPL_OP_INFERSHAPE(MatrixDiag).InferShape(InferShape4MatrixDiag);
} // namespace ops
