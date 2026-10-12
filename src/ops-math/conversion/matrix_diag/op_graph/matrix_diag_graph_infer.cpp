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
#include "log/log.h"
#include "register/op_impl_registry.h"

using namespace ge;
namespace ops {

static graphStatus InferDataType4MatrixDiag(gert::InferDataTypeContext* context)
{
    OP_LOGD(context->GetNodeName(), "InferDataType4MatrixDiag start");
    auto inputXDtype = context->GetInputDataType(1);
    context->SetOutputDataType(0, inputXDtype);
    OP_LOGD(context->GetNodeName(), "InferDataType4MatrixDiag end");
    return GRAPH_SUCCESS;
}

IMPL_OP(MatrixDiag).InferDataType(InferDataType4MatrixDiag);
} // namespace ops
