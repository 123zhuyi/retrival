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
 * \file im2col_graph_infer.cpp
 * \brief infershape func of Im2col
 */
#include "log/log.h"
#include "register/op_impl_registry.h"

using namespace ge;
namespace ops {

static graphStatus InferDataType4Im2col(gert::InferDataTypeContext* context)
{
    OP_LOGD(context->GetNodeName(), "InferDataType4Im2col start");
    auto inputXDtype = context->GetInputDataType(0);
    context->SetOutputDataType(0, inputXDtype);
    OP_LOGD(context->GetNodeName(), "InferDataType4Im2col end");
    return GRAPH_SUCCESS;
}

IMPL_OP(Im2col).InferDataType(InferDataType4Im2col);
} // namespace ops