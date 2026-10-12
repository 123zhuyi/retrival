/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED, 
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include "register/op_impl_registry.h"
#include "log/log.h"

using namespace ge;
namespace ops {
static ge::graphStatus InferDataType4Ger(gert::InferDataTypeContext* context) {
    OP_LOGI(context->GetNodeName(), "InferDataType4Ger enter.");
    auto input_x_dtype = context->GetInputDataType(0);
    context->SetOutputDataType(0, input_x_dtype);
    OP_LOGI(context->GetNodeName(), "InferDataType4Ger end.");
    return GRAPH_SUCCESS;
}

IMPL_OP(Ger).InferDataType(InferDataType4Ger);
} // namespace ops