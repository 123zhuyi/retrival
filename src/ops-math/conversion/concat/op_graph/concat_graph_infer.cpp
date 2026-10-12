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
 * \file concat_graph_infer.cpp
 * \brief concat operater graph infer resource
 */

#include "register/op_impl_registry.h"
#include "log/log.h"

using namespace ge;
namespace ops {
static ge::graphStatus InferDataTypeForConcat(gert::InferDataTypeContext* context)
{
    OP_LOGD(context->GetNodeName(), "Begin to do InferDataTypeForConcat");
    auto inputDesc = context->GetDynamicInputDesc(1, 0);
    OP_CHECK_NULL_WITH_CONTEXT(context, inputDesc);
    auto inputDataType = inputDesc->GetDataType();
    context->SetOutputDataType(0, inputDataType);
    OP_LOGD(context->GetNodeName(), "End to do InferDataTypeForConcat");
    return ge::GRAPH_SUCCESS;
}

IMPL_OP(Concat).InferDataType(InferDataTypeForConcat);
}; // namespace ops
