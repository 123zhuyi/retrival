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
 * \file log_graph_infer.cpp
 * \brief log operater graph infer resource
 */

#include "register/op_impl_registry.h"
#include "log/log.h"

using namespace ge;
namespace ops {

static ge::graphStatus InferDataType4Log(gert::InferDataTypeContext* context)
{
    OP_LOGI("Begin InferDataType4Log");
    const ge::DataType input_type = context->GetInputDataType(0);
    if (input_type == DT_UINT8 || input_type == DT_INT8 || input_type == DT_INT16 || input_type == DT_INT32 ||
        input_type == DT_INT64 || input_type == DT_BOOL) {
        context->SetOutputDataType(0, DT_FLOAT);
        return ge::GRAPH_SUCCESS;
    }

    context->SetOutputDataType(0, input_type);
    return ge::GRAPH_SUCCESS;
}

IMPL_OP(Log).InferDataType(InferDataType4Log);
}; // namespace ops