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
 * \file random_uniform_int_v2_graph_infer.cpp
 * \brief random_uniform_int_v2 operator graph infer resource
 */

#include "log/log.h"
#include "register/op_impl_registry.h"

using namespace ge;
namespace ops {

static graphStatus RandomUniformIntV2InferDataType(gert::InferDataTypeContext* context)
{
    OP_LOGD(context->GetNodeName(), "Begin to do RandomUniformIntV2InferDataType");
    // 实现推导逻辑
    DataType inputDataType = context->GetInputDataType(1);
    context->SetOutputDataType(0, inputDataType);
    context->SetOutputDataType(1, ge::DT_INT64);
    return ge::GRAPH_SUCCESS;
}

IMPL_OP(RandomUniformIntV2).InferDataType(RandomUniformIntV2InferDataType);
} // namespace ops