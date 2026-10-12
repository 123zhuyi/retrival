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
 * \file reduce_var_graph_infer.cpp
 * \brief
 */

#include "register/op_impl_registry.h"
#include "log/log.h"

using namespace ge;
namespace {
const size_t INPUT_INDEX_X = 0;
const size_t OUTPUT_INDEX_VAR = 0;
const size_t OUTPUT_INDEX_MEAN = 1;
}  // namespace
namespace ops {

static ge::graphStatus InferDtype4ReduceVar(gert::InferDataTypeContext* context) {
    OP_LOGD(context->GetNodeName(), "InferDtype4ReduceVar enter");
    const ge::DataType x = context->GetInputDataType(INPUT_INDEX_X);
    context->SetOutputDataType(OUTPUT_INDEX_VAR, x);
    context->SetOutputDataType(OUTPUT_INDEX_MEAN, x);
    OP_LOGD(context->GetNodeName(), "InferDtype4ReduceVar success");
    return GRAPH_SUCCESS;
}

IMPL_OP(ReduceVar).InferDataType(InferDtype4ReduceVar);
}  // namespace ops
