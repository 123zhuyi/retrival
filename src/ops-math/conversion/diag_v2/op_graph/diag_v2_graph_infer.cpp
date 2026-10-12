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
 * \file diag_v2_graph_infer.cpp
 * \brief diag_v2 operater graph infer resource
 */

#include "register/op_impl_registry.h"

using namespace ge;
namespace ops {
static constexpr size_t DiagV2_IN_X_IDX = 0;
static constexpr size_t DiagV2_OUT_Y_IDX = 0;

static graphStatus InferDataType4DiagV2(gert::InferDataTypeContext* context)
{
    context->SetOutputDataType(DiagV2_OUT_Y_IDX, context->GetInputDataType(DiagV2_IN_X_IDX));
    return GRAPH_SUCCESS;
}

IMPL_OP(DiagV2).InferDataType(InferDataType4DiagV2);
}; // namespace ops