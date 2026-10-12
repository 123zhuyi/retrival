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
 * \file hans_encode_graph_infer.cpp
 * \brief hans_encode operater graph infer resource
 */

#include "register/op_impl_registry.h"

using namespace ge;
namespace ops {
static constexpr int IDX_0 = 0;
static constexpr int IDX_1 = 1;
static constexpr int IDX_2 = 2;
static constexpr int IDX_3 = 3;

static ge::graphStatus InferDataType4HansEncode(gert::InferDataTypeContext* context)
{
    ge::DataType sizeDtype = context->GetInputDataType(IDX_0);
    context->SetOutputDataType(IDX_1, sizeDtype);
    context->SetOutputDataType(IDX_2, sizeDtype);
    context->SetOutputDataType(IDX_3, sizeDtype);
    return ge::GRAPH_SUCCESS;
}

IMPL_OP(HansEncode).InferDataType(InferDataType4HansEncode);
}; // namespace ops