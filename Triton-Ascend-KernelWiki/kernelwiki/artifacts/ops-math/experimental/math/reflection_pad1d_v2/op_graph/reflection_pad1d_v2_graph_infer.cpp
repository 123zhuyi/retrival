/**
 * This file is part of the OpenBOAT project at Harbin Institute of Technology (HIT)
 * and is contributed to the CANN Open Software.
 *
 * Copyright (c) 2025 AISS Group, Harbin Institute of Technology (HIT).
 * All Rights Reserved.
 *
 * Authors (accounts):
 * - Zhou Jiamin <@zhou-jiamin-666>
 * - Su Tonghua <@sutonghua>
 *
 * This program is free software: you can redistribute it and/or modify it.
 * Licensed under the CANN Open Software License Agreement Version 2.0 (the "License").
 * You may not use this file except in compliance with the License.
 * See the LICENSE file at the root of the repository for the full text of the License.
 *
 * THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTIES OF ANY KIND, EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 */

/*!
 * \file reflection_pad1d_v2_graph_infer.cpp
 * \brief reflection_pad1d_v2 operater graph infer resource
 */
#include "register/op_impl_registry.h"
#include "log/log.h"

using namespace ge;
namespace ops {
constexpr size_t IDX_INPUT_X = 0;
constexpr size_t IDX_OUT_OUTPUT = 0;

static graphStatus InferDataTypeReflectionPad1dV2(gert::InferDataTypeContext* context)
{
    OP_LOGD(context->GetNodeName(), "InferDataTypeReflectionPad1dV2 enter");
    context->SetOutputDataType(IDX_OUT_OUTPUT, context->GetInputDesc(IDX_INPUT_X)->GetDataType());
    OP_LOGD(context->GetNodeName(), "InferDataTypeReflectionPad1dV2 end");
    return GRAPH_SUCCESS;
}
IMPL_OP(ReflectionPad1dV2).InferDataType(InferDataTypeReflectionPad1dV2);

}; // namespace ops
