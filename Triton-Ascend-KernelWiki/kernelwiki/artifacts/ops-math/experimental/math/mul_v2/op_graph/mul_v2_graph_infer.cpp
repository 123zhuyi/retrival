/**
 * This file is part of the OpenBOAT project at Harbin Institute of Technology (HIT)
 * and is contributed to the CANN Open Software.
 *
 * Copyright (c) 2025 AISS Group, Harbin Institute of Technology (HIT).
 * All Rights Reserved.
 *
 * Authors (accounts):
 * - Tu Yuanhang <@TuYHAAAAAA>
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
 * \file mul_v2_graph_infer.cpp
 * \brief mul_v2 operater graph infer resource
 */
#include "register/op_impl_registry.h"
#include "log/log.h"

namespace ops {
using namespace ge;

static constexpr int64_t IDX_0 = 0;

static ge::graphStatus InferDataTypeMulV2(gert::InferDataTypeContext* context)
{
    OP_LOGD(context->GetNodeName(), "Begin to do InferDataTypeMulV2");

    // 设置输出的dtype MulV2算子逻辑是两个数相乘，因此输出dataType与输入dataType一致
    ge::DataType sizeDtype = context->GetInputDataType(IDX_0);
    OP_CHECK_NULL_WITH_CONTEXT(context, context);
    context->SetOutputDataType(IDX_0, sizeDtype);
    OP_LOGD(context->GetNodeName(), "End to do InferDataTypeMulV2");
    return GRAPH_SUCCESS;
}

IMPL_OP(MulV2).InferDataType(InferDataTypeMulV2);

}; // namespace ops
