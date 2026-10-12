/**
 * This file is part of the OpenBOAT project at Harbin Institute of Technology (HIT)
 * and is contributed to the CANN Open Software.
 *
 * Copyright (c) 2025 AISS Group, Harbin Institute of Technology (HIT).
 * All Rights Reserved.
 *
 * Authors (accounts):
 * - Li Wen <@liwenkkklll>
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
 * \file square_v2_graph_infer.cpp
 * \brief square_v2 operater graph infer resource
 */
#include "register/op_impl_registry.h"
#include "log/log.h"

namespace ops {
using namespace ge;

static constexpr int64_t IDX_0 = 0;

static ge::graphStatus InferDataTypeSquareV2(gert::InferDataTypeContext* context)
{
    OP_LOGD(context->GetNodeName(), "Begin to do InferDataTypeSquareV2");

    ge::DataType xDtype = context->GetInputDataType(IDX_0);
    context->SetOutputDataType(IDX_0, xDtype);

    OP_LOGD(context->GetNodeName(), "End to do InferDataTypeSquareV2");
    return GRAPH_SUCCESS;
}

IMPL_OP(SquareV2).InferDataType(InferDataTypeSquareV2);

}; // namespace ops
