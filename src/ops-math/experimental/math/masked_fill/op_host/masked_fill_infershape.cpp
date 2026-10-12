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
 * \file masked_fill_infer.cpp
 * \brief
 */
#include "register/op_impl_registry.h"
#include "log/log.h"

using namespace ge;

namespace ops {
static constexpr size_t IDX_IN_X = 0;
static constexpr size_t IDX_OUT_OUTPUT = 0;

static ge::graphStatus InferShapeMaskedFill(gert::InferShapeContext* context)
{
    const gert::Shape* x1_shape = context->GetInputShape(IDX_IN_X);
    gert::Shape* y_shape = context->GetOutputShape(IDX_OUT_OUTPUT);
    *y_shape = *x1_shape;
    return ge::GRAPH_SUCCESS;
}
IMPL_OP_INFERSHAPE(MaskedFill).InferShape(InferShapeMaskedFill);
} // namespace ops