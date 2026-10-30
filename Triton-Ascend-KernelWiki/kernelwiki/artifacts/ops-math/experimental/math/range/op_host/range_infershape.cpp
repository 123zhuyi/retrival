/**
 * This file is part of the OpenBOAT project at Harbin Institute of Technology (HIT)
 * and is contributed to the CANN Open Software.
 *
 * Copyright (c) 2025 AISS Group, Harbin Institute of Technology (HIT).
 * All Rights Reserved.
 *
 * Authors (accounts):
 * - Li Zhi <@hitLeechi>
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
 * \file range_infer.cpp
 * \brief
 */
#include "log/log.h"
#include <cmath>
#include "register/op_impl_registry.h"
#include "register/op_def_registry.h"
#include "graph/utils/type_utils.h"
using namespace ge;

namespace ops {
const size_t startIdx = 0;
const size_t endIdx = 1;
const size_t stepIdx = 2;
const size_t outputIdx = 0;

static ge::graphStatus InferShape(gert::InferShapeContext* context)
{
    OP_CHECK_IF(context == nullptr, OP_LOGE(context, "context is nullptr"), return ge::GRAPH_FAILED);
    auto tensorStart = context->GetInputTensor(startIdx);
    auto tensorEnd = context->GetInputTensor(endIdx);
    auto tensorStep = context->GetInputTensor(stepIdx);
    OP_CHECK_IF(
        tensorStart == nullptr || tensorEnd == nullptr || tensorStep == nullptr,
        OP_LOGE(context, "Fault context,Please Check!"), return ge::GRAPH_FAILED);

    float startVal = 0.0f, endVal = 0.0f, stepVal = 1.0f;
    float diff = std::abs(endVal - startVal);
    float stepAbs = std::abs(stepVal);
    if (stepAbs == 0)
        return ge::GRAPH_FAILED;
    int64_t output_length = static_cast<int64_t>(std::ceil(diff / stepAbs));

    gert::Shape* y_shape = context->GetOutputShape(outputIdx);
    OP_CHECK_IF(y_shape == nullptr, OP_LOGE(context, "y_shape is nullptr"), return ge::GRAPH_FAILED);
    y_shape->SetDimNum(1);
    y_shape->SetDim(0, output_length);

    return ge::GRAPH_SUCCESS;
}
IMPL_OP_INFERSHAPE(Range).InputsDataDependency({0, 1, 2}).InferShape(InferShape);
} // namespace ops
