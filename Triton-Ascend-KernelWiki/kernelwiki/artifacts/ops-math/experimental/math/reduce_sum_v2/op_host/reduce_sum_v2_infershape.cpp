/**
 * This file is part of the OpenBOAT project at Harbin Institute of Technology (HIT)
 * and is contributed to the CANN Open Software.
 *
 * Copyright (c) 2025 AISS Group, Harbin Institute of Technology (HIT).
 * All Rights Reserved.
 *
 * Authors (accounts):
 * - Pei Haobo<@xiaopei-1>
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
 * \file reduce_sum_v2_infershape.cpp
 * \brief
 */
#include "register/op_impl_registry.h"
#include "log/log.h"

using namespace ge;

namespace ops {
static constexpr int64_t IDX_0 = 0;

static ge::graphStatus InferShapeReduceSumV2(gert::InferShapeContext* context)
{
    OP_LOGD(context->GetNodeName(), "Begin to do InferShapeReduceSumV2");

    // get input shapes
    const gert::Shape* xShape = context->GetInputShape(IDX_0);
    OP_CHECK_NULL_WITH_CONTEXT(context, xShape);

    // get output shapes
    gert::Shape* yShape = context->GetOutputShape(IDX_0);
    OP_CHECK_NULL_WITH_CONTEXT(context, yShape);

    auto attrs = context->GetAttrs();
    int32_t axes = -1; // 默认值
    int32_t keepdims = 1;
    if (attrs != nullptr) {
        if (attrs->GetInt(0)) {
            axes = *(attrs->GetInt(0));
        }
        if (attrs->GetInt(1)) {
            keepdims = *(attrs->GetInt(1));
        }
    }
    auto xShapeSize = xShape->GetDimNum();
    if (keepdims == 1) {
        yShape->SetDimNum(xShapeSize);
        if (axes == -1) {
            for (size_t i = 0; i < xShapeSize; i++) {
                yShape->SetDim(i, 1);
            }
        } else {
            yShape->SetDim(axes, 1);
            if (axes == 0) {
                yShape->SetDim(1, xShape->GetDim(1));
            } else {
                yShape->SetDim(0, xShape->GetDim(0));
            }
        }
    } else {
        yShape->SetDimNum(1);
        if (axes == 0) {
            yShape->SetDim(0, xShape->GetDim(1));
        } else if (axes == 1) {
            yShape->SetDim(0, xShape->GetDim(0));
        } else {
            yShape->SetDim(0, 1);
        }
    }
    OP_LOGD(context->GetNodeName(), "End to do InferShapeReduceSumV2");
    return GRAPH_SUCCESS;
}

IMPL_OP_INFERSHAPE(ReduceSumV2).InferShape(InferShapeReduceSumV2);
} // namespace ops