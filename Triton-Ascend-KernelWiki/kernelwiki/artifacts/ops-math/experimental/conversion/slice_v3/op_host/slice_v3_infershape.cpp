/**
 * This file is part of the OpenBOAT project at Harbin Institute of Technology (HIT)
 * and is contributed to the CANN Open Software.
 *
 * Copyright (c) 2025 AISS Group, Harbin Institute of Technology (HIT).
 * All Rights Reserved.
 *
 * Authors (accounts):
 * - Cao Xiaojuan <@c15503545287>
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
 * \file slice_v3_infershape.cpp
 * \brief
 */
#include "register/op_impl_registry.h"
#include "log/log.h"

using namespace ge;

namespace ops {

static ge::graphStatus InferShapeSliceV3(gert::InferShapeContext* context)
{
    const gert::Shape* xShape = context->GetInputShape(0);
    OP_CHECK_NULL_WITH_CONTEXT(context, xShape);

    uint32_t dim = xShape->GetDimNum();

    const gert::Tensor* sizeTensor = context->GetInputTensor(2);
    OP_CHECK_NULL_WITH_CONTEXT(context, sizeTensor);

    // 校验 sizeTensor 的 shape
    const gert::Shape* sizeShape = context->GetInputShape(2);
    OP_CHECK_NULL_WITH_CONTEXT(context, sizeShape);

    // 1. 必须是一维张量
    if (sizeShape->GetDimNum() != 1) {
        OP_LOGE(context, "sizeTensor must be 1-D, got %zu-D", sizeShape->GetDimNum());
        return GRAPH_FAILED;
    }

    // 2. 长度必须等于 dim
    int64_t sizeLength = sizeShape->GetDim(0);
    if (sizeLength < 0 || static_cast<uint32_t>(sizeLength) != dim) {
        OP_LOGE(context, "sizeTensor length must be %u, got %ld", dim, sizeLength);
        return GRAPH_FAILED;
    }

    const int64_t* sizeData = sizeTensor->GetData<int64_t>();
    OP_CHECK_NULL_WITH_CONTEXT(context, sizeData);

    gert::Shape yShape;
    yShape.SetDimNum(dim);

    for (uint32_t i = 0; i < dim; ++i) {
        int64_t outputDim = sizeData[i];
        yShape.SetDim(i, outputDim);
    }

    *context->GetOutputShape(0) = yShape;

    return GRAPH_SUCCESS;
}

IMPL_OP_INFERSHAPE(SliceV3).InferShape(InferShapeSliceV3);
} // namespace ops