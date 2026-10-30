/**
 * This file is part of the OpenBOAT project at Harbin Institute of Technology (HIT)
 * and is contributed to the CANN Open Software.
 *
 * Copyright (c) 2025 AISS Group, Harbin Institute of Technology (HIT).
 * All Rights Reserved.
 *
 * Authors (accounts):
 * - Shi Xiangyang <@shi-xiangyang225>
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
 * \file expandv_infer.cpp
 * \brief
 */
#include "register/op_impl_registry.h"
#include "log/log.h"

using namespace ge;

namespace ops {

constexpr size_t IDX_0 = 0;

static ge::graphStatus InferShapeExpandv(gert::InferShapeContext* context)
{
    OP_LOGD(context->GetNodeName(), "Begin to do InferShapeExpandv");

    // get input shapes
    const gert::Shape* xShape = context->GetInputShape(IDX_0);
    OP_CHECK_NULL_WITH_CONTEXT(context, xShape);

    // get output shapes
    gert::Shape* yShape = context->GetOutputShape(IDX_0);
    OP_CHECK_NULL_WITH_CONTEXT(context, yShape);

    // 获取目标广播 shape
    const gert::TypedContinuousVector<int64_t>* shape_list = context->GetAttrs()->GetListInt(0);
    if (shape_list == nullptr) {
        return GRAPH_FAILED;
    }
    const int64_t* shape = shape_list->GetData();
    // 设置输出 shape
    yShape->SetDimNum(shape_list->GetSize());
    for (size_t i = 0; i < shape_list->GetSize(); ++i) {
        yShape->SetDim(i, shape[i]);
    }
    // ====== 合法性检查：是否满足 numpy.broadcast_to 的规则 ======
    const int64_t inRank = xShape->GetDimNum();
    const int64_t outRank = shape_list->GetSize();
    // 从后往前对齐广播
    int64_t inIdx = inRank - 1;
    int64_t outIdx = outRank - 1;

    while (inIdx >= 0 && outIdx >= 0) {
        int64_t in_dim = xShape->GetDim(inIdx);
        int64_t out_dim = shape[outIdx];
        // 只有 in_dim == out_dim 或 in_dim == 1 才能广播
        if (!(in_dim == out_dim || in_dim == 1)) {
            // 违反 numpy 广播规则
            return GRAPH_FAILED;
        }
        inIdx--;
        outIdx--;
    }
    // 剩下的更高维（只存在 out 侧）必须 >=1
    // 因为输入缺省相当于 1
    for (; outIdx >= 0; --outIdx) {
        if (shape[outIdx] <= 0) {
            return GRAPH_FAILED;
        }
    }
    return GRAPH_SUCCESS;

    OP_LOGD(context->GetNodeName(), "End to do InferShapeExpandv");
    return GRAPH_SUCCESS;
}

IMPL_OP_INFERSHAPE(Expandv).InferShape(InferShapeExpandv);
} // namespace ops