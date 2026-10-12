/**
 * This program is free software, you can redistribute it and/or modify it.
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This file is a part of the CANN Open Software.
 * Licensed under CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED, INCLUDING
 * BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

/*!
 * \file pow2_infer.cpp
 * \brief
 */
#include "register/op_impl_registry.h"
#include "log/log.h"

using namespace ge;

namespace ops {

static constexpr int64_t IDX_0 = 0;
static constexpr int64_t IDX_1 = 1;

static ge::graphStatus InferShapePow2(gert::InferShapeContext* context)
{
    OP_LOGD(context->GetNodeName(), "Begin to do InferShape");

    // get input shapes
    const gert::Shape* x1Shape = context->GetInputShape(IDX_0);
    OP_CHECK_NULL_WITH_CONTEXT(context, x1Shape);

    const gert::Shape* x2Shape = context->GetInputShape(IDX_1);
    OP_CHECK_NULL_WITH_CONTEXT(context, x2Shape);

    // get output shapes
    gert::Shape* yShape = context->GetOutputShape(IDX_0);
    OP_CHECK_NULL_WITH_CONTEXT(context, yShape);

    // 获取输入形状的维度信息
    auto dimNum1 = x1Shape->GetDimNum();
    auto dimNum2 = x2Shape->GetDimNum();
    uint32_t maxDim = std::max(dimNum1, dimNum2);

    std::vector<int64_t> outDims(maxDim, 1);

    // 从最后一个维度开始匹配广播
    for (uint32_t i = 0; i < maxDim; i++) {
        int64_t d1 = (i < dimNum1) ? x1Shape->GetDim(dimNum1 - 1 - i) : 1;
        int64_t d2 = (i < dimNum2) ? x2Shape->GetDim(dimNum2 - 1 - i) : 1;
        if (d1 == d2 || d1 == 1 || d2 == 1) {
            outDims[maxDim - 1 - i] = std::max(d1, d2);
        } else {
            OP_LOGE(context->GetNodeName(), "Dimension mismatch: d1=%ld, d2=%ld", d1, d2);
            return GRAPH_FAILED;
        }
    }

    // 设置输出形状
    yShape->SetDimNum(outDims.size());
    for (size_t i = 0; i < outDims.size(); ++i) {
        yShape->SetDim(i, outDims[i]);
    }

    OP_LOGD(context->GetNodeName(), "End to do InferShape");
    return GRAPH_SUCCESS;
}

IMPL_OP_INFERSHAPE(Pow2).InferShape(InferShapePow2);
} // namespace ops