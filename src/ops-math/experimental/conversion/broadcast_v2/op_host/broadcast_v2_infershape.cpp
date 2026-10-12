/**
 * This file is part of the OpenBOAT project at Harbin Institute of Technology (HIT)
 * and is contributed to the CANN Open Software.
 *
 * Copyright (c) 2025 AISS Group, Harbin Institute of Technology (HIT).
 * All Rights Reserved.
 *
 * Authors (accounts):
 * - Zhou Jianhua<@LePenseur>
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
 * \file broadcast_v2_infer.cpp
 * \brief
 */
#include "register/op_impl_registry.h"
#include "log/log.h"
using gert::InferShapeContext;
using gert::Shape;

// using namespace ge;

namespace ops {
static constexpr int64_t IDX_0 = 0;

static ge::graphStatus InferShapeBroadcastV2(InferShapeContext* context)
{
    OP_LOGD(context->GetNodeName(), "Begin to do InferShapeBroadcastV2");

    // 1. 获取输入Shape
    const Shape* xShape = context->GetInputShape(IDX_0);
    OP_CHECK_NULL_WITH_CONTEXT(context, xShape);

    // 2. 获取属性 (使用 gert::RuntimeAttrs)
    const gert::RuntimeAttrs* attrs = context->GetAttrs();
    OP_CHECK_NULL_WITH_CONTEXT(context, attrs);
    // 根据 broadcast_v2_def.cpp 中定义的顺序, axis是第3个(从0开始), num是第4个
    const int64_t axis = *(attrs->GetAttrPointer<int64_t>(3));
    const int64_t num = *(attrs->GetAttrPointer<int64_t>(4));

    // 3. 计算输出Shape
    auto xShapeSize = xShape->GetDimNum();
    std::vector<int64_t> yShapeVec(xShapeSize);
    for (size_t i = 0; i < xShapeSize; i++) {
        yShapeVec[i] = xShape->GetDim(i);
    }
    // 检查axis是否在有效范围内 (进行类型转换以避免编译警告)
    if (axis < 0 || static_cast<size_t>(axis) >= xShapeSize) {
        OP_LOGE(context->GetNodeName(), "Attr axis=%ld is out of range [0, %zu)", axis, xShapeSize);
        return ge::GRAPH_FAILED;
    }
    // 将广播轴的维度乘以num
    yShapeVec[axis] = yShapeVec[axis] * num;

    // 4. 设置输出Shape (直接修改输出Shape对象)
    Shape* yShape = context->GetOutputShape(IDX_0);
    OP_CHECK_NULL_WITH_CONTEXT(context, yShape);
    yShape->SetDimNum(yShapeVec.size());
    for (size_t i = 0; i < yShapeVec.size(); ++i) {
        yShape->SetDim(i, yShapeVec[i]);
    }

    OP_LOGD(context->GetNodeName(), "End to do InferShapeBroadcastV2");
    return ge::GRAPH_SUCCESS;
}

IMPL_OP_INFERSHAPE(BroadcastV2).InferShape(InferShapeBroadcastV2);
} // namespace ops