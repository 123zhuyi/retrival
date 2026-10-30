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
 * \file reflection_pad1d_v2_infer.cpp
 * \brief
 */
#include "register/op_impl_registry.h"
#include "log/log.h"

using namespace ge;

namespace ops {
constexpr size_t IDX_IN_PADDING = 1;
constexpr size_t PAD_W_FRONT_INDEX = 0; // W前填充（padLeft）
constexpr size_t PAD_W_BACK_INDEX = 1;  // W后填充（padRight）
static ge::graphStatus InferShapeReflectionPad1dV2(gert::InferShapeContext* context)
{
    if (context == nullptr) {
        return ge::GRAPH_FAILED;
    }

    // 获取输入形状
    const gert::Shape* inputShape = context->GetInputShape(0); // 输入x的形状
    if (inputShape == nullptr) {
        return ge::GRAPH_FAILED;
    }
    uint32_t inputDim = static_cast<uint32_t>(inputShape->GetDimNum());
    if (inputDim < 1 || inputDim > 3) { // 限制输入维度为1D~3D
        return ge::GRAPH_FAILED;
    }

    // 获取paddings参数
    const auto paddingTensor = context->GetInputTensor(1); // 输入paddings的张量
    if (paddingTensor == nullptr) {
        return ge::GRAPH_FAILED;
    }
    const gert::Shape* padShape = context->GetInputShape(1);
    if (padShape == nullptr || padShape->GetDimNum() != 1 || padShape->GetDim(0) != 2) {
        return ge::GRAPH_FAILED;
    }

    ge::DataType padDtype = paddingTensor->GetDataType();
    uint32_t padLeft = 0, padRight = 0;
    if (padDtype == ge::DT_INT32) {
        const int32_t* padData = paddingTensor->GetData<int32_t>();
        if (padData == nullptr)
            return ge::GRAPH_FAILED;
        padLeft = static_cast<uint32_t>(padData[PAD_W_FRONT_INDEX]);
        padRight = static_cast<uint32_t>(padData[PAD_W_BACK_INDEX]);
    } else if (padDtype == ge::DT_INT64) {
        const int64_t* padData = paddingTensor->GetData<int64_t>();
        if (padData == nullptr)
            return ge::GRAPH_FAILED;
        padLeft = static_cast<uint32_t>(padData[PAD_W_FRONT_INDEX]);
        padRight = static_cast<uint32_t>(padData[PAD_W_BACK_INDEX]);
    } else {
        return ge::GRAPH_FAILED; // 仅支持int32/int64类型paddings
    }

    if (padLeft > static_cast<uint32_t>(INT32_MAX) || padRight > static_cast<uint32_t>(INT32_MAX)) {
        return ge::GRAPH_FAILED;
    }

    // 计算输出形状
    gert::Shape* outputShape = context->GetOutputShape(0);
    if (outputShape == nullptr) {
        return ge::GRAPH_FAILED;
    }
    *outputShape = *inputShape;
    uint32_t wDimIndex = static_cast<uint32_t>(inputShape->GetDimNum() - 1);
    uint32_t inWSize = static_cast<uint32_t>(inputShape->GetDim(wDimIndex));

    if (static_cast<uint64_t>(inWSize) + padLeft + padRight > static_cast<uint64_t>(UINT32_MAX)) {
        return ge::GRAPH_FAILED;
    }
    uint32_t outWSize = inWSize + padLeft + padRight;
    outputShape->SetDim(wDimIndex, outWSize);

    return ge::GRAPH_SUCCESS;
}
IMPL_OP_INFERSHAPE(ReflectionPad1dV2).InputsDataDependency({IDX_IN_PADDING}).InferShape(InferShapeReflectionPad1dV2);
} // namespace ops