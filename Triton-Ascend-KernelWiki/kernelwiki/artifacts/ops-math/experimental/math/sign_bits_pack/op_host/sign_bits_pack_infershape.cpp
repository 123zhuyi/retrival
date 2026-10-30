/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

/*!
 * \file sign_bits_pack_infershape.cpp
 * \brief
 */
#include "register/op_impl_registry.h"
#include "log/log.h"

using namespace ge;

namespace ops {
static constexpr int64_t IDX_0 = 0;
static constexpr int64_t PACKING_FACTOR = 8;

static ge::graphStatus InferShapeSignBitsPack(gert::InferShapeContext* context)
{
    OP_CHECK_IF(context == nullptr, OP_LOGE(context, "context is nullptr"), return ge::GRAPH_FAILED);
    OP_LOGD(context->GetNodeName(), "Begin to do InferShapeSignBitsPack");

    // get input shapes
    const gert::Shape* xShape = context->GetInputShape(IDX_0);
    OP_CHECK_NULL_WITH_CONTEXT(context, xShape);

    // get output shapes
    gert::Shape* yShape = context->GetOutputShape(IDX_0);
    OP_CHECK_NULL_WITH_CONTEXT(context, yShape);

    const size_t inputSize = xShape->GetDimNum() > 0 ? static_cast<size_t>(xShape->GetDim(0)) : 0;

    if (inputSize == 0) {
        OP_LOGE(context->GetNodeName(), "Input shape has 0 elements");
        return ge::GRAPH_FAILED;
    }
    // 计算输出元素个数：每8个输入元素的符号位打包为1个字节
    // const size_t outputSize = (inputSize + PACKING_FACTOR - 1) / PACKING_FACTOR;
    size_t outputSize = (inputSize + PACKING_FACTOR - 1) / PACKING_FACTOR * PACKING_FACTOR;

    // 设置输出形状为一维
    std::vector<int64_t> outputShapeVec = {static_cast<int64_t>(outputSize)};
    gert::Shape outputShape;
    outputShape.SetDimNum(1);
    outputShape.SetDim(0, static_cast<int64_t>(outputSize));
    *yShape = outputShape;
    OP_LOGD(context->GetNodeName(), "Input shape size: %zu, Output shape size: %zu", inputSize, outputSize);
    OP_LOGD(context->GetNodeName(), "End to do InferShapeSignBitsPack");
    return GRAPH_SUCCESS;
}

IMPL_OP_INFERSHAPE(SignBitsPack).InferShape(InferShapeSignBitsPack);
} // namespace ops