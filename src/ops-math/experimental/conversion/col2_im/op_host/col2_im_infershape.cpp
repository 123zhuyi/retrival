/**
 * This file is part of the OpenBOAT project at Harbin Institute of Technology (HIT)
 * and is contributed to the CANN Open Software.
 *
 * Copyright (c) 2025 AISS Group, Harbin Institute of Technology (HIT).
 * All Rights Reserved.
 *
 * Authors (accounts):
 * - Qiu Zhuang <@qiu-zhuang>
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
 * \file col2_im_infer.cpp
 * \brief
 */
#include "register/op_impl_registry.h"
#include "log/log.h"

using namespace ge;

namespace ops {
static constexpr int64_t IDX_0 = 0;

// 常量定义
static constexpr int32_t REQUIRED_INPUT_DIMS = 3;
static constexpr int32_t OUTPUT_DIMS = 4;
static constexpr int32_t INPUT_TENSOR_INDEX = 0;
static constexpr int32_t OUTPUT_TENSOR_INDEX = 0;

// 属性索引常量
static constexpr int32_t ATTR_HEIGHT_INDEX = 0;
static constexpr int32_t ATTR_WIDTH_INDEX = 1;
static constexpr int32_t ATTR_KERNEL_H_INDEX = 2;
static constexpr int32_t ATTR_KERNEL_W_INDEX = 3;

// 维度索引常量
static constexpr int32_t BATCH_DIM_INDEX = 0;
static constexpr int32_t CHANNEL_DIM_INDEX = 1;
static constexpr int32_t LENGTH_DIM_INDEX = 2;

// 输出维度索引常量
static constexpr int32_t OUTPUT_BATCH_DIM_INDEX = 0;
static constexpr int32_t OUTPUT_CHANNEL_DIM_INDEX = 1;
static constexpr int32_t OUTPUT_HEIGHT_DIM_INDEX = 2;
static constexpr int32_t OUTPUT_WIDTH_DIM_INDEX = 3;

// 默认值常量
static constexpr int32_t DEFAULT_KERNEL_H = 2;
static constexpr int32_t DEFAULT_KERNEL_W = 2;
static constexpr int32_t ZERO_VALUE = 0;
static constexpr int32_t POSITIVE_THRESHOLD = 0;

static ge::graphStatus InferShapeCol2Im(gert::InferShapeContext* context)
{
    OP_LOGD(context->GetNodeName(), "Begin to do InferShapeCol2Im");

    if (context == nullptr) {
        OP_LOGE("InferShapeCol2Im", "Failed: context is nullptr");
        return ge::GRAPH_FAILED;
    }

    // get input shapes
    const gert::Shape* InputShape = context->GetInputShape(INPUT_TENSOR_INDEX);
    if (InputShape == nullptr) {
        OP_LOGE(context->GetNodeName(), "Failed: input shape is nullptr");
        return ge::GRAPH_FAILED;
    }

    // get output shapes
    gert::Shape* OutputShape = context->GetOutputShape(OUTPUT_TENSOR_INDEX);
    if (OutputShape == nullptr) {
        OP_LOGE(context->GetNodeName(), "Failed: output shape is nullptr");
        return ge::GRAPH_FAILED;
    }

    // 输入: [N, C*kH*kW, L]
    if (InputShape->GetDimNum() != REQUIRED_INPUT_DIMS) {
        OP_LOGE(
            context->GetNodeName(), "Failed: input shape must be %dD, but got %zuD", REQUIRED_INPUT_DIMS,
            InputShape->GetDimNum());
        return ge::GRAPH_FAILED;
    }

    int32_t N = InputShape->GetDim(BATCH_DIM_INDEX);
    int32_t input_channels = InputShape->GetDim(CHANNEL_DIM_INDEX);

    // 获取输出尺寸和卷积参数
    int32_t H = ZERO_VALUE;
    int32_t W = ZERO_VALUE;
    int32_t kernel_h = DEFAULT_KERNEL_H;
    int32_t kernel_w = DEFAULT_KERNEL_W;

    auto attrPtr = context->GetAttrs();
    if (attrPtr != nullptr) {
        if (attrPtr->GetInt(ATTR_HEIGHT_INDEX) != nullptr) {
            H = static_cast<int32_t>(*(attrPtr->GetInt(ATTR_HEIGHT_INDEX)));
        }
        if (attrPtr->GetInt(ATTR_WIDTH_INDEX) != nullptr) {
            W = static_cast<int32_t>(*(attrPtr->GetInt(ATTR_WIDTH_INDEX)));
        }
        if (attrPtr->GetInt(ATTR_KERNEL_H_INDEX) != nullptr) {
            kernel_h = static_cast<int32_t>(*(attrPtr->GetInt(ATTR_KERNEL_H_INDEX)));
        }
        if (attrPtr->GetInt(ATTR_KERNEL_W_INDEX) != nullptr) {
            kernel_w = static_cast<int32_t>(*(attrPtr->GetInt(ATTR_KERNEL_W_INDEX)));
        }
    }

    int32_t kernel_size = kernel_h * kernel_w;

    // 检查输入通道数是否能被kernel_size整除
    if (kernel_size == ZERO_VALUE) {
        OP_LOGE(
            context->GetNodeName(), "Failed: kernel size cannot be zero (kernel_h=%d, kernel_w=%d)", kernel_h,
            kernel_w);
        return ge::GRAPH_FAILED;
    }

    if (input_channels % kernel_size != ZERO_VALUE) {
        OP_LOGE(
            context->GetNodeName(), "Failed: input channels (%d) must be divisible by kernel size (%d)", input_channels,
            kernel_size);
        return ge::GRAPH_FAILED;
    }

    int32_t C = input_channels / kernel_size;

    // 检查输出尺寸是否有效
    if (H <= POSITIVE_THRESHOLD || W <= POSITIVE_THRESHOLD) {
        OP_LOGE(context->GetNodeName(), "Failed: output height (%d) and width (%d) must be positive", H, W);
        return ge::GRAPH_FAILED;
    }

    // 检查输入第三个维度是否与输出尺寸匹配
    int32_t L = InputShape->GetDim(LENGTH_DIM_INDEX);
    int32_t expected_L = H * W;
    if (L != expected_L) {
        OP_LOGE(
            context->GetNodeName(), "Failed: input third dimension (%d) must equal H*W (%d*%d=%d)", L, H, W,
            expected_L);
        return ge::GRAPH_FAILED;
    }

    // 输出: [N, C, H, W]
    std::vector<int64_t> outputDims(OUTPUT_DIMS);
    outputDims[OUTPUT_BATCH_DIM_INDEX] = N;
    outputDims[OUTPUT_CHANNEL_DIM_INDEX] = C;
    outputDims[OUTPUT_HEIGHT_DIM_INDEX] = H;
    outputDims[OUTPUT_WIDTH_DIM_INDEX] = W;

    OutputShape->SetDimNum(outputDims.size());
    for (size_t i = 0; i < outputDims.size(); ++i) {
        OutputShape->SetDim(i, outputDims[i]);
    }

    OP_LOGD(context->GetNodeName(), "End to do InferShapeCol2Im");
    return ge::GRAPH_SUCCESS;
}

IMPL_OP_INFERSHAPE(Col2Im).InferShape(InferShapeCol2Im);
} // namespace ops