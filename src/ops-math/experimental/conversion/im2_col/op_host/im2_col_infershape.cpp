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
 * \file im2_col_infershape.cpp
 * \brief
 */
#include "register/op_impl_registry.h"
#include "log/log.h"

using namespace ge;

namespace ops {
// 常量定义，用于替换魔法数字
namespace {
constexpr int32_t DEFAULT_KERNEL_H = 2;
constexpr int32_t DEFAULT_KERNEL_W = 2;
constexpr int32_t DEFAULT_STRIDE_H = 1;
constexpr int32_t DEFAULT_STRIDE_W = 1;
constexpr int32_t DEFAULT_PAD_H = 0;
constexpr int32_t DEFAULT_PAD_W = 0;
constexpr int32_t DEFAULT_DILATION_H = 1;
constexpr int32_t DEFAULT_DILATION_W = 1;

constexpr int32_t MIN_DIM_NUM = 2;
constexpr int32_t TENSOR_DIM_NUM_2D = 2;
constexpr int32_t TENSOR_DIM_NUM_3D = 3;
constexpr int32_t TENSOR_DIM_NUM_4D = 4;

constexpr int32_t BATCH_DIM_INDEX = 0;
constexpr int32_t CHANNEL_DIM_INDEX = 1;
constexpr int32_t HEIGHT_DIM_INDEX_OFFSET = -2; // 倒数第二维
constexpr int32_t WIDTH_DIM_INDEX_OFFSET = -1;  // 最后一维

constexpr int32_t SINGLE_BATCH_VALUE = 1;
constexpr int32_t PADDING_MULTIPLIER = 2;
constexpr int32_t STRIDE_OFFSET = 1; // 公式中的 +1

constexpr size_t KERNEL_H_ATTR_INDEX = 0;
constexpr size_t KERNEL_W_ATTR_INDEX = 1;
constexpr size_t STRIDE_VAL_ATTR_INDEX = 2;
constexpr size_t PADDING_VAL_ATTR_INDEX = 3;

constexpr size_t OUTPUT_DIM_COUNT_3D = 3; // 输出形状维度数
} // namespace

static ge::graphStatus InferShapeIm2Col(gert::InferShapeContext* context)
{
    OP_LOGD(context->GetNodeName(), "Begin to do InferShapeIm2Col");

    // get input shapes
    const gert::Shape* InputShape = context->GetInputShape(0);
    if (InputShape == nullptr) {
        OP_LOGE(context->GetNodeName(), "InputShape is nullptr");
        return ge::GRAPH_FAILED;
    }

    // get output shapes
    gert::Shape* OutputShape = context->GetOutputShape(0);
    if (OutputShape == nullptr) {
        OP_LOGE(context->GetNodeName(), "OutputShape is nullptr");
        return ge::GRAPH_FAILED;
    }

    // 获取卷积参数（使用默认值）
    int32_t kernel_h = DEFAULT_KERNEL_H;
    int32_t kernel_w = DEFAULT_KERNEL_W;
    int32_t stride_h = DEFAULT_STRIDE_H;
    int32_t stride_w = DEFAULT_STRIDE_W;
    int32_t pad_h = DEFAULT_PAD_H;
    int32_t pad_w = DEFAULT_PAD_W;
    int32_t dilation_h = DEFAULT_DILATION_H;
    int32_t dilation_w = DEFAULT_DILATION_W;

    auto attrPtr = context->GetAttrs();
    if (attrPtr != nullptr) {
        // 获取属性值
        const int64_t* kernel_h_ptr = attrPtr->GetInt(0);
        const int64_t* kernel_w_ptr = attrPtr->GetInt(1);
        const int64_t* stride_h_ptr = attrPtr->GetInt(2);
        const int64_t* stride_w_ptr = attrPtr->GetInt(3);
        const int64_t* pad_h_ptr = attrPtr->GetInt(4);
        const int64_t* pad_w_ptr = attrPtr->GetInt(5);
        const int64_t* dilation_h_ptr = attrPtr->GetInt(6);
        const int64_t* dilation_w_ptr = attrPtr->GetInt(7);

        if (kernel_h_ptr != nullptr) {
            kernel_h = static_cast<int32_t>(*kernel_h_ptr);
        }
        if (kernel_w_ptr != nullptr) {
            kernel_w = static_cast<int32_t>(*kernel_w_ptr);
        }
        if (stride_h_ptr != nullptr) {
            stride_h = static_cast<int32_t>(*stride_h_ptr);
        }
        if (stride_w_ptr != nullptr) {
            stride_w = static_cast<int32_t>(*stride_w_ptr);
        }
        if (pad_h_ptr != nullptr) {
            pad_h = static_cast<int32_t>(*pad_h_ptr);
        }
        if (pad_w_ptr != nullptr) {
            pad_w = static_cast<int32_t>(*pad_w_ptr);
        }
        if (dilation_h_ptr != nullptr) {
            dilation_h = static_cast<int32_t>(*dilation_h_ptr);
        }
        if (dilation_w_ptr != nullptr) {
            dilation_w = static_cast<int32_t>(*dilation_w_ptr);
        }
    }

    // 解析输入形状
    auto dimNum = InputShape->GetDimNum();
    if (dimNum < MIN_DIM_NUM) {
        OP_LOGE(
            context->GetNodeName(), "Input dimension number %zu is less than minimum required %d", dimNum, MIN_DIM_NUM);
        return ge::GRAPH_FAILED;
    }

    // 获取输入张量的空间维度
    int32_t H = InputShape->GetDim(dimNum + HEIGHT_DIM_INDEX_OFFSET);
    int32_t W = InputShape->GetDim(dimNum + WIDTH_DIM_INDEX_OFFSET);

    // 获取通道数（对于2D输入，通道数为1）
    int32_t C = SINGLE_BATCH_VALUE;
    if (dimNum >= TENSOR_DIM_NUM_4D) {
        C = InputShape->GetDim(CHANNEL_DIM_INDEX);
    }

    // 计算输出特征图的空间尺寸
    int32_t out_H = (H + 2 * pad_h - dilation_h * (kernel_h - 1) - 1) / stride_h + 1;
    int32_t out_W = (W + 2 * pad_w - dilation_w * (kernel_w - 1) - 1) / stride_w + 1;
    int32_t L = out_H * out_W;

    // 计算输出通道数 = 输入通道数 × 卷积核高度 × 卷积核宽度
    int32_t output_channels = C * kernel_h * kernel_w;

    // 设置输出形状 [N, C*kH*kW, L]
    std::vector<int64_t> outputDims;

    if (dimNum == TENSOR_DIM_NUM_4D) {
        // 4D输入：[N, C, H, W] -> [N, C*kH*kW, L]
        outputDims = {
            InputShape->GetDim(BATCH_DIM_INDEX),   // N
            static_cast<int64_t>(output_channels), // C*kH*kW
            static_cast<int64_t>(L)                // L
        };
    } else if (dimNum == TENSOR_DIM_NUM_3D) {
        // 3D输入：[C, H, W] -> [1, C*kH*kW, L]
        outputDims = {
            InputShape->GetDim(BATCH_DIM_INDEX),   // 这里实际上取的是C，但作为批处理维度
            static_cast<int64_t>(output_channels), // C*kH*kW
            static_cast<int64_t>(L)                // L
        };
    } else { // dimNum == TENSOR_DIM_NUM_2D
        // 2D输入：[H, W] -> [1, 1*kH*kW, L]
        outputDims = {
            static_cast<int64_t>(SINGLE_BATCH_VALUE), // N=1
            static_cast<int64_t>(output_channels),    // 1*kH*kW
            static_cast<int64_t>(L)                   // L
        };
    }

    OutputShape->SetDimNum(OUTPUT_DIM_COUNT_3D);
    for (size_t i = 0; i < OUTPUT_DIM_COUNT_3D; ++i) {
        OutputShape->SetDim(i, outputDims[i]);
    }

    OP_LOGD(context->GetNodeName(), "End to do InferShapeIm2Col");
    return ge::GRAPH_SUCCESS;
}
IMPL_OP_INFERSHAPE(Im2Col).InferShape(InferShapeIm2Col);
} // namespace ops