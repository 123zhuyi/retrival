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
 * \file stride_add_infershape.cpp
 * \brief StrideAdd shape inference implementation
 */

#include "register/op_impl_registry.h"
#include "log/log.h"

using namespace ge;

namespace ops {
static constexpr int64_t IDX_0 = 0;
static constexpr int64_t IDX_1 = 1;

// NC1HWC0 物理 shape 维度索引
static constexpr size_t DIM_N = 0;
static constexpr size_t DIM_C1 = 1;
static constexpr size_t DIM_H = 2;
static constexpr size_t DIM_W = 3;
static constexpr size_t DIM_C0 = 4;
static constexpr size_t SHAPE_DIM_NUM = 5;

// 属性索引
static constexpr size_t ATTR_X1_C1_OFFSET_IDX = 0;
static constexpr size_t ATTR_X2_C1_OFFSET_IDX = 1;
static constexpr size_t ATTR_C1_LEN_IDX = 2;

static ge::graphStatus InferShapeStrideAdd(gert::InferShapeContext* context)
{
    OP_LOGD(context->GetNodeName(), "Begin to do InferShapeStrideAdd");

    // 获取输入 shape
    const gert::Shape* x1Shape = context->GetInputShape(IDX_0);
    OP_CHECK_NULL_WITH_CONTEXT(context, x1Shape);
    const gert::Shape* x2Shape = context->GetInputShape(IDX_1);
    OP_CHECK_NULL_WITH_CONTEXT(context, x2Shape);

    // 获取属性（指针判空+解引用模式）
    auto attrs = context->GetAttrs();
    OP_CHECK_NULL_WITH_CONTEXT(context, attrs);
    auto x1OffsetPtr = attrs->GetAttrPointer<int32_t>(ATTR_X1_C1_OFFSET_IDX);
    auto x2OffsetPtr = attrs->GetAttrPointer<int32_t>(ATTR_X2_C1_OFFSET_IDX);
    auto c1LenPtr = attrs->GetAttrPointer<int32_t>(ATTR_C1_LEN_IDX);
    OP_CHECK_NULL_WITH_CONTEXT(context, x1OffsetPtr);
    OP_CHECK_NULL_WITH_CONTEXT(context, x2OffsetPtr);
    OP_CHECK_NULL_WITH_CONTEXT(context, c1LenPtr);
    int32_t x1_c1_offset = *x1OffsetPtr;
    int32_t x2_c1_offset = *x2OffsetPtr;
    int32_t c1_len = *c1LenPtr;

    // 参数校验
    OP_CHECK_IF(x1_c1_offset < 0,
        OP_LOGE_FOR_INVALID_VALUE_WITH_REASON(context->GetNodeName(), "x1_c1_offset",
            std::to_string(x1_c1_offset).c_str(), "x1_c1_offset must be greater than or equal to 0"),
        return ge::GRAPH_FAILED);
    OP_CHECK_IF(x2_c1_offset < 0,
        OP_LOGE_FOR_INVALID_VALUE_WITH_REASON(context->GetNodeName(), "x2_c1_offset",
            std::to_string(x2_c1_offset).c_str(), "x2_c1_offset must be greater than or equal to 0"),
        return ge::GRAPH_FAILED);
    OP_CHECK_IF(c1_len <= 0,
        OP_LOGE_FOR_INVALID_VALUE_WITH_REASON(context->GetNodeName(), "c1_len",
            std::to_string(c1_len).c_str(), "c1_len must be greater than 0"),
        return ge::GRAPH_FAILED);
    OP_CHECK_IF(x1_c1_offset + c1_len > x1Shape->GetDim(DIM_C1),
        OP_LOGE_FOR_INVALID_VALUE_WITH_REASON(context->GetNodeName(), "x1_c1_offset",
            std::to_string(x1_c1_offset).c_str(), "x1_c1_offset + c1_len must not exceed x1's C1 dimension"),
        return ge::GRAPH_FAILED);
    OP_CHECK_IF(x2_c1_offset + c1_len > x2Shape->GetDim(DIM_C1),
        OP_LOGE_FOR_INVALID_VALUE_WITH_REASON(context->GetNodeName(), "x2_c1_offset",
            std::to_string(x2_c1_offset).c_str(), "x2_c1_offset + c1_len must not exceed x2's C1 dimension"),
        return ge::GRAPH_FAILED);

    // 输出 shape 推导: y = [N, c1_len, H, W, C0]
    // NC1HWC0 物理 shape 下 dim[1] = c1_len (C1 块数)
    gert::Shape* yShape = context->GetOutputShape(IDX_0);
    OP_CHECK_NULL_WITH_CONTEXT(context, yShape);
    yShape->SetDimNum(SHAPE_DIM_NUM);
    yShape->SetDim(DIM_N, x1Shape->GetDim(DIM_N));  // N
    yShape->SetDim(DIM_C1, c1_len);                  // c1_len (C1 块数)
    yShape->SetDim(DIM_H, x1Shape->GetDim(DIM_H));  // H
    yShape->SetDim(DIM_W, x1Shape->GetDim(DIM_W));  // W
    yShape->SetDim(DIM_C0, x1Shape->GetDim(DIM_C0));  // C0

    // 输出 dtype 由 def.cpp 的 DataType 映射自动推导（y.dtype = x1.dtype）
    // InferShape 中无需显式设置 dtype

    OP_LOGD(context->GetNodeName(), "End to do InferShapeStrideAdd");
    return GRAPH_SUCCESS;
}

IMPL_OP_INFERSHAPE(StrideAdd).InferShape(InferShapeStrideAdd);
} // namespace ops