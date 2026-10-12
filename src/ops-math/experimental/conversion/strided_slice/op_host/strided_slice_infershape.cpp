/**
 * This file is part of the OpenBOAT project at Harbin Institute of Technology (HIT)
 * and is contributed to the CANN Open Software.
 *
 * Copyright (c) 2025 AISS Group, Harbin Institute of Technology (HIT).
 * All Rights Reserved.
 *
 * Authors (accounts):
 * - Liu Jun <@kbryantttt>
 * - Su Tonghua <@sutonghua>
 *
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */
/*!
 * \file strided_slice_infer.cpp
 * \brief
 */
#include "register/op_impl_registry.h"
#include "log/log.h"
using namespace ge;
namespace ops {
static constexpr int64_t IDX_0 = 0;
static ge::graphStatus InferShapeStridedSlice(gert::InferShapeContext* context)
{
    const gert::Shape* xShape = context->GetInputShape(IDX_0);
    OP_CHECK_NULL_WITH_CONTEXT(context, xShape);
    auto xShapeSize = xShape->GetDimNum();
    int64_t start1 = 0;
    int64_t start2 = 0;
    int64_t end1 = 0;
    int64_t end2 = 0;
    int64_t stride1 = 0;
    int64_t stride2 = 0;
    auto attrs = context->GetAttrs();
    if (attrs) {
        const int64_t* start1Ptr = attrs->GetInt(0);
        if (start1Ptr) {
            start1 = *start1Ptr;
        }
        const int64_t* start2Ptr = attrs->GetInt(1);
        if (start2Ptr) {
            start2 = *start2Ptr;
        }
        const int64_t* end1Ptr = attrs->GetInt(2);
        if (end1Ptr) {
            end1 = *end1Ptr;
        }
        const int64_t* end2Ptr = attrs->GetInt(3);
        if (end2Ptr) {
            end2 = *end2Ptr;
        }
        const int64_t* stride1Ptr = attrs->GetInt(4);
        if (stride1Ptr) {
            stride1 = *stride1Ptr;
        }
        const int64_t* stride2Ptr = attrs->GetInt(5);
        if (stride2Ptr) {
            stride2 = *stride2Ptr;
        }
    }
    if (xShapeSize == 1) {
        start1 = 0;
        end1 = 1;
        stride1 = 1;
    }
    // 校验切片控制量: 拒绝负值/零步长/空切片/越界, 避免 int64->uint32 强转回绕导致的越界读写
    auto CheckSlice = [&](int64_t axisSize, int64_t start, int64_t end, int64_t stride, const char* axisName) -> bool {
        if (start < 0 || end < 0 || stride <= 0) {
            OP_LOGE(context->GetNodeName(),
                    "invalid %s slice param: start=%ld end=%ld stride=%ld, "
                    "require start/end >= 0 and stride > 0",
                    axisName, start, end, stride);
            return false;
        }
        if (start >= end || end > axisSize) {
            OP_LOGE(context->GetNodeName(),
                    "invalid %s slice range: start=%ld end=%ld axisSize=%ld, "
                    "require 0 <= start < end <= axisSize",
                    axisName, start, end, axisSize);
            return false;
        }
        return true;
    };
    if (xShapeSize >= 2) {
        int64_t axis0Size = xShape->GetDim(0);
        int64_t axis1Size = xShape->GetDim(1);
        if (!CheckSlice(axis0Size, start1, end1, stride1, "axis0") ||
            !CheckSlice(axis1Size, start2, end2, stride2, "axis1")) {
            return ge::GRAPH_FAILED;
        }
    } else if (xShapeSize == 1) {
        int64_t axis0Size = xShape->GetDim(0);
        if (!CheckSlice(axis0Size, start2, end2, stride2, "axis0")) {
            return ge::GRAPH_FAILED;
        }
    }
    OP_LOGD(context->GetNodeName(), "Begin to do InferShapeStridedSlice");
    uint32_t yRows;
    uint32_t yCols = (end2 - start2 + stride2 - 1) / stride2;
    gert::Shape* yShape = context->GetOutputShape(IDX_0);
    OP_CHECK_NULL_WITH_CONTEXT(context, yShape);
    // 填充输出shape大小

    yShape->SetDimNum(xShapeSize);
    if (xShapeSize == 1) {
        yShape->SetDim(0, yCols);
    } else {
        yRows = (end1 - start1 + stride1 - 1) / stride1;
        yShape->SetDim(0, yRows);
        yShape->SetDim(1, yCols);
    }
    OP_LOGD(context->GetNodeName(), "End to do InferShapeStridedSlice");
    return GRAPH_SUCCESS;
}
IMPL_OP_INFERSHAPE(StridedSlice).InferShape(InferShapeStridedSlice);
} // namespace ops
