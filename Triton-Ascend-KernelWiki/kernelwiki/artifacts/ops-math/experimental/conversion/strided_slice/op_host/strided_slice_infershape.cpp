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
 * This program is free software: you can redistribute it and/or modify it.
 * Licensed under the CANN Open Software License Agreement Version 2.0 (the "License").
 * You may not use this file except in compliance with the License.
 * See the LICENSE file at the root of the repository for the full text of the License.
 *
 * THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTIES OF ANY KIND, EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
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
    auto xShapeSize = xShape->GetDimNum();
    uint32_t start1 = 0;
    uint32_t start2 = 0;
    uint32_t end1 = 0;
    uint32_t end2 = 0;
    uint32_t stride1 = 0;
    uint32_t stride2 = 0;
    auto attrs = context->GetAttrs();
    if (attrs) {
        const int64_t* start1Ptr = attrs->GetInt(0);
        if (start1Ptr) {
            start1 = static_cast<uint32_t>(*start1Ptr);
        }
        if (xShapeSize == 1) {
            start1 = static_cast<uint32_t>(0);
        }
        const int64_t* start2Ptr = attrs->GetInt(1);
        if (start2Ptr) {
            start2 = static_cast<uint32_t>(*start2Ptr);
        }
        const int64_t* end1Ptr = attrs->GetInt(2);
        if (end1Ptr) {
            end1 = static_cast<uint32_t>(*end1Ptr);
        }
        if (xShapeSize == 1) {
            end1 = static_cast<uint32_t>(1);
        }
        const int64_t* end2Ptr = attrs->GetInt(3);
        if (end2Ptr) {
            end2 = static_cast<uint32_t>(*end2Ptr);
        }
        const int64_t* stride1Ptr = attrs->GetInt(4);
        if (stride1Ptr) {
            stride1 = static_cast<uint32_t>(*stride1Ptr);
        }
        if (xShapeSize == 1) {
            stride1 = static_cast<uint32_t>(1);
        }
        const int64_t* stride2Ptr = attrs->GetInt(5);
        if (stride2Ptr) {
            stride2 = static_cast<uint32_t>(*stride2Ptr);
        }
    }
    OP_LOGD(context->GetNodeName(), "Begin to do InferShapeStridedSlice");
    uint32_t yRows;
    uint32_t yCols = (end2 - start2 + stride2 - 1) / stride2;
    OP_CHECK_NULL_WITH_CONTEXT(context, xShape);
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