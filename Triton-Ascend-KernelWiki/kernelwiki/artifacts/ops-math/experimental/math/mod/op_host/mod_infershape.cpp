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
 * \file mod_infer.cpp
 * \brief
 */
#include "register/op_impl_registry.h"
#include "log/log.h"

using namespace ge;

namespace ops {
static constexpr int64_t IDX_0 = 0;
static constexpr int64_t IDX_1 = 1;

static bool CanBroadcastOtherToSelf(const gert::Shape* selfShape, const gert::Shape* otherShape)
{
    const int64_t selfDimNum = selfShape->GetDimNum();
    const int64_t otherDimNum = otherShape->GetDimNum();
    for (int64_t i = 0; i < otherDimNum; ++i) {
        const int64_t otherIdx = otherDimNum - 1 - i;
        const int64_t selfIdx = selfDimNum - 1 - i;
        const int64_t otherDim = otherShape->GetDim(otherIdx);
        if (selfIdx < 0) {
            if (otherDim != 1) {
                return false;
            }
            continue;
        }
        const int64_t selfDim = selfShape->GetDim(selfIdx);
        if (otherDim != 1 && otherDim != selfDim) {
            return false;
        }
    }
    return true;
}

static ge::graphStatus InferShape4Mod(gert::InferShapeContext* context)
{
    OP_LOGD(context->GetNodeName(), "Begin to do InferShape4Mod");

    // get input shapes
    const gert::Shape* xShape = context->GetInputShape(IDX_0);
    OP_CHECK_NULL_WITH_CONTEXT(context, xShape);
    const gert::Shape* otherShape = context->GetInputShape(IDX_1);
    OP_CHECK_NULL_WITH_CONTEXT(context, otherShape);
    OP_CHECK_IF(
        !CanBroadcastOtherToSelf(xShape, otherShape), OP_LOGE(context, "other shape can not broadcast to self shape."),
        return GRAPH_FAILED);

    // get output shapes
    gert::Shape* yShape = context->GetOutputShape(IDX_0);
    OP_CHECK_NULL_WITH_CONTEXT(context, yShape);

    // 填充输出shape大小
    *yShape = *xShape;
    OP_LOGD(context->GetNodeName(), "End to do InferShape4Mod");
    return GRAPH_SUCCESS;
}

IMPL_OP_INFERSHAPE(Mod).InferShape(InferShape4Mod);
} // namespace ops
