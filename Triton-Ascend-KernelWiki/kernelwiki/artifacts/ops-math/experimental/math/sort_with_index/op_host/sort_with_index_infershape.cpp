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
 * \file sort_with_index_infershape.cpp
 * \brief Infer shape implementation for SortWithIndex operator.
 *        y.shape = x.shape; sorted_index.shape = index.shape;
 *        static shape: x.shape must == index.shape (else shape_mismatch).
 */

#include "register/op_impl_registry.h"
#include "util/shape_util.h"
#include "log/log.h"

using namespace ge;
namespace ops {
static constexpr int INPUT_NODE_NUM = 2;
static constexpr int OUTPUT_NODE_NUM = 2;
static constexpr int X_IDX = 0;
static constexpr int INDEX_IDX = 1;
static constexpr int Y1_IDX = 0;
static constexpr int Y2_IDX = 1;

ge::graphStatus InferShape4SortWithIndex(gert::InferShapeContext* context)
{
    OP_LOGD(context->GetNodeName(), "Begin to do InferShape4SortWithIndex");

    OP_CHECK_IF(
        (context->GetComputeNodeInputNum() != INPUT_NODE_NUM || context->GetComputeNodeOutputNum() != OUTPUT_NODE_NUM),
        OP_LOGE(context->GetNodeName(), "Get input or output num failed, infershape failed."),
        return GRAPH_FAILED);

    const gert::Shape* xShape = context->GetInputShape(X_IDX);
    OP_CHECK_NULL_WITH_CONTEXT(context, xShape);
    const gert::Shape* indexShape = context->GetInputShape(INDEX_IDX);
    OP_CHECK_NULL_WITH_CONTEXT(context, indexShape);

    bool isDynamicShape = (Ops::Base::IsUnknownRank(*xShape) || Ops::Base::IsUnknownRank(*indexShape) ||
                           Ops::Base::IsUnknownShape(*xShape) || Ops::Base::IsUnknownShape(*indexShape));

    OP_LOGD(context->GetNodeName(), "isDynamicShape=[%d]", isDynamicShape);
    if (!isDynamicShape) {
        OP_CHECK_IF((*xShape != *indexShape),
                    OP_LOGE(context->GetNodeName(), "input[x] and input[index] shape is different, infershape failed."),
                    return GRAPH_FAILED);
    }

    gert::Shape* y1Shape = context->GetOutputShape(Y1_IDX);
    OP_CHECK_NULL_WITH_CONTEXT(context, y1Shape);
    gert::Shape* y2Shape = context->GetOutputShape(Y2_IDX);
    OP_CHECK_NULL_WITH_CONTEXT(context, y2Shape);

    *y1Shape = *xShape;
    *y2Shape = *indexShape;

    OP_LOGD(context->GetNodeName(), "End to do InferShape4SortWithIndex");
    return ge::GRAPH_SUCCESS;
}

IMPL_OP_INFERSHAPE(SortWithIndex).InferShape(InferShape4SortWithIndex);
}  // namespace ops
