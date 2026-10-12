/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

/*!
 * \file infershape_broadcast_util.cpp
 * \brief
 */

#include "op_common/op_host/util/shape_util.h"
#include "op_common/log/log.h"
#include "op_common/op_host/infershape_broadcast_util.h"

namespace Ops {
namespace Base {
using namespace ge;

static bool BroadcastDim(int64_t& dim1, const int64_t dim2)
{
    /* column is dim1, row is dim2, matrix value is broadcast(dim1, dim2)
    dim -1  0  1  d2
    -1  -1  0 -1  d2
    0    0  0  0  E
    1   -1  0  1  d2
    d1  d1  E  d1 E
    */

    // no need to broadcast
    if (dim1 == dim2) {
        return true;
    }

    if ((dim1 != 1) && (dim2 != 1)) {
        // dynamic shape infershape
        if ((dim1 == -1) || (dim2 == -1)) {
            dim1 = (dim1 == -1) ? dim2 : dim1;
            return true;
        }

        OP_LOGE("BroadcastDim", "%ld and %ld cannot broadcast!", dim1, dim2);
        return false;
    }

    // static shape infershape
    dim1 = (dim1 == 1) ? dim2 : dim1;

    return true;
}

/*
 * @brief: broadcast new shape to output shape
 * @param [in] shape: const gert::Shape*, new shape to broadcast
 * @param [in/out] shapeOutput: gert::Shape*, output shape
 * @return succeed or not
 */
static bool BroadcastShapeToOutShape(const gert::Shape* shape, gert::Shape* shapeOutput)
{
    OP_LOGD("BroadcastShapeToOutShape", "start broadcast %s to %s!", ToString(*shape).c_str(),
            ToString(*shapeOutput).c_str());

    if (IsUnknownRank(*shape) || IsUnknownRank(*shapeOutput)) {
        OP_LOGD("BroadcastShapeToOutShape", "the input shape is [-2], set output shape is [-2]!");
        SetUnknownRank(*shapeOutput);
        return true;
    }

    size_t shapeLen = shape->GetDimNum();
    size_t shapeYLen = shapeOutput->GetDimNum();
    if (shapeLen > shapeYLen) {
        shapeOutput->SetDimNum(shapeLen);
        size_t lenSub = shapeLen - shapeYLen;
        for (size_t i = shapeYLen; i > 0; i--) {
            int64_t dim1 = shape->GetDim(lenSub + i - 1);
            int64_t dim2 = shapeOutput->GetDim(i - 1);
            OP_CHECK_IF(!BroadcastDim(dim1, dim2),
                        OP_LOGE("BroadcastShapeToOutShape", "%ld and %ld cannot broadcast!", dim1, dim2), return false);
            shapeOutput->SetDim(lenSub + i - 1, dim1);
        }
        for (size_t i = 0; i < lenSub; i++) {
            shapeOutput->SetDim(i, shape->GetDim(i));
        }
    } else {
        auto lenSub = shapeYLen - shapeLen;
        for (size_t i = 0; i < shapeLen; i++) {
            int64_t dim1 = shapeOutput->GetDim(lenSub + i);
            int64_t dim2 = shape->GetDim(i);
            OP_CHECK_IF(!BroadcastDim(dim1, dim2),
                        OP_LOGE("BroadcastShapeToOutShape", "%ld and %ld cannot broadcast!", dim1, dim2), return false);
            shapeOutput->SetDim(lenSub + i, dim1);
        }
    }
    return true;
}

bool BroadcastShape(const gert::Shape* in1Shape, const gert::Shape* in2Shape, gert::Shape* outShape)
{
    *outShape = *in1Shape;

    OP_CHECK_IF(!BroadcastShapeToOutShape(in2Shape, outShape),
                OP_LOGE("BroadcastShape", "shape %s and %s cannot broadcast!", ToString(*in2Shape).c_str(),
                        ToString(*in1Shape).c_str()),
                return false);
    return true;
}

bool BroadcastShape(const std::vector<const gert::Shape*>& inShapes, gert::Shape* outShape)
{
    size_t size = inShapes.size();
    OP_CHECK_IF(size == 0, OP_LOGE("BroadcastShape", "inShapes is empty!"), return false);
    *outShape = *inShapes[0];

    for (size_t i = 1UL; i < size; i++) {
        OP_CHECK_IF(!BroadcastShapeToOutShape(inShapes[i], outShape),
                    OP_LOGE("BroadcastShape", "intput shapes %s cannot broadcast!", ToString(inShapes).c_str()),
                    return false);
    }

    return true;
}

ge::graphStatus InferShape4Broadcast(gert::InferShapeContext* context)
{
    auto inShape1 = context->GetInputShape(0);
    OP_CHECK_NULL_WITH_CONTEXT(context, inShape1);
    auto inShape2 = context->GetInputShape(1);
    OP_CHECK_NULL_WITH_CONTEXT(context, inShape2);
    auto outShape = context->GetOutputShape(0);
    OP_CHECK_NULL_WITH_CONTEXT(context, outShape);

    OP_CHECK_IF(!BroadcastShape(inShape1, inShape2, outShape),
                OP_LOGE(context->GetNodeName(), "InferShape4Broadcast shape %s and %s cannot broadcast!",
                        ToString(*inShape2).c_str(), ToString(*inShape1).c_str()),
                return ge::GRAPH_FAILED);

    return ge::GRAPH_SUCCESS;
}

ge::graphStatus InferShape4Broadcast(gert::InferShapeContext* context, size_t inputNum)
{
    if (inputNum == 0) {
        OP_LOGE(context, "InferShape4Broadcast input num:%zu is invalid!", inputNum);
        return ge::GRAPH_FAILED;
    }
    std::vector<const gert::Shape*> inShapes(inputNum);
    for (size_t i = 0; i < inputNum; i++) {
        auto inShape = context->GetInputShape(i);
        OP_CHECK_NULL_WITH_CONTEXT(context, inShape);
        inShapes[i] = inShape;
    }
    auto outShape = context->GetOutputShape(0);
    OP_CHECK_NULL_WITH_CONTEXT(context, outShape);

    OP_CHECK_IF(!BroadcastShape(inShapes, outShape), OP_LOGE(context, "BroadcastShape failed!"),
                return ge::GRAPH_FAILED);

    return ge::GRAPH_SUCCESS;
}
} // namespace Base
} // namespace Ops
