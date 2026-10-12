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
 * \file split_d_infershape.cpp
 * \brief infershape func of SplitD
 */
#include <cmath>
#include "register/op_impl_registry.h"
#include "log/log.h"
#include "op_host/util/shape_util.h"
#include "op_api/op_util.h"
#include "util/const_util.h"
#include "util/math_util.h"

using namespace ge;
using namespace Ops::Base;

namespace ops {

static constexpr size_t SPLIT_D_IDX_IN_X = 0;
static constexpr size_t SPLIT_D_IDX_OUT_Y = 0;
static constexpr size_t SPLIT_D_IDX_ATTR_SPLIT_DIM = 0;
static constexpr size_t SPLIT_D_IDX_ATTR_NUM_SPLIT = 1;

static graphStatus UpdateDynamicShape(gert::InferShapeContext* context, const gert::Shape* xShape, int64_t numSplit)
{
    for (int64_t i = 0; i < numSplit; i++) {
        gert::Shape* outShapeDynamic = context->GetOutputShape(i);
        OP_CHECK_NULL_WITH_CONTEXT(context, outShapeDynamic);
        *outShapeDynamic = *xShape;
    }
    return GRAPH_SUCCESS;
}

static graphStatus CheckSplitParams(gert::InferShapeContext* context, const gert::Shape* xShape, int64_t splitDim,
                                    int64_t numSplit)
{
    OP_CHECK_IF(numSplit <= 0,
                OP_LOGE(context->GetNodeName(), "%s",
                        ConcatString("num_split must be greater than 0, but it's ", numSplit).c_str()),
                return GRAPH_FAILED);

    int64_t xShapeDim = static_cast<int64_t>(xShape->GetDimNum());
    OP_CHECK_IF(!IsDimValid(xShapeDim, splitDim),
                OP_LOGE(context->GetNodeName(), "%s", GenInvalidDimMsg("split_dim", xShapeDim, splitDim).c_str()),
                return GRAPH_FAILED);

    OP_CHECK_IF((xShape->GetDim(splitDim) % numSplit != 0) && (xShape->GetDim(splitDim) != -1),
                OP_LOGE(context->GetNodeName(), "%s",
                        ConcatString("the split_dim dimension of x_shape must be divided by num_split.",
                                     " x_shape on split_dim is ", xShape->GetDim(splitDim), ", num_split is ", numSplit)
                            .c_str()),
                return GRAPH_FAILED);
    return GRAPH_SUCCESS;
}

static void CalOutShape(const gert::Shape* xShape, gert::Shape* outShape, int64_t numSplit, int64_t splitDim)
{
    int64_t outputDimSize = xShape->GetDim(splitDim) / numSplit;
    *outShape = *xShape;
    outShape->SetDim(splitDim, outputDimSize);
}

static graphStatus InferShape4SplitD(gert::InferShapeContext* context)
{
    OP_LOGD(context->GetNodeName(), "Begin to do InferShape4SplitD");
    const gert::Shape* xShape = context->GetInputShape(SPLIT_D_IDX_IN_X);
    OP_CHECK_NULL_WITH_CONTEXT(context, xShape);

    auto attrs = context->GetAttrs();
    OP_CHECK_NULL_WITH_CONTEXT(context, attrs);

    const int64_t* splitDimPtr = attrs->GetAttrPointer<int64_t>(SPLIT_D_IDX_ATTR_SPLIT_DIM);
    OP_CHECK_NULL_WITH_CONTEXT(context, splitDimPtr);
    int64_t splitDim = *splitDimPtr;

    const int64_t* numSplitPtr = attrs->GetAttrPointer<int64_t>(SPLIT_D_IDX_ATTR_NUM_SPLIT);
    OP_CHECK_NULL_WITH_CONTEXT(context, numSplitPtr);
    int64_t numSplit = *numSplitPtr;

    OP_CHECK_IF(IsUnknownRank(*xShape),
                OP_LOGD(context->GetNodeName(), "input x is unknown rank, will set all output the same as input."),
                return UpdateDynamicShape(context, xShape, numSplit));

    OP_CHECK_IF(CheckSplitParams(context, xShape, splitDim, numSplit) == GRAPH_FAILED,
                OP_LOGE(context->GetNodeName(), "check split params failed."), return GRAPH_FAILED);

    splitDim = splitDim < 0 ? splitDim + static_cast<int64_t>(xShape->GetDimNum()) : splitDim;
    if (xShape->GetDim(splitDim) == -1) {
        OP_LOGD(context->GetNodeName(), "the split dim is -1 input x, will set all output the same as input.");
        return UpdateDynamicShape(context, xShape, numSplit);
    }

    gert::Shape* outShape = context->GetOutputShape(SPLIT_D_IDX_OUT_Y);
    OP_CHECK_NULL_WITH_CONTEXT(context, outShape);
    CalOutShape(xShape, outShape, numSplit, splitDim);

    // update dynamic output
    for (int64_t i = 1; i < numSplit; i++) {
        gert::Shape* outShapeDynamic = context->GetOutputShape(i);
        OP_CHECK_NULL_WITH_CONTEXT(context, outShapeDynamic);
        *outShapeDynamic = *outShape;
    }

    OP_LOGD(context->GetNodeName(), "End to do InferShape4SplitD");
    return GRAPH_SUCCESS;
}

static graphStatus InferDataType4SplitD(gert::InferDataTypeContext* context)
{
    OP_LOGD(context->GetNodeName(), "Begin to do InferDataType4SplitD");
    auto xDtype = context->GetInputDataType(SPLIT_D_IDX_IN_X);
    context->SetOutputDataType(SPLIT_D_IDX_OUT_Y, xDtype);
    OP_LOGD(context->GetNodeName(), "End to do InferDataType4SplitD");
    return GRAPH_SUCCESS;
}

static graphStatus InferShapeRange4SplitD(gert::InferShapeRangeContext* context)
{
    OP_LOGD(context->GetNodeName(), "Begin to do InferShapeRange4SplitD");
    auto attrs = context->GetAttrs();
    OP_CHECK_NULL_WITH_CONTEXT(context, attrs);
    const int64_t* splitDimPtr = attrs->GetAttrPointer<int64_t>(SPLIT_D_IDX_ATTR_SPLIT_DIM);
    OP_CHECK_NULL_WITH_CONTEXT(context, splitDimPtr);
    int64_t splitDim = *splitDimPtr;
    const int64_t* numSplitPtr = attrs->GetAttrPointer<int64_t>(SPLIT_D_IDX_ATTR_NUM_SPLIT);
    OP_CHECK_NULL_WITH_CONTEXT(context, numSplitPtr);
    int64_t numSplit = *numSplitPtr;

    const auto* xRange = context->GetInputShapeRange(SPLIT_D_IDX_IN_X);
    OP_CHECK_NULL_WITH_CONTEXT(context, xRange);
    const auto* xRangeMax = xRange->GetMax();
    const auto* xRangeMin = xRange->GetMin();
    OP_CHECK_NULL_WITH_CONTEXT(context, xRangeMax);
    OP_CHECK_NULL_WITH_CONTEXT(context, xRangeMin);

    int64_t xDim = static_cast<int64_t>(xRangeMax->GetDimNum());
    if (splitDim < 0) {
        splitDim += xDim;
    }

    for (int64_t outIdx = 0; outIdx < numSplit; outIdx++) {
        auto* yRange = context->GetOutputShapeRange(static_cast<size_t>(outIdx));
        OP_CHECK_NULL_WITH_CONTEXT(context, yRange);
        auto* yRangeMax = yRange->GetMax();
        auto* yRangeMin = yRange->GetMin();
        OP_CHECK_NULL_WITH_CONTEXT(context, yRangeMax);
        OP_CHECK_NULL_WITH_CONTEXT(context, yRangeMin);
        yRangeMax->SetDimNum(static_cast<size_t>(xDim));
        yRangeMin->SetDimNum(static_cast<size_t>(xDim));
        for (int64_t i = 0; i < xDim; ++i) {
            if (i == splitDim) {
                if (xRangeMax->GetDim(static_cast<size_t>(i)) == -1) {
                    yRangeMax->SetDim(static_cast<size_t>(i), -1);
                } else {
                    yRangeMax->SetDim(static_cast<size_t>(i),
                                      Ops::Base::CeilDiv(xRangeMax->GetDim(static_cast<size_t>(i)), numSplit));
                }
                if (xRangeMin->GetDim(static_cast<size_t>(i)) == -1) {
                    yRangeMin->SetDim(static_cast<size_t>(i), -1);
                } else if (xRangeMin->GetDim(static_cast<size_t>(i)) == 1) {
                    yRangeMin->SetDim(static_cast<size_t>(i), 1);
                } else {
                    yRangeMin->SetDim(static_cast<size_t>(i), xRangeMin->GetDim(static_cast<size_t>(i)) / numSplit);
                }
            } else {
                yRangeMax->SetDim(static_cast<size_t>(i), xRangeMax->GetDim(static_cast<size_t>(i)));
                yRangeMin->SetDim(static_cast<size_t>(i), xRangeMin->GetDim(static_cast<size_t>(i)));
            }
        }
    }
    OP_LOGD(context->GetNodeName(), "End to do InferShapeRange4SplitD");
    return GRAPH_SUCCESS;
}

IMPL_OP_INFERSHAPE(SplitD)
    .InferShape(InferShape4SplitD)
    .InferDataType(InferDataType4SplitD)
    .InferShapeRange(InferShapeRange4SplitD);
} // namespace ops
