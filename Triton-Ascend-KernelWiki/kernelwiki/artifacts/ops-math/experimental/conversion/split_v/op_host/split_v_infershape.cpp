/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include <cmath>
#include "register/op_impl_registry.h"
#include "log/log.h"
#include "op_host/util/shape_util.h"
#include "op_api/op_util.h"
#include "util/const_util.h"

using namespace ge;
using namespace Ops::Base;

namespace ops {
static graphStatus UpdateDynamicShape(gert::InferShapeContext* context, const gert::Shape* xShape,
                                      const int64_t numSplit)
{
    for (int64_t i = 0; i < numSplit; i++) {
        gert::Shape* outputShape = context->GetOutputShape(i);
        OP_CHECK_NULL_WITH_CONTEXT(context, outputShape);
        *outputShape = *xShape;
    }
    return GRAPH_SUCCESS;
}

static graphStatus UpdateAllUnknownDims(gert::InferShapeContext* context, const int64_t numSplit, const int64_t rank)
{
    for (int64_t i = 0; i < numSplit; i++) {
        gert::Shape* outputShape = context->GetOutputShape(i);
        OP_CHECK_NULL_WITH_CONTEXT(context, outputShape);
        SetUnknownShape(rank, *outputShape);
    }
    return GRAPH_SUCCESS;
}

static graphStatus CheckSplitVParams(const gert::InferShapeContext* context, const gert::Shape* xShape,
                                     const int64_t splitDim, const int64_t numSplit, const int64_t sizeSplitsSize)
{
    OP_CHECK_IF(numSplit <= 0,
                OP_LOGE(context->GetNodeName(), "%s",
                        ConcatString("num_split must be greater than 0, but it's ", numSplit).c_str()),
                return GRAPH_FAILED);

    const int64_t rank = xShape->GetDimNum();
    OP_CHECK_IF(!IsDimValid(rank, splitDim),
                OP_LOGE(context->GetNodeName(), "%s", GenInvalidDimMsg("split_dim", rank, splitDim).c_str()),
                return GRAPH_FAILED);

    OP_CHECK_IF(sizeSplitsSize != numSplit,
                OP_LOGE(context->GetNodeName(), "%s",
                        ConcatString("the size of size_splits must be equal to num_split. size_splits_size is ",
                                     sizeSplitsSize, " num_split is ", numSplit)
                            .c_str()),
                return GRAPH_FAILED);

    return GRAPH_SUCCESS;
}

template <typename T>
static graphStatus CalcSplitVOut(gert::InferShapeContext* context, const gert::Tensor* sizeSplitsTensor)
{
    const gert::Shape* xShape = context->GetInputShape(0);
    OP_CHECK_NULL_WITH_CONTEXT(context, xShape);

    const auto* attrs = context->GetAttrs();
    OP_CHECK_NULL_WITH_CONTEXT(context, attrs);

    const int64_t* numSplitAttr = attrs->GetAttrPointer<int64_t>(0);
    OP_CHECK_NULL_WITH_CONTEXT(context, numSplitAttr);
    const int64_t numSplit = *numSplitAttr;
    OP_CHECK_IF(numSplit <= 0, OP_LOGE(context->GetNodeName(), "num_split must be greater than 0"),
                return GRAPH_FAILED);

    OP_CHECK_IF(IsUnknownRank(*xShape),
                OP_LOGD(context->GetNodeName(), "input x is unknown rank, will set all output the same as input."),
                return UpdateDynamicShape(context, xShape, numSplit));

    int64_t splitDim = 0;
    static constexpr int64_t INPUT_SPLIT_DIM_INDEX = 2;
    if (!GetConstInt(context, INPUT_SPLIT_DIM_INDEX, splitDim)) {
        OP_LOGD(context->GetNodeName(), "get split_dim unsuccessful, will set output to -1.");
        return UpdateAllUnknownDims(context, numSplit, xShape->GetDimNum());
    }

    const int64_t sizeSplitsSize = static_cast<int64_t>(sizeSplitsTensor->GetShapeSize());
    OP_CHECK_IF(CheckSplitVParams(context, xShape, splitDim, numSplit, sizeSplitsSize) == GRAPH_FAILED,
                OP_LOGE(context->GetNodeName(), "check split params failed"), return GRAPH_FAILED);

    splitDim = splitDim < 0 ? splitDim + xShape->GetDimNum() : splitDim;
    const T* splitSizes = sizeSplitsTensor->GetData<T>();
    OP_CHECK_NULL_WITH_CONTEXT(context, splitSizes);
    int64_t dynamicValueIndex = -1;
    int64_t splitSizeSum = 0;
    int64_t dynamicValueCount = 0;

    for (int64_t i = 0; i < sizeSplitsSize; i++) {
        if (splitSizes[i] == -1) {
            dynamicValueCount++;
            OP_CHECK_IF(dynamicValueCount > 1,
                        OP_LOGE(context->GetNodeName(), "value of split_size can only have one -1"),
                        return GRAPH_FAILED);
            dynamicValueIndex = i;
        } else {
            splitSizeSum += splitSizes[i];
        }
    }

    for (int64_t i = 0; i < numSplit; i++) {
        gert::Shape* outputShape = context->GetOutputShape(i);
        OP_CHECK_NULL_WITH_CONTEXT(context, outputShape);
        *outputShape = *xShape;
        outputShape->SetDim(splitDim, splitSizes[i]);
    }

    if (dynamicValueIndex != -1) {
        gert::Shape* outputShape = context->GetOutputShape(dynamicValueIndex);
        OP_CHECK_NULL_WITH_CONTEXT(context, outputShape);
        if (xShape->GetDim(splitDim) == -1) {
            outputShape->SetDim(splitDim, -1);
        } else {
            outputShape->SetDim(splitDim, xShape->GetDim(splitDim) - splitSizeSum);
        }
    }

    return GRAPH_SUCCESS;
}

static graphStatus InferShape4SplitV(gert::InferShapeContext* context)
{
    OP_CHECK_IF(context == nullptr, OP_LOGE(context, "context is nullptr"), return GRAPH_FAILED);
    const gert::Tensor* sizeSplitsTensor = context->GetInputTensor(1);
    OP_CHECK_NULL_WITH_CONTEXT(context, sizeSplitsTensor);

    const DataType sizeSplitsDtype = sizeSplitsTensor->GetDataType();
    if (sizeSplitsDtype == DT_INT32) {
        OP_CHECK_IF(CalcSplitVOut<int32_t>(context, sizeSplitsTensor) == GRAPH_FAILED,
                    OP_LOGE(context->GetNodeName(), "Failed to calculate the output of split_v"), return GRAPH_FAILED);
    } else if (sizeSplitsDtype == DT_INT64) {
        OP_CHECK_IF(CalcSplitVOut<int64_t>(context, sizeSplitsTensor) == GRAPH_FAILED,
                    OP_LOGE(context->GetNodeName(), "Failed to calculate the output of split_v"), return GRAPH_FAILED);
    } else {
        OP_LOGE(context->GetNodeName(), "size_splits only supports int32 or int64");
        return GRAPH_FAILED;
    }

    return GRAPH_SUCCESS;
}
IMPL_OP_INFERSHAPE(SplitV).InferShape(InferShape4SplitV).InputsDataDependency({1, 2});

} // namespace ops
