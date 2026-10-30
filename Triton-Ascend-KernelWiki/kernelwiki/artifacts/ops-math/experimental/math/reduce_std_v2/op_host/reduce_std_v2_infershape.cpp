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
 * \file reduce_std_v2_infershape.cpp
 * \brief
 */
#include "op_host/infershape_reduce_util.h"
#include "register/op_impl_registry.h"
#include "util/shape_util.h"

using namespace ge;

namespace ops {
const size_t INPUT_INDEX_X = 0;
const size_t ATTR_INDEX_AXES = 0;
const size_t OUTPUT_INDEX_STD = 0;
const size_t OUTPUT_INDEX_MEAN = 1;
const size_t VAR_INDEX_ATTR_KEEPDIM = 2;

static ge::graphStatus GetReduceStdV2Shapes(gert::InferShapeContext* context, const gert::Shape*& inputShape,
    gert::Shape*& varShape, gert::Shape*& meanShape)
{
    inputShape = context->GetInputShape(INPUT_INDEX_X);
    OP_CHECK_NULL_WITH_CONTEXT(context, inputShape);
    varShape = context->GetOutputShape(OUTPUT_INDEX_STD);
    OP_CHECK_NULL_WITH_CONTEXT(context, varShape);
    meanShape = context->GetOutputShape(OUTPUT_INDEX_MEAN);
    OP_CHECK_NULL_WITH_CONTEXT(context, meanShape);
    return GRAPH_SUCCESS;
}

static ge::graphStatus GetReduceStdV2Attrs(gert::InferShapeContext* context, const gert::RuntimeAttrs*& attrs,
    const gert::ContinuousVector*& axesShape)
{
    attrs = context->GetAttrs();
    OP_CHECK_NULL_WITH_CONTEXT(context, attrs);
    axesShape = attrs->GetAttrPointer<gert::ContinuousVector>(ATTR_INDEX_AXES);
    OP_CHECK_NULL_WITH_CONTEXT(context, axesShape);
    return GRAPH_SUCCESS;
}

static std::vector<int64_t> GetReduceAxes(const gert::Shape* inputShape,
    const gert::ContinuousVector* axesShape, std::stringstream& strs)
{
    std::vector<int64_t> axes;
    int64_t inputDimNum = inputShape->GetDimNum();
    int64_t axesSize = axesShape->GetSize();
    if (axesSize == 0) {
        axes.resize(inputDimNum);
        for (int64_t i = 0; i < inputDimNum; i++) {
            axes[i] = i;
            strs << axes[i] << " ";
        }
    } else {
        auto axesData = static_cast<const int64_t*>(axesShape->GetData());
        axes.resize(axesSize);
        for (int64_t i = 0; i < axesSize; i++) {
            axes[i] = axesData[i];
            strs << axes[i] << " ";
        }
    }
    return axes;
}

static ge::graphStatus NormalizeReduceAxes(gert::InferShapeContext* context, const gert::Shape* inputShape,
    std::vector<int64_t>& axes)
{
    int64_t inputDimNum = inputShape->GetDimNum();
    for (auto& axis : axes) {
        if ((axis < -inputDimNum) || (axis >= inputDimNum)) {
            OP_LOGE(context->GetNodeName(), "dim value %ld is out of range [%ld, %ld).",
                axis, -inputDimNum, inputDimNum);
            return GRAPH_FAILED;
        }
        if (axis < 0) {
            axis += inputDimNum;
        }
    }
    return GRAPH_SUCCESS;
}

static ge::graphStatus InferReducedShape(const gert::Shape* inputShape, const std::vector<int64_t>& axes,
    bool keepDims, gert::Shape* varShape)
{
    const int64_t* axesData = axes.empty() ? nullptr : axes.data();
    if (keepDims) {
        return Ops::Base::ReduceDimsWithKeepDims<int64_t>(
            inputShape, axesData, static_cast<int32_t>(axes.size()), varShape);
    }
    return Ops::Base::ReduceDimsWithoutKeepDims<int64_t>(
        inputShape, axesData, static_cast<int32_t>(axes.size()), varShape);
}

static void LogInferShapeResult(gert::InferShapeContext* context, const gert::Shape* inputShape,
    const std::stringstream& strs, bool keepDims, const gert::Shape* varShape, const gert::Shape* meanShape)
{
    OP_LOGD(context->GetNodeName(),
        "inputShape:%s reduce axes:%s keepDims:%d, get infer varShape:%s meanShape:%s.",
        Ops::Base::ToString(*inputShape).c_str(), strs.str().c_str(), keepDims,
        Ops::Base::ToString(*varShape).c_str(), Ops::Base::ToString(*meanShape).c_str());
}

static ge::graphStatus InferShape4ReduceStdV2(gert::InferShapeContext* context)
{
    const gert::Shape* inputShape = nullptr;
    gert::Shape* varShape = nullptr;
    gert::Shape* meanShape = nullptr;
    if (GetReduceStdV2Shapes(context, inputShape, varShape, meanShape) != GRAPH_SUCCESS) {
        return GRAPH_FAILED;
    }

    const gert::RuntimeAttrs* attrs = nullptr;
    const gert::ContinuousVector* axesShape = nullptr;
    if (GetReduceStdV2Attrs(context, attrs, axesShape) != GRAPH_SUCCESS) {
        return GRAPH_FAILED;
    }

    if (Ops::Base::IsUnknownRank(*inputShape)) {
        Ops::Base::SetUnknownRank(*varShape);
        Ops::Base::SetUnknownRank(*meanShape);
        return GRAPH_SUCCESS;
    }

    std::stringstream strs;
    std::vector<int64_t> axes = GetReduceAxes(inputShape, axesShape, strs);
    if (NormalizeReduceAxes(context, inputShape, axes) != GRAPH_SUCCESS) {
        return GRAPH_FAILED;
    }
    const bool* attrKeepDims = attrs->GetAttrPointer<bool>(VAR_INDEX_ATTR_KEEPDIM);
    bool keepDims = (attrKeepDims == nullptr) ? false : (*attrKeepDims);
    ge::graphStatus inferStat = InferReducedShape(inputShape, axes, keepDims, varShape);

    if (inferStat == ge::GRAPH_SUCCESS) {
        // GE没有可选输出的概念，会给每个输出都申请内存，这里一定要推导mean shape
        *meanShape = *varShape;
        LogInferShapeResult(context, inputShape, strs, keepDims, varShape, meanShape);
    }

    return inferStat;
}

IMPL_OP_INFERSHAPE(ReduceStdV2).InferShape(InferShape4ReduceStdV2);
}  // namespace ops
