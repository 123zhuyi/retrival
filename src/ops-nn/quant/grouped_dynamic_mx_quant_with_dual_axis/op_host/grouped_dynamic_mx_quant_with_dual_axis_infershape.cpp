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
 * \file grouped_dynamic_mx_quant_with_dual_axis_infershape.cpp
 * \brief
 */

#include <algorithm>
#include <string>
#include <vector>
#include "log/log.h"
#include "register/op_impl_registry.h"
#include "util/math_util.h"
#include "util/shape_util.h"
#include "graph/utils/type_utils.h"

using namespace ge;
namespace ops {
constexpr size_t INDEX_INPUT_X = 0;
constexpr size_t INDEX_INPUT_GROUP_INDEX = 1;
constexpr size_t INDEX_OUTPUT_Y1 = 0;
constexpr size_t INDEX_OUTPUT_SCALE1 = 1;
constexpr size_t INDEX_OUTPUT_Y2 = 2;
constexpr size_t INDEX_OUTPUT_SCALE2 = 3;
constexpr size_t INDEX_ATTR_ROUND_MODE = 0;
constexpr size_t INDEX_ATTR_SCALE_ALG = 1;
constexpr size_t INDEX_ATTR_DST_DTYPE = 2;
constexpr size_t INDEX_ATTR_MAX_DTYPE_VALUE = 3;
constexpr int64_t BLOCK_SIZE = 32;
constexpr int64_t SCALE_PAIR = 2;
constexpr int64_t SCALE_AXIS_UNIT = BLOCK_SIZE * SCALE_PAIR;
constexpr int64_t UNKNOWN_DIM = -1;
constexpr float ZERO_FLOAT = 0.0f;

static const std::vector<ge::DataType> Y_SUPPORT_DTYPE_SET = {ge::DT_FLOAT8_E4M3FN, ge::DT_FLOAT8_E5M2};

static bool IsUnknownDim(int64_t dim) { return dim == UNKNOWN_DIM; }

static graphStatus ValidateAttrs(gert::InferShapeContext* context)
{
    auto attrsPtr = context->GetAttrs();
    OP_CHECK_NULL_WITH_CONTEXT(context, attrsPtr);

    const char* roundMode = attrsPtr->GetAttrPointer<char>(INDEX_ATTR_ROUND_MODE);
    OP_CHECK_NULL_WITH_CONTEXT(context, roundMode);
    OP_CHECK_IF(std::string(roundMode) != "rint", OP_LOGE(context, "round_mode should be rint, but got %s.", roundMode),
                return ge::GRAPH_FAILED);
    const int64_t* scaleAlg = attrsPtr->GetAttrPointer<int64_t>(INDEX_ATTR_SCALE_ALG);
    OP_CHECK_NULL_WITH_CONTEXT(context, scaleAlg);
    OP_CHECK_IF(*scaleAlg != 1, OP_LOGE(context, "scale_alg should be 1, but got %ld.", *scaleAlg),
                return ge::GRAPH_FAILED);

    const int64_t* dstDtype = attrsPtr->GetAttrPointer<int64_t>(INDEX_ATTR_DST_DTYPE);
    OP_CHECK_NULL_WITH_CONTEXT(context, dstDtype);
    ge::DataType outDtype = static_cast<ge::DataType>(*dstDtype);
    OP_CHECK_IF(
        std::find(Y_SUPPORT_DTYPE_SET.begin(), Y_SUPPORT_DTYPE_SET.end(), outDtype) == Y_SUPPORT_DTYPE_SET.end(),
        OP_LOGE(context, "dst_dtype should be FLOAT8_E4M3FN or FLOAT8_E5M2, but got %s.",
                ge::TypeUtils::DataTypeToSerialString(outDtype).c_str()),
        return ge::GRAPH_FAILED);

    const float* maxDtypeValue = attrsPtr->GetAttrPointer<float>(INDEX_ATTR_MAX_DTYPE_VALUE);
    OP_CHECK_NULL_WITH_CONTEXT(context, maxDtypeValue);
    OP_CHECK_IF(*maxDtypeValue != ZERO_FLOAT,
                OP_LOGE(context, "max_dtype_value should be 0.0, but got %f.", *maxDtypeValue),
                return ge::GRAPH_FAILED);

    return ge::GRAPH_SUCCESS;
}

static graphStatus SetUnknownRankOutputShapes(gert::InferShapeContext* context)
{
    gert::Shape* y1Shape = context->GetOutputShape(INDEX_OUTPUT_Y1);
    OP_CHECK_NULL_WITH_CONTEXT(context, y1Shape);
    gert::Shape* scale1Shape = context->GetOutputShape(INDEX_OUTPUT_SCALE1);
    OP_CHECK_NULL_WITH_CONTEXT(context, scale1Shape);
    gert::Shape* y2Shape = context->GetOutputShape(INDEX_OUTPUT_Y2);
    OP_CHECK_NULL_WITH_CONTEXT(context, y2Shape);
    gert::Shape* scale2Shape = context->GetOutputShape(INDEX_OUTPUT_SCALE2);
    OP_CHECK_NULL_WITH_CONTEXT(context, scale2Shape);
    Ops::Base::SetUnknownRank(*y1Shape);
    Ops::Base::SetUnknownRank(*scale1Shape);
    Ops::Base::SetUnknownRank(*y2Shape);
    Ops::Base::SetUnknownRank(*scale2Shape);
    return ge::GRAPH_SUCCESS;
}

static graphStatus SetOutputShapes(gert::InferShapeContext* context, const gert::Shape* xShape,
                                   const gert::Shape* groupIndexShape)
{
    gert::Shape* y1Shape = context->GetOutputShape(INDEX_OUTPUT_Y1);
    OP_CHECK_NULL_WITH_CONTEXT(context, y1Shape);
    gert::Shape* scale1Shape = context->GetOutputShape(INDEX_OUTPUT_SCALE1);
    OP_CHECK_NULL_WITH_CONTEXT(context, scale1Shape);
    gert::Shape* y2Shape = context->GetOutputShape(INDEX_OUTPUT_Y2);
    OP_CHECK_NULL_WITH_CONTEXT(context, y2Shape);
    gert::Shape* scale2Shape = context->GetOutputShape(INDEX_OUTPUT_SCALE2);
    OP_CHECK_NULL_WITH_CONTEXT(context, scale2Shape);

    const int64_t m = xShape->GetDim(0);
    const int64_t n = xShape->GetDim(1);
    const int64_t groupNum = groupIndexShape->GetDim(0);
    *y1Shape = *xShape;
    *y2Shape = *xShape;

    scale1Shape->SetDimNum(3);
    scale1Shape->SetDim(0, m);
    scale1Shape->SetDim(1, IsUnknownDim(n) ? UNKNOWN_DIM : (n + SCALE_AXIS_UNIT - 1) / SCALE_AXIS_UNIT);
    scale1Shape->SetDim(2, SCALE_PAIR);

    scale2Shape->SetDimNum(3);
    const int64_t scale2Row = (IsUnknownDim(m) || IsUnknownDim(groupNum)) ? UNKNOWN_DIM :
                                                                            (m / SCALE_AXIS_UNIT + groupNum);
    scale2Shape->SetDim(0, scale2Row);
    scale2Shape->SetDim(1, n);
    scale2Shape->SetDim(2, SCALE_PAIR);
    return ge::GRAPH_SUCCESS;
}

graphStatus InferShapeForGroupedDynamicMxQuantWithDualAxis(gert::InferShapeContext* context)
{
    OP_LOGD(context->GetNodeName(), "Begin to do InferShapeForGroupedDynamicMxQuantWithDualAxis");
    const gert::Shape* xShape = context->GetInputShape(INDEX_INPUT_X);
    OP_CHECK_NULL_WITH_CONTEXT(context, xShape);
    const gert::Shape* groupIndexShape = context->GetInputShape(INDEX_INPUT_GROUP_INDEX);
    OP_CHECK_NULL_WITH_CONTEXT(context, groupIndexShape);

    if (ValidateAttrs(context) != ge::GRAPH_SUCCESS) {
        return ge::GRAPH_FAILED;
    }

    if (Ops::Base::IsUnknownRank(*xShape) || Ops::Base::IsUnknownRank(*groupIndexShape)) {
        return SetUnknownRankOutputShapes(context);
    }

    const int64_t xDimNum = xShape->GetDimNum();
    OP_CHECK_IF(xDimNum != 2, OP_LOGE(context, "x rank should be 2, but got %ld.", xDimNum), return ge::GRAPH_FAILED);
    const int64_t groupDimNum = groupIndexShape->GetDimNum();
    OP_CHECK_IF(groupDimNum != 1, OP_LOGE(context, "group_index rank should be 1, but got %ld.", groupDimNum),
                return ge::GRAPH_FAILED);

    const int64_t n = xShape->GetDim(1);
    const int64_t groupNum = groupIndexShape->GetDim(0);
    OP_CHECK_IF(groupNum == 0,
                OP_LOGE(context, "group_index shape should not be empty, but got %s.",
                        Ops::Base::ToString(*groupIndexShape).c_str()),
                return ge::GRAPH_FAILED);
    OP_CHECK_IF(
        !IsUnknownDim(n) && n % SCALE_AXIS_UNIT != 0,
        OP_LOGE(context, "x.shape[1] should be 64 aligned, but got x shape %s.", Ops::Base::ToString(*xShape).c_str()),
        return ge::GRAPH_FAILED);

    if (SetOutputShapes(context, xShape, groupIndexShape) != ge::GRAPH_SUCCESS) {
        return ge::GRAPH_FAILED;
    }
    OP_LOGD(context->GetNodeName(), "End to do InferShapeForGroupedDynamicMxQuantWithDualAxis");
    return ge::GRAPH_SUCCESS;
}

ge::graphStatus InferDataTypeForGroupedDynamicMxQuantWithDualAxis(gert::InferDataTypeContext* context)
{
    OP_LOGD(context->GetNodeName(), "Begin to do InferDataTypeForGroupedDynamicMxQuantWithDualAxis");
    auto attrsPtr = context->GetAttrs();
    OP_CHECK_NULL_WITH_CONTEXT(context, attrsPtr);
    const int64_t* dstDtype = attrsPtr->GetAttrPointer<int64_t>(INDEX_ATTR_DST_DTYPE);
    OP_CHECK_NULL_WITH_CONTEXT(context, dstDtype);
    ge::DataType outDtype = static_cast<ge::DataType>(*dstDtype);

    context->SetOutputDataType(INDEX_OUTPUT_Y1, outDtype);
    context->SetOutputDataType(INDEX_OUTPUT_SCALE1, ge::DT_FLOAT8_E8M0);
    context->SetOutputDataType(INDEX_OUTPUT_Y2, outDtype);
    context->SetOutputDataType(INDEX_OUTPUT_SCALE2, ge::DT_FLOAT8_E8M0);
    OP_LOGD(context->GetNodeName(), "End to do InferDataTypeForGroupedDynamicMxQuantWithDualAxis");
    return ge::GRAPH_SUCCESS;
}

IMPL_OP_INFERSHAPE(GroupedDynamicMxQuantWithDualAxis)
    .InferShape(InferShapeForGroupedDynamicMxQuantWithDualAxis)
    .InferDataType(InferDataTypeForGroupedDynamicMxQuantWithDualAxis);
} // namespace ops
