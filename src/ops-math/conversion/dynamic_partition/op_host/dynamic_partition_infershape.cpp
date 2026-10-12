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
 * \file dynamic_partition_infershape.cpp
 * \brief infershape func of DynamicPartition
 */
#include <climits>

#include "register/op_impl_registry.h"
#include "log/log.h"

using namespace ge;
namespace ops
{
static bool CheckHasUnknownDim(const gert::Shape* ptrShape, size_t dimNum)
{
    for (size_t i = 0; i < dimNum; ++i) {
        if (ptrShape->GetDim(i) == -1L) {
            return true;
        }
    }
    return false;
}

template <typename T>
static graphStatus GetNumPartitions(const T* context, int64_t& numParts)
{
    const gert::RuntimeAttrs* attrs = context->GetAttrs();
    OP_CHECK_NULL_WITH_CONTEXT(context, attrs);
    if (attrs->GetAttrNum() > 0) {
        const int64_t* ptrNumParts = attrs->GetAttrPointer<int64_t>(0);
        OP_CHECK_NULL_WITH_CONTEXT(context, ptrNumParts);
        numParts = *ptrNumParts;
    }
    return GRAPH_SUCCESS;
}

static graphStatus InferFirstOutputRange(gert::InferShapeRangeContext* context, const gert::Shape* ptrXMaxShape,
                                         const gert::Shape* ptrXMinShape, const gert::Shape* ptrPartMaxShape,
                                         gert::Shape* ptrYMaxShape, gert::Shape* ptrYMinShape)
{
    auto xDimNum = ptrXMaxShape->GetDimNum();
    auto pDimNum = ptrPartMaxShape->GetDimNum();
    OP_CHECK_IF(pDimNum > xDimNum,
             OP_LOGE(context->GetNodeName(), "The partitions dim num %zu is larger than x's %zu!", pDimNum, xDimNum),
             return GRAPH_FAILED);

    auto yDimNum = xDimNum - pDimNum + 1;
    ptrYMaxShape->SetDimNum(yDimNum);
    ptrYMinShape->SetDimNum(yDimNum);
    int64_t dimVal = 1;
    if (CheckHasUnknownDim(ptrXMaxShape, pDimNum) && CheckHasUnknownDim(ptrPartMaxShape, pDimNum)) {
        dimVal = -1L;
    } else if (CheckHasUnknownDim(ptrPartMaxShape, pDimNum)) {
        for (size_t i = 0; i < pDimNum; ++i) {
            dimVal *= ptrXMaxShape->GetDim(i);
        }
    } else {
        dimVal = ptrPartMaxShape->IsScalar() ? 1L : ptrPartMaxShape->GetShapeSize();
    }
    ptrYMaxShape->SetDim(0, dimVal);
    ptrYMinShape->SetDim(0, 0);
    for (size_t j = 1; j < yDimNum; ++j) {
        ptrYMaxShape->SetDim(j, ptrXMaxShape->GetDim(pDimNum - 1 + j));
        ptrYMinShape->SetDim(j, ptrXMinShape->GetDim(pDimNum - 1 + j));
    }

    return GRAPH_SUCCESS;
}

static graphStatus InferShapeRange4DynamicPartition(gert::InferShapeRangeContext* context)
{
    OP_LOGD(context->GetNodeName(), "InferShapeRange4DynamicPartition start");

    auto ptrXShapeRange = context->GetInputShapeRange(0);
    OP_CHECK_NULL_WITH_CONTEXT(context, ptrXShapeRange);
    auto ptrPartShapeRange = context->GetInputShapeRange(1);
    OP_CHECK_NULL_WITH_CONTEXT(context, ptrPartShapeRange);
    auto ptrYShapeRange = context->GetOutputShapeRange(0);
    OP_CHECK_NULL_WITH_CONTEXT(context, ptrYShapeRange);

    int64_t numParts = 1;
    OP_CHECK_IF(GetNumPartitions<gert::InferShapeRangeContext>(context, numParts) != GRAPH_SUCCESS,
             OP_LOGE(context->GetNodeName(), "Failed to get attr num_partitions!"), return GRAPH_FAILED);

    auto ptrXMaxShape = ptrXShapeRange->GetMax();
    auto ptrXMinShape = ptrXShapeRange->GetMin();
    auto ptrPartMaxShape = ptrPartShapeRange->GetMax();
    auto ptrYMaxShape = ptrYShapeRange->GetMax();
    auto ptrYMinShape = ptrYShapeRange->GetMin();
    OP_CHECK_NULL_WITH_CONTEXT(context, ptrXMaxShape);
    OP_CHECK_NULL_WITH_CONTEXT(context, ptrXMinShape);
    OP_CHECK_NULL_WITH_CONTEXT(context, ptrPartMaxShape);
    OP_CHECK_NULL_WITH_CONTEXT(context, ptrYMaxShape);
    OP_CHECK_NULL_WITH_CONTEXT(context, ptrYMinShape);

    OP_CHECK_IF(InferFirstOutputRange(context, ptrXMaxShape, ptrXMinShape, ptrPartMaxShape, ptrYMaxShape, ptrYMinShape) !=
                 GRAPH_SUCCESS,
             OP_LOGE(context->GetNodeName(), "Failed to infer range for first output!"), return GRAPH_FAILED);

    for (int64_t oIdx = 1; oIdx < numParts; ++oIdx) {
        auto ptrYShapeRange1 = context->GetOutputShapeRange(oIdx);
        OP_CHECK_NULL_WITH_CONTEXT(context, ptrYShapeRange1);
        auto ptrMax = ptrYShapeRange1->GetMax();
        auto ptrMin = ptrYShapeRange1->GetMin();
        OP_CHECK_NULL_WITH_CONTEXT(context, ptrMax);
        OP_CHECK_NULL_WITH_CONTEXT(context, ptrMin);
        *ptrMax = *ptrYMaxShape;
        *ptrMin = *ptrYMinShape;
    }

    OP_LOGD(context->GetNodeName(), "InferShapeRange4DynamicPartition end");
    return GRAPH_SUCCESS;
}

IMPL_OP_INFERSHAPE(DynamicPartition)
     .InferShapeRange(InferShapeRange4DynamicPartition);
}  // namespace ops
