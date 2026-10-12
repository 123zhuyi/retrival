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
 * \file get_dynamic_dims_infershape.cpp
 * \brief
 */

#include <algorithm>

#include "log/log.h"
#include "register/op_impl_registry.h"

using namespace ge;

namespace ops {
namespace {
constexpr size_t kInputIrIndex = 0U;
constexpr size_t kOutputDimsIndex = 0U;
constexpr size_t kShapeInfoAttrIndex = 0U;
constexpr size_t kAttrNIndex = 1U;
constexpr int64_t kUnknownDim = -1;
} // namespace

static ge::graphStatus InferShape4GetDynamicDims(gert::InferShapeContext* context)
{
    OP_LOGD(context->GetNodeName(), "Begin to do InferShape4GetDynamicDims");
    const gert::RuntimeAttrs* attrs = context->GetAttrs();
    OP_CHECK_NULL_WITH_CONTEXT(context, attrs);

    const gert::TypedContinuousVector<int64_t>* shapeInfo = attrs->GetListInt(kShapeInfoAttrIndex);
    OP_CHECK_NULL_WITH_CONTEXT(context, shapeInfo);
    const int64_t* nAttr = attrs->GetInt(kAttrNIndex);
    OP_CHECK_NULL_WITH_CONTEXT(context, nAttr);

    const gert::AnchorInstanceInfo* inputInstanceInfo = context->GetIrInputInstanceInfo(kInputIrIndex);
    OP_CHECK_NULL_WITH_CONTEXT(context, inputInstanceInfo);
    const size_t inputInstanceNum = inputInstanceInfo->GetInstanceNum();
    if (static_cast<int64_t>(inputInstanceNum) != *nAttr) {
        OP_LOGE(context->GetNodeName(), "Inputs size [%zu] must equal attr N [%ld]", inputInstanceNum, *nAttr);
        return ge::GRAPH_FAILED;
    }

    const int64_t* shapeInfoData = shapeInfo->GetData();
    int64_t unknownDimsNum = std::count(shapeInfoData, shapeInfoData + shapeInfo->GetSize(), kUnknownDim);
    if (unknownDimsNum == 0) {
        OP_LOGE(context->GetNodeName(), "No need to perform GetDynamicDims in a known shape");
        return ge::GRAPH_FAILED;
    }

    gert::Shape vectorShape = {unknownDimsNum};
    gert::Shape* dimsShape = context->GetOutputShape(kOutputDimsIndex);
    OP_CHECK_NULL_WITH_CONTEXT(context, dimsShape);
    *dimsShape = vectorShape;
    OP_LOGD(context->GetNodeName(), "End to do InferShape4GetDynamicDims");
    return ge::GRAPH_SUCCESS;
}

IMPL_OP_INFERSHAPE(GetDynamicDims).InferShape(InferShape4GetDynamicDims);
} // namespace ops
