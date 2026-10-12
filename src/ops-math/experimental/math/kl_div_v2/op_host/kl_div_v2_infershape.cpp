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
 * \file kl_div_v2_infershape.cpp
 * \brief
 */

#include "register/op_impl_registry.h"
#include "exe_graph/runtime/infer_shape_context.h"
#include "op_common/log/log.h"
#include <cstring>

using namespace ge;

namespace ops {

static constexpr size_t IDX_0 = 0;
static constexpr size_t ATTR_REDUCTION_IDX = 0;

static ge::graphStatus InferShapeKLDivV2(gert::InferShapeContext* context)
{
    const gert::Shape* xShape = context->GetInputShape(IDX_0);
    OP_CHECK_NULL_WITH_CONTEXT(context, xShape);
    gert::Shape* yShape = context->GetOutputShape(IDX_0);
    OP_CHECK_NULL_WITH_CONTEXT(context, yShape);

    const char* reduction = "mean";
    auto attrs = context->GetAttrs();
    if (attrs != nullptr && attrs->GetAttrNum() > ATTR_REDUCTION_IDX) {
        const char* attrStr = attrs->GetStr(ATTR_REDUCTION_IDX);
        if (attrStr != nullptr) {
            reduction = attrStr;
        }
    }

    bool isNone = (strcmp(reduction, "none") == 0);
    if (isNone) {
        *yShape = *xShape;
    } else {
        yShape->SetDimNum(1);
        yShape->SetDim(0, 1);
    }
    return ge::GRAPH_SUCCESS;
}

static ge::graphStatus InferDataTypeKLDivV2(gert::InferDataTypeContext* context)
{
    context->SetOutputDataType(IDX_0, context->GetInputDataType(IDX_0));
    return ge::GRAPH_SUCCESS;
}

IMPL_OP_INFERSHAPE(KLDivV2).InferShape(InferShapeKLDivV2).InferDataType(InferDataTypeKLDivV2);

} // namespace ops
