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
 * \file assign_sub_infershape.cpp
 * \brief
 */

#include "register/op_impl_registry.h"
#include "exe_graph/runtime/infer_shape_context.h"
#include "op_common/log/log.h"

using namespace ge;

namespace ops {

static ge::graphStatus InferShapeAssignSub(gert::InferShapeContext* context)
{
    const gert::Shape* varShape = context->GetInputShape(0);
    OP_CHECK_NULL_WITH_CONTEXT(context, varShape);
    const gert::Shape* valueShape = context->GetInputShape(1);
    OP_CHECK_NULL_WITH_CONTEXT(context, valueShape);
    gert::Shape* outShape = context->GetOutputShape(0);
    OP_CHECK_NULL_WITH_CONTEXT(context, outShape);

    if (varShape->GetDimNum() != valueShape->GetDimNum()) {
        OP_LOGE(context, "var and value dim num mismatch: %zu vs %zu", varShape->GetDimNum(), valueShape->GetDimNum());
        return ge::GRAPH_FAILED;
    }
    for (size_t i = 0; i < varShape->GetDimNum(); i++) {
        if (varShape->GetDim(i) != valueShape->GetDim(i)) {
            OP_LOGE(context, "var and value shape mismatch at dim %zu", i);
            return ge::GRAPH_FAILED;
        }
    }
    *outShape = *varShape;
    return ge::GRAPH_SUCCESS;
}

static ge::graphStatus InferDataTypeAssignSub(gert::InferDataTypeContext* context)
{
    context->SetOutputDataType(0, context->GetInputDataType(0));
    return ge::GRAPH_SUCCESS;
}

IMPL_OP_INFERSHAPE(AssignSub).InferShape(InferShapeAssignSub).InferDataType(InferDataTypeAssignSub);

} // namespace ops
