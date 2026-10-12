/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

/* !
 * \file strided_slice_assign_v2_infershape.cpp
 * \brief
 */

#include "log/log.h"
#include "register/op_impl_registry.h"
#include "util/shape_util.h"

using namespace ge;

namespace ops {
static graphStatus InferDataType4SameAsInput(gert::InferDataTypeContext *context)
{
    OP_LOGD(context->GetNodeName(), "InferDataType4SameAsInput enter");
    auto input_x_dtype = context->GetInputDataType(0);
    context->SetOutputDataType(0, input_x_dtype);
    OP_LOGD(context->GetNodeName(), "InferDataType4SameAsInput end");
    return GRAPH_SUCCESS;
}

static graphStatus InferShape4Elewise(gert::InferShapeContext *context)
{
    auto in_shape = context->GetInputShape(0);
    OP_CHECK_NULL_WITH_CONTEXT(context, in_shape);
    auto out_shape = context->GetOutputShape(0);
    OP_CHECK_NULL_WITH_CONTEXT(context, out_shape);

    if (Ops::Base::IsUnknownRank(*in_shape)) {
        Ops::Base::SetUnknownRank(*out_shape);
    } else {
        *out_shape = *in_shape;
    }

    return ge::GRAPH_SUCCESS;
}

IMPL_OP_INFERSHAPE(StridedSliceAssignV2)
    .InferShape(InferShape4Elewise)
    .InferDataType(InferDataType4SameAsInput)
    .InputsDataDependency({ 2, 3, 4, 5 });
} // namespace ops