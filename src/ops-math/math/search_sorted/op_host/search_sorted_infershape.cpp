/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include "log/log.h"
#include "register/op_impl_registry.h"

using namespace ge;
namespace ops {
static graphStatus InferShape4SearchSorted(gert::InferShapeContext* context)
{
    OP_LOGI("Begin InferShape4SearchSorted");
    const gert::Shape* values_shape = context->GetInputShape(1);
    OP_CHECK_NULL_WITH_CONTEXT(context, values_shape);

    gert::Shape* out_shape = context->GetOutputShape(0);
    OP_CHECK_NULL_WITH_CONTEXT(context, out_shape);
    *out_shape = *values_shape;

    return GRAPH_SUCCESS;
}

IMPL_OP_INFERSHAPE(SearchSorted).InferShape(InferShape4SearchSorted);
} // namespace ops
