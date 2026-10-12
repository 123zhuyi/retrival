/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

/*!
 * \file non_finite_check.cc
 * \brief
 */

#include "register/op_impl_registry.h"
#include "log/log.h"

using namespace ge;

namespace ops {
const int64_t OUTPUT_IDX = 0;

static ge::graphStatus InferShapeForNonFiniteCheck(gert::InferShapeContext* context)
{
    auto out_shape = context->GetOutputShape(OUTPUT_IDX);
    OP_CHECK_NULL_WITH_CONTEXT(context, out_shape);
    out_shape->SetDimNum(1);
    out_shape->SetDim(0, 1);
    return ge::GRAPH_SUCCESS;
}

IMPL_OP_INFERSHAPE(NonFiniteCheck)
    .InferShape(InferShapeForNonFiniteCheck);
} // namespace ops
