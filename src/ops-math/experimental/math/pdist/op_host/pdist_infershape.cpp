/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include "register/op_impl_registry.h"
#include "log/log.h"

using namespace ge;

namespace ops {

static ge::graphStatus InferShape4Pdist(gert::InferShapeContext* context)
{
    const gert::Shape* inputShape = context->GetInputShape(0);
    OP_CHECK_NULL_WITH_CONTEXT(context, inputShape);

    OP_CHECK_IF(inputShape->GetDimNum() != 2,
        OP_LOGE(context, "Pdist requires 2D input"), return ge::GRAPH_FAILED);

    int64_t N = inputShape->GetDim(0);
    OP_CHECK_IF(N < 2, OP_LOGE(context, "Pdist requires N >= 2, got %ld", N), return ge::GRAPH_FAILED);

    gert::Shape* outputShape = context->GetOutputShape(0);
    OP_CHECK_NULL_WITH_CONTEXT(context, outputShape);

    int64_t outputSize = N * (N - 1) / 2;

    outputShape->SetDimNum(1);
    outputShape->SetDim(0, outputSize);

    return ge::GRAPH_SUCCESS;
}

IMPL_OP_INFERSHAPE(Pdist).InferShape(InferShape4Pdist);

} // namespace ops
