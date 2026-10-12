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
 * \file squared_difference_infershape.cpp
 * \brief SquaredDifference 算子形状推导（NumPy broadcast 语义）
 */

#include "register/op_impl_registry.h"
#include "exe_graph/runtime/infer_shape_context.h"
#include <algorithm>

using namespace ge;

namespace ops {

static ge::graphStatus InferShapeSquaredDifference(gert::InferShapeContext* context)
{
    const gert::Shape* s1 = context->GetInputShape(0);
    const gert::Shape* s2 = context->GetInputShape(1);
    gert::Shape* outShape = context->GetOutputShape(0);
    if (s1 == nullptr || s2 == nullptr || outShape == nullptr) {
        return ge::GRAPH_FAILED;
    }

    int64_t n1 = static_cast<int64_t>(s1->GetDimNum());
    int64_t n2 = static_cast<int64_t>(s2->GetDimNum());
    int64_t n = std::max(n1, n2);

    if (n == 0) {
        outShape->SetDimNum(0);
        return ge::GRAPH_SUCCESS;
    }

    outShape->SetDimNum(static_cast<size_t>(n));
    for (int64_t i = 0; i < n; i++) {
        int64_t d1 = (i < n - n1) ? 1 : s1->GetDim(i - (n - n1));
        int64_t d2 = (i < n - n2) ? 1 : s2->GetDim(i - (n - n2));
        if (d1 != d2 && d1 != 1 && d2 != 1) {
            return ge::GRAPH_FAILED;
        }
        int64_t outDim = (d1 == d2) ? d1 : ((d1 == 1) ? d2 : d1);
        outShape->SetDim(i, outDim);
    }
    return ge::GRAPH_SUCCESS;
}

IMPL_OP_INFERSHAPE(SquaredDifference).InferShape(InferShapeSquaredDifference);

} // namespace ops
