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
 * \file matrix_diag_part_infershape.cpp
 * \brief infershape func of MatrixDiagPart
 */
#include <cmath>
#include <climits>
#include "log/log.h"
#include "register/op_impl_registry.h"
#include "util/shape_util.h"
#include <algorithm>

using namespace ge;

namespace ops {
static constexpr int64_t IDX_0 = 0;

static ge::graphStatus InferShapeMatrixDiagPart(gert::InferShapeContext* context)
{
    OP_LOGD(context->GetNodeName(), "Enter InferShapeMatrixDiagPart");

    const gert::Shape* xShape = context->GetInputShape(IDX_0);
    OP_CHECK_NULL_WITH_CONTEXT(context, xShape);

    gert::Shape* yShape = context->GetOutputShape(IDX_0);
    OP_CHECK_NULL_WITH_CONTEXT(context, yShape);

    // Unknown rank (-2): rank is unknown, cannot infer the diag shape; pass through as unknown rank
    if (Ops::Base::IsUnknownRank(*xShape)) {
        Ops::Base::SetUnknownRank(*yShape);
        return GRAPH_SUCCESS;
    }

    auto xShapeSize = xShape->GetDimNum();
    if (xShapeSize < 2) {
        OP_LOGE_FOR_INVALID_SHAPESIZE_WITH_REASON(context->GetNodeName(), "x", std::to_string(xShapeSize).c_str(),
                                                  "input rank must be >= 2");
        return ge::GRAPH_FAILED;
    }

    int64_t M = xShape->GetDim(xShapeSize - 2);
    int64_t N = xShape->GetDim(xShapeSize - 1);
    int64_t d = std::min(M, N);
    if (d < 0) {
        d = -1;
    }

    yShape->SetDimNum(xShapeSize - 1);
    for (size_t i = 0; i < xShapeSize - 2; i++) {
        yShape->SetDim(i, xShape->GetDim(i));
    }
    yShape->SetDim(xShapeSize - 2, d);

    return GRAPH_SUCCESS;
}

IMPL_OP_INFERSHAPE(MatrixDiagPart).InferShape(InferShapeMatrixDiagPart);
} // namespace ops
