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
 * \file matrix_set_diag_infershape.cpp
 * \brief infershape func of MatrixSetDiag
 */
#include "log/log.h"
#include "register/op_impl_registry.h"
#include "op_host/util/shape_util.h"

using namespace ge;
namespace ops {

ge::graphStatus MatrixSetDiagInferShapeFunc(gert::InferShapeContext* context)
{
    const gert::Shape* shapeIn = context->GetInputShape(0);
    OP_CHECK_NULL_WITH_CONTEXT(context, shapeIn);
    const gert::Shape* shapeDiag = context->GetInputShape(1);
    OP_CHECK_NULL_WITH_CONTEXT(context, shapeDiag);
    auto shapeOut = context->GetOutputShape(0);
    OP_CHECK_NULL_WITH_CONTEXT(context, shapeOut);
    if (Ops::Base::IsUnknownRank(*shapeIn) || Ops::Base::IsUnknownRank(*shapeDiag)) {
        Ops::Base::SetUnknownRank(*shapeOut);
        return ge::GRAPH_SUCCESS;
    }

    // check size
    size_t sizeIn = shapeIn->GetDimNum();
    OP_CHECK_IF(
        sizeIn < 2, OP_LOGE(context->GetNodeName(), "input shape must at least 2 dims."), return ge::GRAPH_FAILED);
    size_t sizeDiag = shapeDiag->GetDimNum();
    OP_CHECK_IF(
        sizeDiag != sizeIn - 1, OP_LOGE(context->GetNodeName(), "diag dims must input dims-1"),
        return ge::GRAPH_FAILED);

    int64_t smallDimInput = std::min(shapeIn->GetDim(sizeIn - 1), shapeIn->GetDim(sizeIn - 2));
    int64_t smallDimDiag = shapeDiag->GetDim(sizeDiag - 1);
    if (smallDimInput != ge::UNKNOWN_DIM && smallDimDiag != ge::UNKNOWN_DIM) {
        OP_CHECK_IF(
            smallDimInput != smallDimDiag, OP_LOGE(context->GetNodeName(), "diag check with input failed"),
            return ge::GRAPH_FAILED);
    }

    *shapeOut = *shapeIn;
    for (size_t i = 0; i < sizeDiag - 1; i++) {
        if (shapeIn->GetDim(i) == ge::UNKNOWN_DIM) {
            if (shapeDiag->GetDim(i) != ge::UNKNOWN_DIM) {
                shapeOut->SetDim(i, shapeDiag->GetDim(i));
            }
        } else {
            if (shapeDiag->GetDim(i) != ge::UNKNOWN_DIM) {
                OP_CHECK_IF(
                    shapeIn->GetDim(i) != shapeDiag->GetDim(i),
                    OP_LOGE(context->GetNodeName(), "dim %zu not the same", i), return ge::GRAPH_FAILED);
            }
        }
    }

    OP_LOGD(context, "out shape: %s", Ops::Base::ToString(*shapeOut).c_str());
    return ge::GRAPH_SUCCESS;
}

IMPL_OP_INFERSHAPE(MatrixSetDiag).InferShape(MatrixSetDiagInferShapeFunc);
} // namespace ops
