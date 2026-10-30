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
static constexpr size_t INPUT_IDX_X = 0;
static constexpr size_t OUTPUT_IDX_Y = 0;
static constexpr int64_t UNKNOWN_DIM = -1;

static ge::graphStatus Infershape4SliceLastDim(gert::InferShapeContext* context)
{
    auto xShape = context->GetInputShape(INPUT_IDX_X);
    OP_CHECK_NULL_WITH_CONTEXT(context, xShape);
    auto yShape = context->GetOutputShape(OUTPUT_IDX_Y);
    OP_CHECK_NULL_WITH_CONTEXT(context, yShape);

    auto attrs = context->GetAttrs();
    OP_CHECK_NULL_WITH_CONTEXT(context, attrs);
    auto startPtr = attrs->GetAttrPointer<int64_t>(0);
    OP_CHECK_NULL_WITH_CONTEXT(context, startPtr);
    auto endPtr = attrs->GetAttrPointer<int64_t>(1);
    OP_CHECK_NULL_WITH_CONTEXT(context, endPtr);

    int64_t stride = 1;
    auto stridePtr = attrs->GetAttrPointer<int64_t>(2);
    if (stridePtr != nullptr) {
        stride = *stridePtr;
    }

    int64_t rank = static_cast<int64_t>(xShape->GetDimNum());
    yShape->SetDimNum(0);

    for (int64_t i = 0; i < rank - 1; ++i) {
        yShape->AppendDim(xShape->GetDim(i));
    }

    int64_t lastDim = xShape->GetDim(rank - 1);
    if (lastDim == UNKNOWN_DIM) {
        yShape->AppendDim(UNKNOWN_DIM);
    } else {
        int64_t s = *startPtr;
        int64_t e = *endPtr;
        if (s < 0) {
            s += lastDim;
        }
        if (e < 0) {
            e += lastDim;
        }
        if (s < 0) {
            s = 0;
        }
        if (e > lastDim) {
            e = lastDim;
        }
        int64_t length = (e > s) ? (e - s + stride - 1) / stride : 0;
        yShape->AppendDim(length);
    }

    return ge::GRAPH_SUCCESS;
}

IMPL_OP_INFERSHAPE(SliceLastDim).InferShape(Infershape4SliceLastDim);
} // namespace ops
