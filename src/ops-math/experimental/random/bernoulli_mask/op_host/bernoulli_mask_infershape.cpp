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

namespace ops {
namespace {
constexpr size_t OUTPUT_SHAPE_ATTR = 0;
constexpr size_t MAX_SHAPE_LENGTH = 8;
} // namespace

static ge::graphStatus InferShapeBernoulliMask(gert::InferShapeContext* context)
{
    if (context == nullptr) {
        return ge::GRAPH_FAILED;
    }
    auto attrs = context->GetAttrs();
    if (attrs == nullptr) {
        return ge::GRAPH_FAILED;
    }
    auto shapeAttr = attrs->GetListInt(OUTPUT_SHAPE_ATTR);
    if (shapeAttr == nullptr) {
        return ge::GRAPH_FAILED;
    }
    const size_t dimNum = shapeAttr->GetSize();
    if (dimNum > MAX_SHAPE_LENGTH) {
        return ge::GRAPH_FAILED;
    }
    const auto* dims = shapeAttr->GetData();
    if (dimNum > 0 && dims == nullptr) {
        return ge::GRAPH_FAILED;
    }
    for (size_t i = 0; i < dimNum; ++i) {
        if (dims[i] < 0) {
            return ge::GRAPH_FAILED;
        }
    }
    auto outShape = context->GetOutputShape(0);
    if (outShape == nullptr) {
        return ge::GRAPH_FAILED;
    }

    outShape->SetDimNum(dimNum);
    for (size_t i = 0; i < dimNum; ++i) {
        outShape->SetDim(i, dims[i]);
    }
    return ge::GRAPH_SUCCESS;
}

IMPL_OP_INFERSHAPE(BernoulliMask).InferShape(InferShapeBernoulliMask);
} // namespace ops
