// ----------------------------------------------------------------------------
// Copyright (c) Huawei Device Co., Ltd.
// This program is free software, you can redistribute it and/or modify it under the terms and conditions of
// CANN Open Software License Agreement Version 2.0 (the "License").
// Please refer to the License for details. You may not use this file except in compliance with the License.
// THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
// INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
// See LICENSE in the root of the software repository for the full text of the License.
// ----------------------------------------------------------------------------

#include "register/op_impl_registry.h"

namespace ops {
static ge::graphStatus InferShapeMaskedScale(gert::InferShapeContext* context)
{
    const gert::Shape* selfShape = context->GetInputShape(0);
    const gert::Shape* maskShape = context->GetInputShape(1);
    if (selfShape == nullptr || maskShape == nullptr) {
        return ge::GRAPH_FAILED;
    }
    if (selfShape->GetShapeSize() != maskShape->GetShapeSize()) {
        return ge::GRAPH_FAILED;
    }
    gert::Shape* yShape = context->GetOutputShape(0);
    if (yShape == nullptr) {
        return ge::GRAPH_FAILED;
    }
    *yShape = *selfShape;
    return ge::GRAPH_SUCCESS;
}

IMPL_OP_INFERSHAPE(MaskedScale).InferShape(InferShapeMaskedScale);
} // namespace ops
