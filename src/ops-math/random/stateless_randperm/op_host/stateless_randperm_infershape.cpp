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
* \file stateless_randperm_infershape.cpp
* \brief
*/
#include "log/log.h"
#include "register/op_impl_registry.h"

using namespace ge;
namespace ops {
template <typename T>
ge::graphStatus StatelessRandpermInferShapeImpl(const T *shapeDims, gert::Shape &outputShape,
                                            size_t shapeSize) {
    outputShape.SetDimNum(shapeSize);
    outputShape.SetDim(0, shapeDims[0]);
    return ge::GRAPH_SUCCESS;
}

static ge::graphStatus InferShapeForStatelessRandperm(gert::InferShapeContext *context) {
    auto nShapeTensor = context->GetInputTensor(0);
    auto outputShape = context->GetOutputShape(0);
    OP_CHECK_NULL_WITH_CONTEXT(context, nShapeTensor);
    OP_CHECK_NULL_WITH_CONTEXT(context, outputShape);

    auto nShapeSize = nShapeTensor->GetShapeSize();
    if (nShapeSize != 1) {
        return ge::GRAPH_FAILED;
    }
    auto nShapeData = nShapeTensor->GetData<int64_t>();
    return StatelessRandpermInferShapeImpl<int64_t>(nShapeData, *outputShape, static_cast<size_t>(nShapeSize));
}

IMPL_OP_INFERSHAPE(StatelessRandperm).InputsDataDependency({0}).InferShape(InferShapeForStatelessRandperm);
} // namespace ops