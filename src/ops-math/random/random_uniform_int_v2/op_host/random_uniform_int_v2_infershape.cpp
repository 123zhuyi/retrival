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
 * \file random_uniform_int_v2_infershape.cpp
 * \brief
 */
#include "log/log.h"
#include "register/op_impl_registry.h"

using namespace ge;
namespace ops {

static constexpr uint16_t IN_SHAPE_IDX = 0;

template <typename T>
ge::graphStatus RandomUniformIntV2InferShapeImpl(const T *shape_dims, gert::Shape &outputShape,
                                            size_t shape_size) {
    outputShape.SetDimNum(shape_size);
    for (size_t i = 0U; i < shape_size; i++) {
        outputShape.SetDim(i, shape_dims[i]);
    }
    return ge::GRAPH_SUCCESS;
}

static ge::graphStatus InferShapeForRandomUniformIntV2(gert::InferShapeContext *context) {
    auto x_shape_tensor = context->GetInputTensor(0);
    auto outputShape = context->GetOutputShape(0);
    auto const_shape = context->GetOutputShape(1);
    OP_CHECK_NULL_WITH_CONTEXT(context, x_shape_tensor);
    OP_CHECK_NULL_WITH_CONTEXT(context, outputShape);
    OP_CHECK_NULL_WITH_CONTEXT(context, const_shape);

    const_shape->SetDimNum(1);
    const_shape->SetDim(0, 1);

    auto x_shape_size = x_shape_tensor->GetShapeSize();
    if (x_shape_size < 0) {
        return ge::GRAPH_FAILED;
    }
    if (x_shape_tensor->GetDataType() == ge::DT_INT32) {
        auto xShapeData = x_shape_tensor->GetData<int32_t>();
        return RandomUniformIntV2InferShapeImpl<int32_t>(xShapeData, *outputShape, static_cast<size_t>(x_shape_size));
    } else {
        auto xShapeData = x_shape_tensor->GetData<int64_t>();
        return RandomUniformIntV2InferShapeImpl<int64_t>(xShapeData, *outputShape, static_cast<size_t>(x_shape_size));
    }

    return ge::GRAPH_SUCCESS;
}

IMPL_OP_INFERSHAPE(RandomUniformIntV2).InputsDataDependency({IN_SHAPE_IDX}).InferShape(InferShapeForRandomUniformIntV2);
} // namespace ops