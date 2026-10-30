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
 * \file mul_addn.cpp
 * \brief
 */
#include "register/op_impl_registry.h"
#include "log/log.h"

using namespace ge;
namespace ops {
constexpr int64_t INPUT_X1_IDX = 0;
constexpr int64_t INPUT_DIM_NUM = 0;
constexpr int64_t Tiling_KEY_FLOAT32 = 0;
constexpr int64_t Tiling_KEY_FLOAT16 = 0;
constexpr int64_t Tiling_KEY_BFLOAT16 = 0;

constexpr int64_t IR_IDX1 = 0;
constexpr int64_t IR_IDX2 = 1;
constexpr int64_t RALATIVE_IDX = 0;
constexpr int64_t OUTPUT_Y_IDX = 0;
constexpr int64_t X_IDX = 0;

constexpr size_t DIM0 = 0;
constexpr size_t DIM1 = 1;
constexpr size_t DIM2 = 2;
constexpr size_t outDimNum = 3;

static ge::graphStatus InferShapeForMulAddn(gert::InferShapeContext* context)
{
    auto x1Shape = context->GetDynamicInputShape(IR_IDX1, RALATIVE_IDX);
    auto x2Shape = context->GetDynamicInputShape(IR_IDX2, RALATIVE_IDX);
    OP_CHECK_NULL_WITH_CONTEXT(context, x1Shape);
    OP_CHECK_NULL_WITH_CONTEXT(context, x2Shape);

    auto B = x1Shape->GetDim(DIM0);
    auto M = x1Shape->GetDim(DIM1);
    auto N = x2Shape->GetDim(DIM2);

    gert::Shape* yShape = context->GetOutputShape(OUTPUT_Y_IDX);
    OP_CHECK_NULL_WITH_CONTEXT(context, yShape);

    yShape->SetDimNum(outDimNum);
    yShape->SetDim(DIM0, B);
    yShape->SetDim(DIM1, M);
    yShape->SetDim(DIM2, N);
    return GRAPH_SUCCESS;
}

static ge::graphStatus InferDataTypeForMulAddn(gert::InferDataTypeContext* context)
{
    const ge::DataType x1DataType = context->GetInputDataType(X_IDX);
    context->SetOutputDataType(X_IDX, x1DataType);
    return ge::GRAPH_SUCCESS;
}

IMPL_OP_INFERSHAPE(MulAddn).InferShape(InferShapeForMulAddn).InferDataType(InferDataTypeForMulAddn);
} // namespace ops