/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

/* !
 * \file unfold_grad_infershape.cpp
 * \brief
 */

#include "log/log.h"
#include "register/op_impl_registry.h"

using namespace ge;

namespace {
constexpr uint32_t MAX_DIM_NUM = 8;
constexpr uint32_t INPUT_GRADOUT_IDX = 0;
constexpr uint32_t INPUT_INPUTSIZE_IDX = 1;
constexpr uint32_t OUTPUT_GRADIN_IDX = 0;
} // namespace

namespace ops {
template <typename T>
static ge::graphStatus InferShapeCopyData2Array(const gert::Tensor *listTensor,
    int64_t listSize, int64_t dataList[])
{
    const T *listDataPtr = listTensor->GetData<T>();
    for (int64_t i = 0; i < listSize; i++) {
        dataList[i] = listDataPtr[i];
    }

    return ge::GRAPH_SUCCESS;
}

template <typename T>
static ge::graphStatus InferShapeGetConstInputData(gert::InferShapeContext *context, const size_t idxInput, T &dataList,
    int64_t &dataListLength)
{
    auto listTensor = context->GetInputTensor(idxInput);
    if (listTensor == nullptr) {
        return ge::GRAPH_FAILED;
    }
    auto inputSizeDesc = context->GetInputDesc(INPUT_INPUTSIZE_IDX);
    OP_CHECK_NULL_WITH_CONTEXT(context, inputSizeDesc);
    auto listDataType = inputSizeDesc->GetDataType();
    auto inputSizeShapePtr = context->GetInputShape(INPUT_INPUTSIZE_IDX);
    OP_CHECK_NULL_WITH_CONTEXT(context, inputSizeShapePtr);
    int64_t listSize = static_cast<int64_t>(inputSizeShapePtr->GetDim(0));
    dataListLength = listSize;

    if (listDataType == ge::DT_INT32) {
        return InferShapeCopyData2Array<int32_t>(listTensor, listSize, dataList);
    }
    if (listDataType == ge::DT_INT64) {
        return InferShapeCopyData2Array<int64_t>(listTensor, listSize, dataList);
    }

    return ge::GRAPH_FAILED;
}

static ge::graphStatus InferShapeForUnfoldGrad(gert::InferShapeContext *context)
{
    int64_t inputSizeArray[MAX_DIM_NUM] = {0};
    int64_t inputSizeLength2 = 0;

    if (InferShapeGetConstInputData(context, INPUT_INPUTSIZE_IDX, inputSizeArray, inputSizeLength2) !=
        ge::GRAPH_SUCCESS) {
        return ge::GRAPH_FAILED;
    }

    gert::Shape *y_shape = context->GetOutputShape(OUTPUT_GRADIN_IDX);
    OP_CHECK_NULL_WITH_CONTEXT(context, y_shape);
    y_shape->SetDimNum(inputSizeLength2);
    for (int i = 0; i < inputSizeLength2; i++) {
        y_shape->SetDim(i, inputSizeArray[i]);
    }

    return GRAPH_SUCCESS;
}

static ge::graphStatus InferDataTypeForUnfoldGrad(gert::InferDataTypeContext *context)
{
    const auto gardInDTypeInfer = context->GetInputDataType(INPUT_GRADOUT_IDX);
    context->SetOutputDataType(OUTPUT_GRADIN_IDX, gardInDTypeInfer);

    return ge::GRAPH_SUCCESS;
}

IMPL_OP_INFERSHAPE(UnfoldGrad).InferShape(InferShapeForUnfoldGrad).InferDataType(InferDataTypeForUnfoldGrad);
} // namespace ops