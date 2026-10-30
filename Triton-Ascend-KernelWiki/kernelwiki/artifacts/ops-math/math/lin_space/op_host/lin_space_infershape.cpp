/**
 * Copyright (c) 2025-2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

/*!
 * \file lin_space_infershape.cpp
 * \brief
 */
#include "log/log.h"
#include "register/op_impl_registry.h"

using namespace ge;
namespace ops {
static constexpr size_t LINSPACE_IDX_IN_START = 0;
static constexpr size_t LINSPACE_IDX_IN_NUM = 2;
static constexpr size_t LINSPACE_IDX_OUT_OUTPUT = 0;
static constexpr int64_t LINSPACE_UNKOWNSHAPE = -2;

static ge::graphStatus GetNumValue(const gert::Tensor* numTensor, int64_t& numValue)
{
    auto tensorDataType = numTensor->GetDataType();
    if (tensorDataType == ge::DT_INT32) {
        const int32_t* constDataPtr = numTensor->GetData<int32_t>();
        if (constDataPtr) {
            numValue = static_cast<int64_t>(*constDataPtr);
        } else {
            numValue = LINSPACE_UNKOWNSHAPE;
        }
    } else if (tensorDataType == ge::DT_INT64) {
        const int64_t* constDataPtr = numTensor->GetData<int64_t>();
        if (constDataPtr) {
            numValue = (*constDataPtr);
        } else {
            numValue = LINSPACE_UNKOWNSHAPE;
        }
    } else {
        return ge::GRAPH_FAILED;
    }
    return ge::GRAPH_SUCCESS;
}

static ge::graphStatus InferShape4LinSpace(gert::InferShapeContext* context)
{
    OP_LOGD(context->GetNodeName(), "Begin to do InferShape4LinSpace");

    auto numTensor = context->GetInputTensor(LINSPACE_IDX_IN_NUM);
    OP_CHECK_NULL_WITH_CONTEXT(context, numTensor);
    auto outShape = context->GetOutputShape(LINSPACE_IDX_OUT_OUTPUT);
    OP_CHECK_NULL_WITH_CONTEXT(context, outShape);
    int64_t numValue = 0;
    auto res = GetNumValue(numTensor, numValue);
    if (res != ge::GRAPH_SUCCESS) {
        outShape->SetDimNum(0);
        OP_LOGE(context->GetNodeName(), "the dtype of num only support int32_t and int64_t, infershape failed!");
        return ge::GRAPH_FAILED;
    }
    outShape->SetDimNum(1);
    outShape->SetDim(0, numValue);
    OP_LOGD(context->GetNodeName(), "The numValue is %ld", numValue);
    OP_LOGD(context->GetNodeName(), "End to do InferShape4LinSpace");
    return ge::GRAPH_SUCCESS;
}

IMPL_OP_INFERSHAPE(LinSpace)
    .InputsDataDependency({LINSPACE_IDX_IN_NUM})
    .InferShape(InferShape4LinSpace);
} // namespace ops
