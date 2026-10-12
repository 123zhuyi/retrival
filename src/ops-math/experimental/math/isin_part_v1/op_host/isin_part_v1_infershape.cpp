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
 * \file isin_part_v1_infer.cpp
 * \brief
 */
#include "register/op_impl_registry.h"
#include "log/log.h"

using namespace ge;

namespace ops {
static constexpr size_t IDX_IN_NUM = 2;
static constexpr size_t IDX_OUT_OUTPUT = 0;

static ge::graphStatus GetNumValue(const gert::Tensor* numTensor, int32_t& elementsNum)
{
    auto tensorDataType = numTensor->GetDataType();
    if (tensorDataType == ge::DT_INT32) {
        const int32_t* constDataPtr = numTensor->GetData<int32_t>();
        elementsNum = (*constDataPtr);
    } else {
        return ge::GRAPH_FAILED;
    }
    return ge::GRAPH_SUCCESS;
}

static ge::graphStatus InferShapeIsinPartV1(gert::InferShapeContext* context)
{
    OP_LOGD(context->GetNodeName(), "Begin to do InferShapeIsinPartV1");

    auto numTensor = context->GetInputTensor(IDX_IN_NUM);
    OP_CHECK_NULL_WITH_CONTEXT(context, numTensor);
    auto outShape = context->GetOutputShape(IDX_OUT_OUTPUT);
    OP_CHECK_NULL_WITH_CONTEXT(context, outShape);
    int32_t elementsNum = 0;
    auto res = GetNumValue(numTensor, elementsNum);
    if (res != ge::GRAPH_SUCCESS) {
        outShape->SetDimNum(0);
        OP_LOGE(context->GetNodeName(), "the dtype of num only support int64_t, infershape failed!");
        return ge::GRAPH_FAILED;
    }
    outShape->SetDimNum(1);
    outShape->SetDim(0, elementsNum);

    OP_LOGD(context->GetNodeName(), "End to do InferShapeIsinPartV1");
    return ge::GRAPH_SUCCESS;
}

IMPL_OP_INFERSHAPE(IsinPartV1).InferShape(InferShapeIsinPartV1);
} // namespace ops