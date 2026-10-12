/**
 * This file is part of the OpenBOAT project at Harbin Institute of Technology (HIT)
 * and is contributed to the CANN Open Software.
 *
 * Copyright (c) 2025 AISS Group, Harbin Institute of Technology (HIT).
 * All Rights Reserved.
 *
 * Authors (accounts):
 * - Zhou Jiamin <@zhou-jiamin-666>
 * - Su Tonghua <@sutonghua>
 *
 * This program is free software: you can redistribute it and/or modify it.
 * Licensed under the CANN Open Software License Agreement Version 2.0 (the "License").
 * You may not use this file except in compliance with the License.
 * See the LICENSE file at the root of the repository for the full text of the License.
 *
 * THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTIES OF ANY KIND, EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 */

/*!
 * \file lin_space_d_infer.cpp
 * \brief
 */
#include "register/op_impl_registry.h"
#include "log/log.h"

using namespace ge;

namespace ops {
static constexpr size_t IDX_IN_START = 0;
static constexpr size_t IDX_IN_NUM = 2;
static constexpr size_t IDX_OUT_OUTPUT = 0;

static ge::graphStatus GetNumValue(const gert::Tensor* numTensor, int64_t& numValue)
{
    auto tensorDataType = numTensor->GetDataType();
    if (tensorDataType == ge::DT_INT64) {
        const int64_t* constDataPtr = numTensor->GetData<int64_t>();
        numValue = (*constDataPtr);
    } else {
        return ge::GRAPH_FAILED;
    }
    return ge::GRAPH_SUCCESS;
}

static ge::graphStatus InferShapeLinSpaceD(gert::InferShapeContext* context)
{
    OP_LOGD(context->GetNodeName(), "Begin to do InferShapeLinSpaceD");

    auto numTensor = context->GetInputTensor(IDX_IN_NUM);
    OP_CHECK_NULL_WITH_CONTEXT(context, numTensor);
    auto outShape = context->GetOutputShape(IDX_OUT_OUTPUT);
    OP_CHECK_NULL_WITH_CONTEXT(context, outShape);
    int64_t numValue = 0;
    auto res = GetNumValue(numTensor, numValue);
    if (res != ge::GRAPH_SUCCESS) {
        outShape->SetDimNum(0);
        OP_LOGE(context->GetNodeName(), "the dtype of num only support int64_t, infershape failed!");
        return ge::GRAPH_FAILED;
    }
    outShape->SetDimNum(1);
    outShape->SetDim(0, numValue);

    OP_LOGD(context->GetNodeName(), "End to do InferShapeLinSpaceD");
    return ge::GRAPH_SUCCESS;
}

IMPL_OP_INFERSHAPE(LinSpaceD).InputsDataDependency({IDX_IN_NUM}).InferShape(InferShapeLinSpaceD);
} // namespace ops