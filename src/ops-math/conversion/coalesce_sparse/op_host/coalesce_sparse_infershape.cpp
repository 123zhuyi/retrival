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
 * \file coalesce_sparse_infershape.cpp
 * \brief
 */
#include "log/log.h"
#include "register/op_impl_registry.h"
using namespace ge;
namespace ops {
static ge::graphStatus InferShape4CoalesceSparse(gert::InferShapeContext* context)
{
    const gert::Shape* uniqueLenShape = context->GetInputShape(0);
    const gert::Shape* uniqueShape = context->GetInputShape(1);
    const gert::Shape* indicesShape = context->GetInputShape(2);
    const gert::Shape* valueShape = context->GetInputShape(3);
    OP_CHECK_NULL_WITH_CONTEXT(context, uniqueLenShape);
    OP_CHECK_NULL_WITH_CONTEXT(context, uniqueShape);
    OP_CHECK_NULL_WITH_CONTEXT(context, indicesShape);
    OP_CHECK_NULL_WITH_CONTEXT(context, valueShape);

    gert::Shape* newIndicesShape = context->GetOutputShape(0);
    gert::Shape* newValueShape = context->GetOutputShape(1);

    uint64_t len = uniqueLenShape->GetDim(0);
    OP_CHECK_NULL_WITH_CONTEXT(context, newIndicesShape);
    newIndicesShape->AppendDim(len);
    newIndicesShape->AppendDim(indicesShape->GetDim(1));

    uint64_t valueShapeSize = valueShape->GetDimNum();
    OP_CHECK_NULL_WITH_CONTEXT(context, newValueShape);
    newValueShape->AppendDim(len);
    for (uint64_t i = 1; i < valueShapeSize; i++) {
        newValueShape->AppendDim(valueShape->GetDim(i));
    }

    return ge::GRAPH_SUCCESS;
}

static ge::graphStatus InferDataType4CoalesceSparse(gert::InferDataTypeContext* context)
{
    OP_LOGD(context, "Begin to do InferDataType4CoalesceSparse");

    auto indicesDataType = context->GetInputDataType(2);
    OP_LOGD(context, "indices dtype:%s", Ops::Base::ToString(indicesDataType).c_str());

    auto valueDataType = context->GetInputDataType(3);
    OP_LOGD(context, "values dtype:%s", Ops::Base::ToString(valueDataType).c_str());

    context->SetOutputDataType(0, indicesDataType);
    context->SetOutputDataType(1, valueDataType);

    OP_LOGD(context, "End to do InferDataType4CoalesceSparse end");
    return ge::GRAPH_SUCCESS;
}

IMPL_OP_INFERSHAPE(CoalesceSparse).InferShape(InferShape4CoalesceSparse).InferDataType(InferDataType4CoalesceSparse);
} // namespace ops
