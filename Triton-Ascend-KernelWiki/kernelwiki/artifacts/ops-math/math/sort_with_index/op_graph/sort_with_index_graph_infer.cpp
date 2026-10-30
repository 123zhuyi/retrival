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
 * \file sort_with_index_graph_infer.cpp
 * \brief
 */
#include "register/op_impl_registry.h"
#include "log/log.h"

using namespace ge;
namespace ops
{
static constexpr int X_IDX = 0;
static constexpr int INDEX_IDX = 1;
static constexpr int Y1_IDX = 0;
static constexpr int Y2_IDX = 1;

graphStatus InferDataType4SortWithIndex(gert::InferDataTypeContext* context)
{
    OP_LOGD(context->GetNodeName(), "Begin to do InferDataType4SortWithIndex");
    context->SetOutputDataType(Y1_IDX, context->GetInputDataType(X_IDX));
    context->SetOutputDataType(Y2_IDX, context->GetInputDataType(INDEX_IDX));
    OP_LOGD(context->GetNodeName(), "End to do InferDataType4SortWithIndex");
    return GRAPH_SUCCESS;
}

IMPL_OP(SortWithIndex).InferDataType(InferDataType4SortWithIndex);
}  // namespace ops