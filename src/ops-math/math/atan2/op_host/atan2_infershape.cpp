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
 * \file atan2_infershape.cpp
 * \brief Atan2 InferShape and InferDataType.
 *        Output shape is the broadcast result of x1 and x2,
 *        output dtype is the same as input x1.
 */
#include "register/op_impl_registry.h"
#include "infershape_broadcast_util.h"
#include "log/log.h"

using namespace ge;

namespace ops {
constexpr size_t INPUT_COUNT = 2;
constexpr size_t INPUT_X1_INDEX = 0;
constexpr size_t INPUT_X2_INDEX = 1;
constexpr size_t OUTPUT_Y_INDEX = 0;

static ge::graphStatus InferShapeForAtan2(gert::InferShapeContext* context)
{
    OP_LOGD(context->GetNodeName(), "Begin to do InferShapeForAtan2");
    // kernel 不支持混合 dtype，两个输入必须为相同 dtype
    const auto* x1Desc = context->GetInputDesc(INPUT_X1_INDEX);
    OP_CHECK_NULL_WITH_CONTEXT(context, x1Desc);
    const auto* x2Desc = context->GetInputDesc(INPUT_X2_INDEX);
    OP_CHECK_NULL_WITH_CONTEXT(context, x2Desc);
    OP_CHECK_IF(x1Desc->GetDataType() != x2Desc->GetDataType(),
                OP_LOGE(context->GetNodeName(), "x1 and x2 dtype must be the same"), return GRAPH_FAILED);
    // 对 x1、x2 做广播得到输出 y 的 shape
    return Ops::Base::InferShape4Broadcast(context, INPUT_COUNT);
}

static ge::graphStatus InferDataTypeForAtan2(gert::InferDataTypeContext* context)
{
    const auto x1Dtype = context->GetInputDataType(INPUT_X1_INDEX);
    // y.dtype -> x1.dtype
    context->SetOutputDataType(OUTPUT_Y_INDEX, x1Dtype);
    OP_LOGD(context->GetNodeName(), "End to do InferDataTypeForAtan2");
    return GRAPH_SUCCESS;
}

IMPL_OP_INFERSHAPE(Atan2).InferShape(InferShapeForAtan2).InferDataType(InferDataTypeForAtan2);
} // namespace ops
