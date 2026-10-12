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
 * \file fused_mul_add_infershape.cpp
 * \brief
 */
#include "register/op_impl_registry.h"
#include "op_common/op_host/infershape_broadcast_util.h"
#include "log/log.h"

using namespace ge;
namespace ops {
constexpr size_t FUSED_MUL_ADD_INPUT_X1_INDEX = 0;
constexpr size_t FUSED_MUL_ADD_OUTPUT_Y_INDEX = 0;

static ge::graphStatus InferShape4FusedMulAdd(gert::InferShapeContext* context)
{
    OP_LOGD(context->GetNodeName(), "Begin to do InferShape4FusedMulAdd in ops-math");
    const size_t INPUT_NUM_THREE = 3;
    return Ops::Base::InferShape4Broadcast(context, INPUT_NUM_THREE);
}

static ge::graphStatus InferDataType4FusedMulAdd(gert::InferDataTypeContext* context)
{
    OP_LOGD(context->GetNodeName(), "Begin to do InferDataType4FusedMulAdd in ops-math");
    // Align with op_proto: output dtype is the same as input x1.
    context->SetOutputDataType(FUSED_MUL_ADD_OUTPUT_Y_INDEX, context->GetInputDataType(FUSED_MUL_ADD_INPUT_X1_INDEX));
    return ge::GRAPH_SUCCESS;
}
IMPL_OP_INFERSHAPE(FusedMulAdd).InferShape(InferShape4FusedMulAdd).InferDataType(InferDataType4FusedMulAdd);
} // namespace ops
