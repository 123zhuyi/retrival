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
 * \file diag_ops.cc
 * \brief
 */

#include "infershape_broadcast_util.h"
#include "register/op_impl_registry.h"
#include "log/log.h"

using namespace ge;
namespace ops {

static constexpr size_t DIAG_IN_X_IDX = 0;
static constexpr size_t DIAG_OUT_Y_IDX = 0;
static constexpr size_t INT_DATA_2 = 2;

static ge::graphStatus Infershape4DiagPart(gert::InferShapeContext* context)
{
    OP_LOGD(context->GetNodeName(), "Begin to do DiagPartInfershape.");
    const gert::Shape* input_x_shape = context->GetInputShape(DIAG_IN_X_IDX);
    OP_CHECK_NULL_WITH_CONTEXT(context, input_x_shape);

    int64_t input_to_output_dims_times = 2;
    OP_CHECK_IF(
        input_x_shape->GetDimNum() % 2 != 0,
        OP_LOGE(context->GetNodeName(), "input_x_shape->GetDimNum() %% 2 != 0 is not supported."),
        return ge::GRAPH_FAILED);
    int64_t output_shape_len = input_x_shape->GetDimNum() / input_to_output_dims_times;

    gert::Shape* output_y_shape = context->GetOutputShape(DIAG_OUT_Y_IDX);
    OP_CHECK_NULL_WITH_CONTEXT(context, output_y_shape);

    output_y_shape->SetDimNum(output_shape_len);
    for (int64_t i = 0; i < output_shape_len; i++) {
        output_y_shape->SetDim(i, input_x_shape->GetDim(i));
    }

    OP_LOGD(context->GetNodeName(), "output_y_shape = %s.", Ops::Base::ToString(*output_y_shape).c_str());
    OP_LOGD(context->GetNodeName(), "End to do DiagPartInfershape.");

    return ge::GRAPH_SUCCESS;
}

IMPL_OP_INFERSHAPE(DiagPart).InferShape(Infershape4DiagPart);

} // namespace ops