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
 * \file angle_v2_graph_infer.cpp
 * \brief angle_v2 operater graph infer resource
 */

#include "register/op_impl_registry.h"
#include "log/log.h"

using namespace ge;
namespace ops {
static constexpr int64_t X_IDX = 0;
static constexpr int64_t Y_IDX = 0;

static ge::graphStatus AngleV2InferDataTypeFunc(gert::InferDataTypeContext* context)
{
    OP_LOGD(context, "Begin to do AngleV2InferDataTypeFunc");

    auto inputDtype = context->GetInputDataType(X_IDX);
    OP_LOGD(context, "x dtype: %s", Ops::Base::ToString(inputDtype).c_str());

    if (inputDtype == ge::DT_FLOAT16) {
        context->SetOutputDataType(Y_IDX, inputDtype);
    } else {
        context->SetOutputDataType(Y_IDX, ge::DT_FLOAT);
    }
    auto outputDataType = context->GetOutputDataType(Y_IDX);
    OP_LOGD(context, "y dtype: %s", Ops::Base::ToString(outputDataType).c_str());
    OP_LOGD(context, "End to do AngleV2InferDataTypeFunc end");
    return GRAPH_SUCCESS;
}

IMPL_OP(AngleV2).InferDataType(AngleV2InferDataTypeFunc);
}; // namespace ops