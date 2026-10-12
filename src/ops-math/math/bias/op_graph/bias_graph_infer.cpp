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
 * \file bias_graph_infer.cpp
 * \brief bias operator graph infer resource
 */
#include "register/op_impl_registry.h"
#include "log/log.h"

using namespace ge;
namespace ops {
static constexpr size_t BIAS_IN_X_IDX = 0;
static constexpr size_t BIAS_OUT_Y_IDX = 0;

static ge::graphStatus InferDataTypeBias(gert::InferDataTypeContext* context)
{
    OP_LOGD(context->GetNodeName(), "Begin to do InferDataTypeBias");

    const ge::DataType xDataType = context->GetInputDataType(BIAS_IN_X_IDX);
    auto ret = context->SetOutputDataType(BIAS_OUT_Y_IDX, xDataType);

    OP_LOGD(context->GetNodeName(), "End to do InferDataTypeBias");
    return ret;
}

IMPL_OP(Bias).InferDataType(InferDataTypeBias);
}; // namespace ops
