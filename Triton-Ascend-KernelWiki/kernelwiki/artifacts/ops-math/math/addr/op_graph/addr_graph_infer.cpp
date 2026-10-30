/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include "log/log.h"
#include "register/op_impl_registry.h"

using namespace ge;
namespace ops {

constexpr size_t INPUT_NUM_TWO = 2;
constexpr size_t INPUT_NUM_THREE = 3;
constexpr size_t INPUT_NUM_FOUR = 4;

static ge::graphStatus InferDataType4Addr(gert::InferDataTypeContext* context)
{
    OP_LOGD(context->GetNodeName(), "InferDataType4Addr enter");
    auto x1_data_type = context->GetInputDataType(0);
    auto x2_data_type = context->GetInputDataType(1);
    auto x3_data_type = context->GetInputDataType(INPUT_NUM_TWO);
    auto beta_data_type = context->GetInputDataType(INPUT_NUM_THREE);
    auto alpha_data_type = context->GetInputDataType(INPUT_NUM_FOUR);

    OP_CHECK_IF(
        x1_data_type != x2_data_type || x1_data_type != x3_data_type || x1_data_type != beta_data_type ||
            x1_data_type != alpha_data_type,
        OP_LOGE(context->GetNodeName(), "[InferDataType4Addr] Failed, input datatype must be same."),
        return ge::GRAPH_FAILED);

    context->SetOutputDataType(0, x1_data_type);
    OP_LOGD(context->GetNodeName(), "InferDataType4Addr end");
    return ge::GRAPH_SUCCESS;
}

IMPL_OP(Addr).InferDataType(InferDataType4Addr).InputsDataDependency({INPUT_NUM_THREE, INPUT_NUM_FOUR});

} // namespace ops