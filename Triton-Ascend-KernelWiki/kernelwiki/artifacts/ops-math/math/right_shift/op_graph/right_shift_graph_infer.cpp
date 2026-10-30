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
 * \file right_shift_graph_infer.cpp
 * \brief right_shift graph infer
 */

#include "register/op_impl_registry.h"
#include "log/log.h"

using namespace ge;
namespace ops {
static constexpr size_t kInputIndex0 = 0U;
static constexpr size_t kOutputIndex0 = 0U;
static ge::graphStatus InferDataTypeForRightShift(gert::InferDataTypeContext* context)
{
    OP_LOGI(context->GetNodeName(), "Begin to do InferDtypeForRightShift");
    const auto x_data_type = context->GetInputDataType(kInputIndex0);
    return context->SetOutputDataType(kOutputIndex0, x_data_type);
}

IMPL_OP(RightShift).InferDataType(InferDataTypeForRightShift);
} // namespace ops
