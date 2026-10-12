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
 * \file tensor_redirect_graph_infer.cpp
 * \brief tensor_redirect operator graph infer resource
 */
#include "register/op_impl_registry.h"
#include "log/log.h"

using namespace ge;
namespace ops {
static constexpr size_t TENSOR_REDIRECT_IN_X_IDX = 0;
static constexpr size_t TENSOR_REDIRECT_OUT_Y_IDX = 0;

// output_x 与 x 同 dtype（same-as-input）
static ge::graphStatus InferDataType4TensorRedirect(gert::InferDataTypeContext* context)
{
    OP_LOGD(context->GetNodeName(), "Begin to do InferDataType4TensorRedirect");
    auto ret = context->SetOutputDataType(TENSOR_REDIRECT_OUT_Y_IDX,
                                          context->GetInputDataType(TENSOR_REDIRECT_IN_X_IDX));
    OP_LOGD(context->GetNodeName(), "End to do InferDataType4TensorRedirect");
    return ret;
}

IMPL_OP(TensorRedirect).InferDataType(InferDataType4TensorRedirect);

} // namespace ops
