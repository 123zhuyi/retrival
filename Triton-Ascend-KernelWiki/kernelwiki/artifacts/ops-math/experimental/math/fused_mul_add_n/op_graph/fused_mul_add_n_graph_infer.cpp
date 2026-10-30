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
 * \file fused_mul_add_n_graph_infer.cpp
 * \brief FusedMulAddN graph-mode dtype inference. y = x1 * x3[0] + x2 keeps the
 *        input dtype (x1/x2/x3/y share the same dtype), so the output dtype mirrors x1's.
 */
#include "register/op_impl_registry.h"
#include "log/log.h"

namespace ops {
using namespace ge;

static constexpr int64_t IDX_0 = 0;

static ge::graphStatus InferDataTypeFusedMulAddN(gert::InferDataTypeContext* context)
{
    OP_LOGD(context->GetNodeName(), "Begin to do InferDataTypeFusedMulAddN");

    // 输出 dtype 与输入 x1 dtype 一致（x1/x2/x3/y 同 dtype，y = x1 * x3[0] + x2）。
    ge::DataType inputDtype = context->GetInputDataType(IDX_0);
    context->SetOutputDataType(IDX_0, inputDtype);

    OP_LOGD(context->GetNodeName(), "End to do InferDataTypeFusedMulAddN");
    return GRAPH_SUCCESS;
}

IMPL_OP(FusedMulAddN).InferDataType(InferDataTypeFusedMulAddN);

}; // namespace ops
