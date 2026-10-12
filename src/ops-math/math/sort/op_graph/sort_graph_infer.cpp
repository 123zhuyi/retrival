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
 * \file sort_graph_infer.cpp
 * \brief
 */
#include "log/log.h"
#include "register/op_impl_registry.h"

using namespace ge;
namespace ops {
graphStatus InferDataType4Sort(gert::InferDataTypeContext* context)
{
    context->SetOutputDataType(0, context->GetInputDataType(0));
    auto attrs = context->GetAttrs();
    OP_CHECK_NULL_WITH_CONTEXT(context, attrs);
    int64_t y2Dtype = static_cast<int64_t>(ge::DataType::DT_INT32);
    ge::DataType indicesDtype = ge::DT_INT32;
    OP_LOGI(context->GetNodeName(), "The default dtype of output indices is int32.");
    auto out_idx_dtype_ptr = attrs->GetAttrPointer<int64_t>(3);
    if (out_idx_dtype_ptr != nullptr) {
        y2Dtype = *out_idx_dtype_ptr;
        if (y2Dtype == static_cast<int64_t>(ge::DataType::DT_INT64)) {
            OP_LOGI(context->GetNodeName(), "The dtype of output indices is set as int64.");
            indicesDtype = ge::DT_INT64;
        } else {
            if (y2Dtype != static_cast<int64_t>(ge::DataType::DT_INT32)) {
                OP_LOGE_FOR_INVALID_DTYPE(context->GetNodeName(), "indices_dtype",
                                          Ops::Base::ToString(static_cast<ge::DataType>(y2Dtype)).c_str(),
                                          "INT32 or INT64");
                return GRAPH_FAILED;
            }
        }
    }
    context->SetOutputDataType(1, indicesDtype);
    return GRAPH_SUCCESS;
}

IMPL_OP(Sort).InferDataType(InferDataType4Sort);
} // namespace ops
