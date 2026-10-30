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
#include "util/shape_util.h"
using namespace ge;
static constexpr int OUTPUT_VALUES_INDEX = 0;
static constexpr int OUTPUT_INDICES_INDEX = 1;
static constexpr int INPUT_X_INDEX = 0;
static constexpr int DATATYPE_INDEX = 3;
namespace ops {
static graphStatus InferDataType4TopKV2(gert::InferDataTypeContext* context)
{
    OP_LOGD(context->GetNodeName(), "Begin to do InferDataType4TopKV2");
    context->SetOutputDataType(OUTPUT_VALUES_INDEX, context->GetInputDataType(INPUT_X_INDEX));
    auto attrs = context->GetAttrs();
    OP_CHECK_NULL_WITH_CONTEXT(context, attrs);
    int64_t raw_indices_dtype = static_cast<int64_t>(ge::DataType::DT_INT32);
    ge::DataType indicesDtype = ge::DT_INT32;
    OP_LOGI(context->GetNodeName(), "The default dtype of output indices is int32.");
    auto indices_dtype_ptr = attrs->GetAttrPointer<int64_t>(DATATYPE_INDEX);
    if (indices_dtype_ptr != nullptr) {
        raw_indices_dtype = *indices_dtype_ptr;
        if (raw_indices_dtype == static_cast<int64_t>(ge::DataType::DT_INT64)) {
            OP_LOGI(context->GetNodeName(), "The dtype of output indices is set as int64.");
            indicesDtype = ge::DT_INT64;
        } else if (raw_indices_dtype != static_cast<int64_t>(ge::DataType::DT_INT32)) {
            OP_LOGE_FOR_INVALID_DTYPE(context->GetNodeName(), "indices_dtype",
                                      Ops::Base::ToString(static_cast<ge::DataType>(raw_indices_dtype)).c_str(),
                                      "INT32 or INT64");
            return GRAPH_FAILED;
        }
    }
    context->SetOutputDataType(OUTPUT_INDICES_INDEX, indicesDtype);
    OP_LOGD(context->GetNodeName(), "End to do InferDataType4TopKV2");
    return GRAPH_SUCCESS;
}

IMPL_OP(TopKV2).InferDataType(InferDataType4TopKV2);
} // namespace ops
