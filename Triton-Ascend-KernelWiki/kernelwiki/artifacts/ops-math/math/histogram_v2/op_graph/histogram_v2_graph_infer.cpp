/**
 * Copyright (c) 2025-2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

/*!
 * \file histogram_v2_graph_infer.cpp
 * \brief histogram_v2 operater graph infer resource
 */

#include "register/op_impl_registry.h"
#include "log/log.h"

namespace ops {

static ge::graphStatus HistogramV2InferDataTypeFunc(gert::InferDataTypeContext* context)
{
    OP_LOGD(context, "Begin to do HistogramV2InferDataTypeFunc");
    auto inputDtype = context->GetInputDataType(0);
    OP_LOGD(context, "x dtype: %s", Ops::Base::ToString(inputDtype).c_str());

    // y_dtype属性指定输出类型，默认为DT_INT32(3)，可设置为DT_FLOAT(9)
    auto attrs = context->GetAttrs();
    ge::DataType outDtype = ge::DT_INT32;
    if (attrs != nullptr) {
        auto yDtypePtr = attrs->GetAttrPointer<int64_t>(1);
        OP_LOGD(context, "yDtypePtr: %p", yDtypePtr);
        if (yDtypePtr != nullptr) {
            OP_LOGD(context, "yDtypePtr value: %ld", *yDtypePtr);
            if (*yDtypePtr == ge::DT_FLOAT) {
                outDtype = ge::DT_FLOAT;
            }
        }
    }

    OP_LOGD(context, "befer set y dtype: %s", Ops::Base::ToString(context->GetOutputDataType(0)).c_str());
    context->SetOutputDataType(0, outDtype);
    OP_LOGD(context, "after set y dtype: %s", Ops::Base::ToString(context->GetOutputDataType(0)).c_str());
    OP_LOGD(context, "End to do HistogramV2InferDataTypeFunc end");
    return ge::GRAPH_SUCCESS;
}

IMPL_OP(HistogramV2).InferDataType(HistogramV2InferDataTypeFunc);

}; // namespace ops