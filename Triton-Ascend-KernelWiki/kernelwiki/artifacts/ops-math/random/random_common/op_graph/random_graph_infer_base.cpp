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
 * \file random_graph_infer_base.cpp
 * \brief
 */
#include "random_graph_infer_base.h"
namespace ops {
namespace GraphCommon {
ge::graphStatus InferDataTypeByAttr(
    gert::InferDataTypeContext* context, const int32_t dtypeIndex, ge::DataType& OutDtype)
{
    auto* attrs = context->GetAttrs();
    OP_CHECK_NULL_WITH_CONTEXT(context, attrs);

    const int64_t* attrDtype = attrs->GetAttrPointer<int64_t>(dtypeIndex);
    OP_CHECK_NULL_WITH_CONTEXT(context, attrDtype);
    OutDtype = static_cast<ge::DataType>(*attrDtype);
    return ge::GRAPH_SUCCESS;
}

ge::graphStatus CommonInferType(
    gert::InferDataTypeContext* context, int32_t mode, int32_t dtypeIndex,
    const std::vector<OutputSpec>& extraOutputMap, const std::set<ge::DataType>& supportDtype, bool isCheck)
{
    if (context == nullptr) {
        OP_LOGE(context, "Null context pointer");
        return ge::GRAPH_FAILED;
    }

    ge::DataType outDtype = ge::DT_UNDEFINED;
    OP_LOGD(context->GetNodeName(), "Begin to do infer data type.");

    switch (mode) {
        case MODE_ATTR:
            if (InferDataTypeByAttr(context, dtypeIndex, outDtype) != ge::GRAPH_SUCCESS) {
                return ge::GRAPH_FAILED;
            }
            break;
        case MODE_INPUT_EQUAL_OUTPUT:
            outDtype = context->GetInputDataType(0);
            break;
        case MODE_ONE_TYPE:
            if (supportDtype.empty()) {
                OP_LOGE(context->GetNodeName(), "Support_dtype is empty for MODE_ONE_TYPE");
                return ge::GRAPH_FAILED;
            }
            outDtype = *supportDtype.begin();
            break;
        default:
            OP_LOGE(context->GetNodeName(), "Invalid mode value: %d", mode);
            return ge::GRAPH_FAILED;
    }

    OP_CHECK_IF(
        isCheck && supportDtype.count(outDtype) == 0,
        OP_LOGE(context->GetNodeName(), "Unsupported dtype: %s", Ops::Base::ToString(outDtype).c_str()),
        return ge::GRAPH_FAILED);

    context->SetOutputDataType(0, outDtype);

    for (const auto& spec : extraOutputMap) {
        ge::DataType extraOutputType = std::get<2>(spec);
        size_t extraOutputIndex = std::get<1>(spec);
        context->SetOutputDataType(extraOutputIndex, extraOutputType);
    }

    OP_LOGD(context->GetNodeName(), "END to do infer data type.");
    return ge::GRAPH_SUCCESS;
}
} // namespace GraphCommon
} // namespace ops
