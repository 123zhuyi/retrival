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
#include "graph/utils/type_utils.h"

namespace ops {
constexpr size_t EYE_ATTR_DTYPE = 3;
constexpr size_t EYE_OUTPUT_Y = 0;
using namespace Ops::Base;
ge::graphStatus InferDataTypeForEye(gert::InferDataTypeContext *context)
{
    OP_LOGD(context->GetNodeName(), "InferDataTypeForEye start");
    auto attrPtr = context->GetAttrs();
    OP_CHECK_NULL_WITH_CONTEXT(context, attrPtr);
    auto dTypePtr = attrPtr->GetAttrPointer<int64_t>(EYE_ATTR_DTYPE);
    OP_CHECK_NULL_WITH_CONTEXT(context, dTypePtr);
    ge::DataType outDtype = static_cast<ge::DataType>(*dTypePtr);
    std::set<ge::DataType> supportedDtype = {ge::DT_FLOAT, ge::DT_FLOAT16, ge::DT_UINT8,
                                            ge::DT_INT8, ge::DT_INT32, ge::DT_INT16,
                                            ge::DT_BOOL, ge::DT_INT64, ge::DT_BF16};
    auto it = supportedDtype.find(outDtype);
    if (it == supportedDtype.end()) {
        OP_LOGE(context->GetNodeName(), "unsupported dtype[%s].",
                ge::TypeUtils::DataTypeToSerialString(outDtype).c_str());
        return ge::GRAPH_FAILED;
    }
    context->SetOutputDataType(EYE_OUTPUT_Y, outDtype);
    OP_LOGD(context->GetNodeName(), "InferDataTypeForEye end");
    return ge::GRAPH_SUCCESS;
}

IMPL_OP(Eye).InferDataType(InferDataTypeForEye);
}  // namespace ops