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
 * \file add_v2_graph_infer.cpp
 * \brief add_v2 operater graph infer resource
 */
#include "register/op_impl_registry.h"
#include "log/log.h"

using namespace ge;
namespace ops {
static constexpr size_t ADD_V2_IN_X1_IDX = 0;
static constexpr size_t ADD_V2_IN_X2_IDX = 1;
static constexpr size_t ADD_V2_OUT_Y_IDX = 0;

// canonical AddV2 契约（canndev op_graph 原型与 CheckTwoInputDtypeSame Verifier）
// 要求 x1/x2 同 dtype，y 与 x1 同 dtype；本算子不注册异类型组合，故此处不做类型提升。
static ge::graphStatus InferDataTypeAddV2(gert::InferDataTypeContext* context)
{
    OP_LOGD(context->GetNodeName(), "Begin to do InferDataTypeAddV2");

    const ge::DataType x1DataType = context->GetInputDataType(ADD_V2_IN_X1_IDX);
    const ge::DataType x2DataType = context->GetInputDataType(ADD_V2_IN_X2_IDX);
    OP_CHECK_IF(
        x1DataType != x2DataType,
        OP_LOGE_FOR_INVALID_DTYPES_WITH_REASON(
            "AddV2", "x1 and x2", (Ops::Base::ToString(x1DataType) + " and " + Ops::Base::ToString(x2DataType)).c_str(),
            "x1 and x2 must have the same dtype"),
        return ge::GRAPH_FAILED);

    auto ret = context->SetOutputDataType(ADD_V2_OUT_Y_IDX, x1DataType);

    OP_LOGD(context->GetNodeName(), "End to do InferDataTypeAddV2");
    return ret;
}

IMPL_OP(AddV2).InferDataType(InferDataTypeAddV2);

}; // namespace ops
