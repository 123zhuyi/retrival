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
 * \file square_sum_all_graph_infer.cpp
 * \brief square_sum_all operator graph infer resource
 */
#include "register/op_impl_registry.h"
#include "log/log.h"

using namespace ge;
namespace ops {
static constexpr size_t SQUARE_SUM_ALL_OUT_Y1_IDX = 0;
static constexpr size_t SQUARE_SUM_ALL_OUT_Y2_IDX = 1;

// y1/y2 恒为 float32：算子只受理 float32 输入，两路平方和也只以 float32 输出
static ge::graphStatus InferDataType4SquareSumAll(gert::InferDataTypeContext* context)
{
    OP_LOGD(context->GetNodeName(), "Begin to do InferDataType4SquareSumAll");
    auto ret = context->SetOutputDataType(SQUARE_SUM_ALL_OUT_Y1_IDX, ge::DT_FLOAT);
    if (ret != ge::GRAPH_SUCCESS) {
        return ret;
    }
    ret = context->SetOutputDataType(SQUARE_SUM_ALL_OUT_Y2_IDX, ge::DT_FLOAT);
    OP_LOGD(context->GetNodeName(), "End to do InferDataType4SquareSumAll");
    return ret;
}

IMPL_OP(SquareSumAll).InferDataType(InferDataType4SquareSumAll);

} // namespace ops
