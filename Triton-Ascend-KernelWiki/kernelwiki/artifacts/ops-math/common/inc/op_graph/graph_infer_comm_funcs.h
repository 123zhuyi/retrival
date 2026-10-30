/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#ifndef MATH_COMMON_OP_GRAPH_GRAPH_INFER_COMM_FUNCS_H
#define MATH_COMMON_OP_GRAPH_GRAPH_INFER_COMM_FUNCS_H

#include "log/log.h"
#include "graph/types.h"
#include "exe_graph/runtime/infer_datatype_context.h"

namespace ops {

/**
 * data type 推导实现，根据 input 推导 output
 * @tparam OUTPUT_IDX 输出索引，默认为 0
 * @tparam INPUT_IDX 输入索引，默认为 0
 */
template <size_t OUTPUT_IDX = 0, size_t INPUT_IDX = 0>
ge::graphStatus InferDataTypeOutputSameAsInput(gert::InferDataTypeContext* context)
{
    OP_CHECK_NULL_WITH_CONTEXT(context, context);
    OP_LOGD(context, "the data type of output(idx=%lu) is same as that of input(idx=%lu)", OUTPUT_IDX, INPUT_IDX);
    return context->SetOutputDataType(OUTPUT_IDX, context->GetInputDataType(INPUT_IDX));
}
} // namespace ops

#endif
