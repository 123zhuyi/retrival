/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

/**
 * NOTE: Portions of this code were AI-generated and have been
 * technically reviewed for functional accuracy and security
 */

/**
 * \file population_count_infershape.cpp
 * \brief PopulationCount InferShape / InferDataType
 *
 * Semantics:
 *   - y.shape = x.shape (element-wise, no broadcast)
 *   - y.dtype = UINT8 (fixed, independent of x.dtype)
 */

#include "register/op_impl_registry.h"
#include "exe_graph/runtime/infer_shape_context.h"
#include "op_common/log/log.h"

using namespace ge;

namespace ops {

static ge::graphStatus InferShape4PopulationCount(gert::InferShapeContext* context)
{
    const gert::Shape* input_shape = context->GetInputShape(0);
    OP_CHECK_NULL_WITH_CONTEXT(context, input_shape);

    gert::Shape* output_shape = context->GetOutputShape(0);
    OP_CHECK_NULL_WITH_CONTEXT(context, output_shape);

    // Shape passthrough: y.shape = x.shape
    *output_shape = *input_shape;

    return ge::GRAPH_SUCCESS;
}

static ge::graphStatus InferDataType4PopulationCount(gert::InferDataTypeContext* context)
{
    // y.dtype fixed to UINT8, independent of x.dtype
    context->SetOutputDataType(0, ge::DT_UINT8);
    return ge::GRAPH_SUCCESS;
}

IMPL_OP_INFERSHAPE(PopulationCount)
    .InferShape(InferShape4PopulationCount)
    .InferDataType(InferDataType4PopulationCount);

} // namespace ops
