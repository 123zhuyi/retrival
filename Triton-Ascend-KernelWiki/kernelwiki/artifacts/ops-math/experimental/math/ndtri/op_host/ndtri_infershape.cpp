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
/*!
 * \file ndtri_infershape.cpp
 * \brief Ndtri 算子形状推导实现
 *
 * 输出 out 形状 == 输入 self 形状。
 */

#include "register/op_impl_registry.h"
#include "exe_graph/runtime/infer_shape_context.h"
#include "op_common/log/log.h"

using namespace ge;

namespace ops {

static constexpr size_t IDX_SELF = 0;
static constexpr size_t OUT_OUT = 0;

static ge::graphStatus InferShape4Ndtri(gert::InferShapeContext* context)
{
    const gert::Shape* selfShape = context->GetInputShape(IDX_SELF);
    OP_CHECK_NULL_WITH_CONTEXT(context, selfShape);
    gert::Shape* outShape = context->GetOutputShape(OUT_OUT);
    OP_CHECK_NULL_WITH_CONTEXT(context, outShape);
    *outShape = *selfShape;
    return ge::GRAPH_SUCCESS;
}

IMPL_OP_INFERSHAPE(Ndtri).InferShape(InferShape4Ndtri);

} // namespace ops
