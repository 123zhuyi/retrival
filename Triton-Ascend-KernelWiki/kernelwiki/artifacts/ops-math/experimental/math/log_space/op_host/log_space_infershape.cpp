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
 * \file log_space_infershape.cpp
 * \brief LogSpace InferShape 实现
 *
 * LogSpace 无输入 tensor；输出 shape = [steps]，由属性 steps 决定。
 */

#include "register/op_impl_registry.h"
#include "exe_graph/runtime/infer_shape_context.h"
#include "op_common/log/log.h"

using namespace ge;

namespace ops {

// 属性索引常量（与 op_host/log_space_def.cpp / op_host/log_space_tiling.cpp 对齐）
constexpr int ATTR_IDX_START = 0;
constexpr int ATTR_IDX_END = 1;
constexpr int ATTR_IDX_STEPS = 2;
constexpr int ATTR_IDX_BASE = 3;

static ge::graphStatus InferShape4LogSpace(gert::InferShapeContext* context)
{
    auto attrs = context->GetAttrs();
    OP_CHECK_NULL_WITH_CONTEXT(context, attrs);

    const int64_t* stepsPtr = attrs->GetAttrPointer<int64_t>(ATTR_IDX_STEPS);
    OP_CHECK_NULL_WITH_CONTEXT(context, stepsPtr);
    int64_t steps = *stepsPtr;
    // 与 Tiling 对齐：steps<0 直接返回 GRAPH_FAILED（aclnn 入口已做同等校验，此为图模式保险）
    OP_CHECK_IF(steps < 0, OP_LOGE(context, "steps must be >= 0, got %ld", steps), return ge::GRAPH_FAILED);

    gert::Shape* outputShape = context->GetOutputShape(0);
    OP_CHECK_NULL_WITH_CONTEXT(context, outputShape);
    outputShape->SetDimNum(1);
    outputShape->SetDim(0, steps);

    return ge::GRAPH_SUCCESS;
}

IMPL_OP_INFERSHAPE(LogSpace).InferShape(InferShape4LogSpace);

} // namespace ops
