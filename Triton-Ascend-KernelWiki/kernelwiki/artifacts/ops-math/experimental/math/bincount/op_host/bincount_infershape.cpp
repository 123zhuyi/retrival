/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under
 * the terms and conditions of CANN Open Software License Agreement Version 2.0
 * (the "License"). Please refer to the License for details. You may not use
 * this file except in compliance with the License. THIS SOFTWARE IS PROVIDED ON
 * AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS
 * FOR A PARTICULAR PURPOSE. See LICENSE in the root of the software repository
 * for the full text of the License.
 */

/*!
 * \file bincount_infershape.cpp
 * \brief bincount infershape
 */

#include "log/log.h"
#include "register/op_impl_registry.h"
using namespace ge;
namespace ops {
static constexpr int64_t IDX_0 = 0;
static ge::graphStatus InferShapeForBincount(gert::InferShapeContext* context)
{
    OP_CHECK_IF(context == nullptr, OP_LOGE(context, "context is nullptr"), return ge::GRAPH_FAILED);
    OP_LOGD(context->GetNodeName(), "Begin to do InferShapeForBincount");
    gert::Shape* outShape = context->GetOutputShape(IDX_0);
    OP_CHECK_NULL_WITH_CONTEXT(context, outShape);
    // bincount 输出为 1 维；负数不做索引偏移，由 kernel 运行期检查并报错。
    // 基线：aclnn 调用时 out 由调用方按规则预分配，此处置为 1 维动态(-1)。
    outShape->SetDimNum(1);
    outShape->SetDim(IDX_0, -1);
    OP_LOGD(context->GetNodeName(), "End to do InferShapeForBincount");
    return GRAPH_SUCCESS;
}
IMPL_OP_INFERSHAPE(Bincount).InferShape(InferShapeForBincount);
} // namespace ops
