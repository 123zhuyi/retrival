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
 * \file xdivy_infershape.cpp
 * \brief Xdivy InferShape — broadcast_max(x, y)
 */
#include "register/op_impl_registry.h"
#include "infershape_broadcast_util.h"
#include "log/log.h"
#include "util/shape_util.h"

using namespace ge;

namespace ops {

static ge::graphStatus InferShapeForXdivy(gert::InferShapeContext* context)
{
    const ge::graphStatus status = Ops::Base::InferShape4Broadcast(context);
    if (status == ge::GRAPH_SUCCESS) {
        const gert::Shape* outputShape = context->GetOutputShape(0);
        OP_CHECK_NULL_WITH_CONTEXT(context, outputShape);
        OP_LOGI(context->GetNodeName(), "Xdivy output shape: %s.", Ops::Base::ToString(*outputShape).c_str());
    }
    return status;
}

static ge::graphStatus InferDataTypeForXdivy(gert::InferDataTypeContext* context)
{
    context->SetOutputDataType(0, context->GetInputDataType(0));
    return ge::GRAPH_SUCCESS;
}

IMPL_OP_INFERSHAPE(Xdivy).InferShape(InferShapeForXdivy).InferDataType(InferDataTypeForXdivy);

} // namespace ops
