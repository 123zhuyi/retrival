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
 * \file infershape_elewise_util.cpp
 * \brief
 */
#include "op_common/op_host/util/shape_util.h"
#include "op_common/log/log.h"
#include "op_common/op_host/infershape_elewise_util.h"

namespace Ops {
namespace Base {
using namespace ge;

ge::graphStatus InferShape4Elewise(gert::InferShapeContext* context)
{
    auto inShape = context->GetInputShape(0);
    OP_CHECK_NULL_WITH_CONTEXT(context, inShape);
    auto outShape = context->GetOutputShape(0);
    OP_CHECK_NULL_WITH_CONTEXT(context, outShape);

    if (IsUnknownRank(*inShape)) {
        OP_LOGD(context->GetNodeName(), "input shape is UnknownRank, set output shape to (-2, )");
        SetUnknownRank(*outShape);
        return ge::GRAPH_SUCCESS;
    }

    *outShape = *inShape;
    return ge::GRAPH_SUCCESS;
}

} // namespace Base
} // namespace Ops