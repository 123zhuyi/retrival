/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN " AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include "log/log.h"
#include "register/op_impl_registry.h"
#include "util/shape_util.h"

using namespace ge;
namespace ops {
namespace {
constexpr int64_t GET_SHAPE_MAX_TOTAL_DIM = 128;
constexpr int64_t GET_SHAPE_MAX_DIM_PER_TENSOR = 8;
} // namespace

static ge::graphStatus InferShapeForGetShape(gert::InferShapeContext* context)
{
    auto inputNum = context->GetComputeNodeInputNum();

    if (inputNum == 0) {
        OP_LOGE_FOR_INVALID_VALUE_WITH_REASON(context->GetNodeName(), "x", "0",
                                              "The value of x must be greater than 0");
        return ge::GRAPH_FAILED;
    }

    int64_t totalDimNum = 0;
    for (size_t i = 0; i < inputNum; ++i) {
        auto xShape = context->GetDynamicInputShape(0, i);
        OP_CHECK_NULL_WITH_CONTEXT(context, xShape);
        if (Ops::Base::IsUnknownRank(*xShape)) {
            OP_LOGE_FOR_INVALID_SHAPE_WITH_REASON(context->GetNodeName(), "x", "unknown rank",
                                                  "x cannot be an unknown rank tensor");
            return ge::GRAPH_FAILED;
        }
        auto dimNum = xShape->GetDimNum();
        if (static_cast<int64_t>(dimNum) > GET_SHAPE_MAX_DIM_PER_TENSOR) {
            OP_LOGE_FOR_INVALID_SHAPEDIM(context->GetNodeName(), "x", std::to_string(dimNum).c_str(), "8");
            return ge::GRAPH_FAILED;
        }
        totalDimNum += static_cast<int64_t>(dimNum);
    }

    if (totalDimNum == 0) {
        OP_LOGE_FOR_INVALID_SHAPESIZE(context->GetNodeName(), "y", "0", "greater than 0");
        return ge::GRAPH_FAILED;
    }
    if (totalDimNum > GET_SHAPE_MAX_TOTAL_DIM) {
        OP_LOGE_FOR_INVALID_SHAPESIZE(context->GetNodeName(), "y", std::to_string(totalDimNum).c_str(), "128");
        return ge::GRAPH_FAILED;
    }

    auto yShape = context->GetOutputShape(0);
    OP_CHECK_NULL_WITH_CONTEXT(context, yShape);
    yShape->SetDimNum(1);
    yShape->SetDim(0, totalDimNum);

    return ge::GRAPH_SUCCESS;
}

IMPL_OP_INFERSHAPE(GetShape).InferShape(InferShapeForGetShape);
} // namespace ops
