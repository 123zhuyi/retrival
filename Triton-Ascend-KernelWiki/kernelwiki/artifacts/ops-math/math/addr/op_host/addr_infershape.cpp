/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include "log/log.h"
#include "register/op_impl_registry.h"

using namespace Ops::Base;
using namespace ge;
namespace ops {

constexpr size_t INPUT_NUM_TWO = 2;
constexpr size_t INPUT_NUM_THREE = 3;
constexpr size_t INPUT_NUM_FOUR = 4;
constexpr size_t DIM_NUM_TWO = 2;
constexpr int64_t UNKNOWN_RANK_DIM_VALUE_ = -2LL;
/*
 * str cat util function
 * param[in] params need concat to string
 * return concatted string
 */
template <typename T, typename... Ts>
std::string ConcatString(const T& arg, const Ts&... argLeft)
{
    std::ostringstream oss;
    oss << arg;
    oss << ConcatString(argLeft...);
    return oss.str();
}

static inline bool IsUnknownRank(const gert::Shape* check_shape)
{
    return check_shape->GetDimNum() == 1 && check_shape->GetDim(0) == UNKNOWN_RANK_DIM_VALUE_;
}

static inline ge::graphStatus SetUnknownRank(gert::Shape* output_shape)
{
    OP_CHECK_IF(
        output_shape == nullptr, OP_LOGD("SetUnknownRank", "the output_shape is nullptr, return unsuccess"),
        return ge::GRAPH_FAILED);
    output_shape->SetDimNum(0);
    output_shape->AppendDim(UNKNOWN_RANK_DIM_VALUE_);

    OP_LOGD("SetUnknownRank", "set unknown rank = -2, output = %s", ToString(*output_shape).c_str());
    return ge::GRAPH_SUCCESS;
}

static ge::graphStatus InferShape4Addr(gert::InferShapeContext* context)
{
    OP_LOGD(context->GetNodeName(), "InferShape4Addr enter");
    auto x1_shape = context->GetInputShape(0);
    OP_CHECK_NULL_WITH_CONTEXT(context, x1_shape);
    auto x2_shape = context->GetInputShape(1);
    OP_CHECK_NULL_WITH_CONTEXT(context, x2_shape);
    auto x3_shape = context->GetInputShape(INPUT_NUM_TWO);
    OP_CHECK_NULL_WITH_CONTEXT(context, x3_shape);
    auto out_shape = context->GetOutputShape(0);
    OP_CHECK_NULL_WITH_CONTEXT(context, out_shape);

    if (IsUnknownRank(x2_shape) || IsUnknownRank(x3_shape)) {
        OP_LOGD(context->GetNodeName(), "the input shape is [-2], set output shape is [-2]!");
        SetUnknownRank(out_shape);
        return ge::GRAPH_SUCCESS;
    }

    size_t x1_dim = x1_shape->GetDimNum();
    size_t x2_dim = x2_shape->GetDimNum();
    size_t x3_dim = x3_shape->GetDimNum();
    
    OP_CHECK_IF(
        (x1_dim != DIM_NUM_TWO && x1_dim != 1) || x2_dim != 1 || x3_dim != 1,
        OP_LOGE(
            context->GetNodeName(), "check input dim failed. x1: %s, x2: %s, x3: %s", ToString(*x1_shape).c_str(),
            ToString(*x2_shape).c_str(), ToString(*x3_shape).c_str()),
        return ge::GRAPH_FAILED);
    out_shape->SetDimNum(DIM_NUM_TWO);
    out_shape->SetDim(0, x2_shape->GetDim(0));
    out_shape->SetDim(1, x3_shape->GetDim(0));
    OP_LOGD(context->GetNodeName(), "Output shape is [%ld, %ld]", out_shape->GetDim(0), out_shape->GetDim(1));
    return ge::GRAPH_SUCCESS;
}

IMPL_OP_INFERSHAPE(Addr).InferShape(InferShape4Addr);
} // namespace ops