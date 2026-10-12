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
 * \file zeta_infershape.cpp
 * \brief InferShape for Zeta operator:
 *        z.dtype = x.dtype, z.shape = broadcast(x, q)
 */
#include <algorithm>
#include <utility>
#include <vector>

#include "log/log.h"
#include "register/op_impl_registry.h"
#include "op_host/infershape_broadcast_util.h"

using namespace ge;
namespace ops {

static constexpr size_t INPUT_X_INDEX = 0;
static constexpr size_t INPUT_Q_INDEX = 1;
static constexpr size_t OUTPUT_Z_INDEX = 0;

static void ZetaAddToOutputRange(std::vector<std::pair<int64_t, int64_t>>& out_range,
                                 const std::pair<int64_t, int64_t>& shape_range_x,
                                 const std::pair<int64_t, int64_t>& shape_range_q)
{
    // first_range == max first
    int64_t first_range = (shape_range_x.first * shape_range_q.first == 0) ?
                              0 :
                              std::max(shape_range_x.first, shape_range_q.first);

    if (shape_range_x.second * shape_range_q.second == -1) {
        out_range.push_back(std::pair<int64_t, int64_t>(first_range, -1));
    } else if ((shape_range_x.first == 0 || shape_range_x.first == 1) &&
               (shape_range_q.first == 0 || shape_range_q.first == 1)) {
        // two range.first just be 0 or 1, second_range == max second
        int64_t second_range = (shape_range_x.second == -1 || shape_range_q.second == -1) ?
                                   -1 :
                                   std::max(shape_range_x.second, shape_range_q.second);
        out_range.push_back(std::pair<int64_t, int64_t>(first_range, second_range));
    } else if (shape_range_x.first == 1 || shape_range_q.first == 1) {
        // one shape size maybe 1, so will support broadcast
        int64_t second_range = shape_range_x.first == 1 ? shape_range_q.second : shape_range_x.second;
        out_range.push_back(std::pair<int64_t, int64_t>(first_range, second_range));
    } else if (shape_range_x.first == 0 || shape_range_q.first == 0) {
        // one shape size maybe 0, so will support broadcast
        int64_t second_range = shape_range_x.first == 0 ? shape_range_q.second : shape_range_x.second;
        out_range.push_back(std::pair<int64_t, int64_t>(first_range, second_range));
    } else {
        // no 0 and 1 in range.first, mean no broadcast for range
        // get intersect range
        int64_t second_range = std::min(shape_range_x.second, shape_range_q.second);
        second_range = (shape_range_x.second == -1 || shape_range_q.second == -1) ?
                           std::max(shape_range_x.second, shape_range_q.second) :
                           second_range;
        out_range.push_back(std::pair<int64_t, int64_t>(first_range, second_range));
    }
}

static ge::graphStatus InferShapeRange4Zeta(gert::InferShapeRangeContext* context)
{
    OP_LOGD(context->GetNodeName(), "Begin to do Zeta InferShapeRange");
    auto x_shape_range = context->GetInputShapeRange(INPUT_X_INDEX);
    OP_CHECK_NULL_WITH_CONTEXT(context, x_shape_range);
    auto x_shape_range_max = x_shape_range->GetMax();
    auto x_shape_range_min = x_shape_range->GetMin();
    OP_CHECK_NULL_WITH_CONTEXT(context, x_shape_range_max);
    OP_CHECK_NULL_WITH_CONTEXT(context, x_shape_range_min);

    auto q_shape_range = context->GetInputShapeRange(INPUT_Q_INDEX);
    OP_CHECK_NULL_WITH_CONTEXT(context, q_shape_range);
    auto q_shape_range_max = q_shape_range->GetMax();
    auto q_shape_range_min = q_shape_range->GetMin();
    OP_CHECK_NULL_WITH_CONTEXT(context, q_shape_range_max);
    OP_CHECK_NULL_WITH_CONTEXT(context, q_shape_range_min);

    auto out_shape_range = context->GetOutputShapeRange(OUTPUT_Z_INDEX);
    OP_CHECK_NULL_WITH_CONTEXT(context, out_shape_range);
    auto out_shape_range_max = out_shape_range->GetMax();
    auto out_shape_range_min = out_shape_range->GetMin();
    OP_CHECK_NULL_WITH_CONTEXT(context, out_shape_range_max);
    OP_CHECK_NULL_WITH_CONTEXT(context, out_shape_range_min);

    std::vector<std::pair<int64_t, int64_t>> shape_range_x;
    std::vector<std::pair<int64_t, int64_t>> shape_range_q;
    int64_t x_dim = x_shape_range_max->GetDimNum();
    int64_t q_dim = q_shape_range_max->GetDimNum();
    int64_t min_dim = x_dim < q_dim ? x_dim : q_dim;
    int64_t max_dim = x_dim < q_dim ? q_dim : x_dim;
    for (int64_t i = 0; i < min_dim; i++) {
        shape_range_x.push_back(
            std::pair<int64_t, int64_t>(x_shape_range_min->GetDim(i), x_shape_range_max->GetDim(i)));
        shape_range_q.push_back(
            std::pair<int64_t, int64_t>(q_shape_range_min->GetDim(i), q_shape_range_max->GetDim(i)));
    }

    // 低维对齐：较短的一侧在头部补 (1,1)
    if (min_dim < x_dim) {
        for (int64_t i = min_dim; i < x_dim; i++) {
            shape_range_x.push_back(
                std::pair<int64_t, int64_t>(x_shape_range_min->GetDim(i), x_shape_range_max->GetDim(i)));
            shape_range_q.insert(shape_range_q.begin(), std::pair<int64_t, int64_t>(1, 1));
        }
    } else {
        for (int64_t i = min_dim; i < q_dim; i++) {
            shape_range_x.insert(shape_range_x.begin(), std::pair<int64_t, int64_t>(1, 1));
            shape_range_q.push_back(
                std::pair<int64_t, int64_t>(q_shape_range_min->GetDim(i), q_shape_range_max->GetDim(i)));
        }
    }

    std::vector<std::pair<int64_t, int64_t>> out_range;
    out_shape_range_min->SetDimNum(max_dim);
    out_shape_range_max->SetDimNum(max_dim);
    for (int64_t i = 0; i < max_dim; i++) {
        ZetaAddToOutputRange(out_range, shape_range_x[i], shape_range_q[i]);
        out_shape_range_min->SetDim(i, out_range[i].first);
        out_shape_range_max->SetDim(i, out_range[i].second);
    }

    OP_LOGD(context->GetNodeName(), "End to do Zeta InferShapeRange");
    return ge::GRAPH_SUCCESS;
}

static ge::graphStatus InferShape4Zeta(gert::InferShapeContext* context)
{
    OP_LOGD(context->GetNodeName(), "Begin to do Zeta InferShape");
    // 对 x、q 做广播得到 z 的 shape
    return Ops::Base::InferShape4Broadcast(context);
}

static ge::graphStatus InferDataType4Zeta(gert::InferDataTypeContext* context)
{
    // z.dtype ← x.dtype
    context->SetOutputDataType(OUTPUT_Z_INDEX, context->GetInputDataType(INPUT_X_INDEX));
    OP_LOGD(context->GetNodeName(), "End to do Zeta InferDataType");
    return GRAPH_SUCCESS;
}

IMPL_OP_INFERSHAPE(Zeta)
    .InferShape(InferShape4Zeta)
    .InferShapeRange(InferShapeRange4Zeta)
    .InferDataType(InferDataType4Zeta);

} // namespace ops
