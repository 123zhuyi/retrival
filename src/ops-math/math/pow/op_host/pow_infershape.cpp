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
 * \file pow_infershape.cpp
 * \brief
 */
#include "log/log.h"
#include "register/op_impl_registry.h"
#include "op_host/infershape_broadcast_util.h"

namespace ops {

using namespace Ops::Base;

static void PowAddToOutputRange(std::vector<std::pair<int64_t, int64_t>>& out_range,
                             const std::pair<int64_t, int64_t>& shape_range_x,
                             const std::pair<int64_t, int64_t>& shape_range_y) {
  // first_range == max first
  int64_t first_range = (shape_range_x.first * shape_range_y.first == 0)
                        ? 0 : std::max(shape_range_x.first, shape_range_y.first);

  if (shape_range_x.second * shape_range_y.second == -1) {
    out_range.push_back(std::pair<int64_t, int64_t>(first_range, -1));
  } else if ((shape_range_x.first == 0 || shape_range_x.first == 1) && \
             (shape_range_y.first == 0 || shape_range_y.first == 1)) {
    // two range.first jsut be 0 or 1, second_range == max second
    int64_t second_range = (shape_range_x.second == -1 || shape_range_y.second == -1)
                            ? -1 : std::max(shape_range_x.second, shape_range_y.second);
    out_range.push_back(std::pair<int64_t, int64_t>(first_range, second_range));
  } else if (shape_range_x.first == 1 || shape_range_y.first == 1) {
    // one shape size maybe 1, so will support broadcast
    int64_t second_range = shape_range_x.first == 1 ? shape_range_y.second : shape_range_x.second;
    out_range.push_back(std::pair<int64_t, int64_t>(first_range, second_range));
  } else if (shape_range_x.first == 0 || shape_range_y.first == 0) {
    // one shape size maybe 1, so will support broadcast
    int64_t second_range = shape_range_x.first == 0 ? shape_range_y.second : shape_range_x.second;
    out_range.push_back(std::pair<int64_t, int64_t>(first_range, second_range));
  } else {
    // no 0 and 1 in range.first, mean no broadcast for range
    // get intersect range
    int64_t second_range = std::min(shape_range_x.second, shape_range_y.second);
    second_range = (shape_range_x.second == -1 || shape_range_y.second == -1)
                    ? std::max(shape_range_x.second, shape_range_y.second) : second_range;
    out_range.push_back(std::pair<int64_t, int64_t>(first_range, second_range));
  }
}

ge::graphStatus InferShapeRange4Pow(gert::InferShapeRangeContext* context)
{
  auto x1_shape_range = context->GetInputShapeRange(0);
  OP_CHECK_NULL_WITH_CONTEXT(context, x1_shape_range);
  auto x1_shape_range_max = x1_shape_range->GetMax();
  auto x1_shape_range_min = x1_shape_range->GetMin();
  OP_CHECK_NULL_WITH_CONTEXT(context, x1_shape_range_max);
  OP_CHECK_NULL_WITH_CONTEXT(context, x1_shape_range_min);

  auto x2_shape_range = context->GetInputShapeRange(1);
  OP_CHECK_NULL_WITH_CONTEXT(context, x2_shape_range);
  auto x2_shape_range_max = x2_shape_range->GetMax();
  auto x2_shape_range_min = x2_shape_range->GetMin();
  OP_CHECK_NULL_WITH_CONTEXT(context, x2_shape_range_max);
  OP_CHECK_NULL_WITH_CONTEXT(context, x2_shape_range_min);

  auto out_shape_range = context->GetOutputShapeRange(0);
  OP_CHECK_NULL_WITH_CONTEXT(context, out_shape_range);
  auto out_shape_range_max = out_shape_range->GetMax();
  auto out_shape_range_min = out_shape_range->GetMin();
  OP_CHECK_NULL_WITH_CONTEXT(context, out_shape_range_max);
  OP_CHECK_NULL_WITH_CONTEXT(context, out_shape_range_min);

  std::vector<std::pair<int64_t, int64_t>> shape_range_x1;
  std::vector<std::pair<int64_t, int64_t>> shape_range_x2;
  int64_t x1_dim = x1_shape_range_max->GetDimNum();
  int64_t x2_dim = x2_shape_range_max->GetDimNum();
  int64_t min_dim = x1_dim < x2_dim ? x1_dim : x2_dim;
  int64_t max_dim = x1_dim < x2_dim ? x2_dim : x1_dim;
  for (int64_t i = 0; i < min_dim; i++) {
    shape_range_x1.push_back(std::pair<int64_t, int64_t>(x1_shape_range_min->GetDim(i), x1_shape_range_max->GetDim(i)));
    shape_range_x2.push_back(std::pair<int64_t, int64_t>(x2_shape_range_min->GetDim(i), x2_shape_range_max->GetDim(i)));
  }

  if (min_dim < x1_dim) {
    for (int64_t i = min_dim; i < max_dim; i++) {
      shape_range_x1.push_back(std::pair<int64_t, int64_t>(x1_shape_range_min->GetDim(i), x1_shape_range_max->GetDim(i)));
      shape_range_x2.insert(shape_range_x2.begin(), std::pair<int64_t, int64_t>(1, 1));
    }
  } else {
    for (int64_t i = min_dim; i < max_dim; i++) {
      shape_range_x1.insert(shape_range_x1.begin(), std::pair<int64_t, int64_t>(1, 1));
      shape_range_x2.push_back(std::pair<int64_t, int64_t>(x2_shape_range_min->GetDim(i), x2_shape_range_max->GetDim(i)));
    }
  }

  std::vector<std::pair<int64_t, int64_t>> out_range;
  out_shape_range_min->SetDimNum(max_dim);
  out_shape_range_max->SetDimNum(max_dim);
  for (int64_t i = 0; i < max_dim; i++) {
    PowAddToOutputRange(out_range, shape_range_x1[i], shape_range_x2[i]);
    out_shape_range_min->SetDim(i, out_range[i].first);
    out_shape_range_max->SetDim(i, out_range[i].second);
  }

  return ge::GRAPH_SUCCESS;
}

static ge::graphStatus InferShapeForBroadcast(gert::InferShapeContext* context) {
  auto in_shape1 = context->GetInputShape(0);
  OP_CHECK_NULL_WITH_CONTEXT(context, in_shape1);
  auto in_shape2 = context->GetInputShape(1);
  OP_CHECK_NULL_WITH_CONTEXT(context, in_shape2);
  auto out_shape = context->GetOutputShape(0);
  OP_CHECK_NULL_WITH_CONTEXT(context, out_shape);

  OP_CHECK_IF(!BroadcastShape(in_shape1, in_shape2, out_shape),
           OP_LOGE(context->GetNodeName(), "shape %s and %s cannot broadcast!", ToString(*in_shape2).c_str(), ToString(*in_shape1).c_str()),
           return ge::GRAPH_FAILED);

  return ge::GRAPH_SUCCESS;
}

IMPL_OP_INFERSHAPE(Pow).InferShape(InferShapeForBroadcast).InferShapeRange(InferShapeRange4Pow);
}  // namespace ops
