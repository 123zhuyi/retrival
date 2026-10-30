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
 * \file unsqueezev2_infershape.cc
 * \brief
 */
#include "runtime/infer_shape_context.h"
#include "register/op_impl_registry.h"
#include "log/log.h"

using namespace ge;
namespace ops {
namespace {
ge::graphStatus SetOutputAsUnknownRank(gert::Shape *y_shape) {
  y_shape->SetDimNum(1U);
  y_shape->SetDim(0U, ge::UNKNOWN_DIM_NUM);
  return ge::GRAPH_SUCCESS;
}

ge::graphStatus InferShapeMainImplForUnsqueeze(const gert::Shape *x_shape, gert::Shape *y_shape,
                                               const gert::TypedContinuousVector<int64_t> *axes) {
  const auto x_dim_num = x_shape->GetDimNum();
  const auto y_dim_num = x_dim_num + axes->GetSize();
  y_shape->SetDimNum(y_dim_num);
  if (y_dim_num > y_shape->kMaxDimNum) {
    OP_LOGE("UnsqueezeOp", "DimNum of output shape is %zu, larger than kMaxDimNum which is %zu!",
                           y_dim_num, y_shape->kMaxDimNum);
    return ge::GRAPH_FAILED;
  }
  for (size_t i = 0U; i < y_dim_num; ++i) {
    y_shape->SetDim(i, 0);
  }
  const int64_t signed_y_dim_num = static_cast<int64_t>(y_dim_num);
  for (size_t i = 0U; i < axes->GetSize(); ++i) {
    const int64_t raw_axis = (axes->GetData())[i];
    const int64_t final_axis = (raw_axis < 0) ? (raw_axis + signed_y_dim_num) : raw_axis;
    if ((final_axis < 0) || (final_axis >= signed_y_dim_num)) {
      OP_LOGE("UnsqueezeOp", "Unsqueeze axis must be in range [-%zu, %zu)!", y_dim_num, y_dim_num);
      return ge::GRAPH_FAILED;
    }
    if (y_shape->GetDim(final_axis) == 1) {
      OP_LOGE("UnsqueezeOp", "Unsqueeze axis repeated!");
      return ge::GRAPH_FAILED;
    }
    y_shape->SetDim(final_axis, 1);
  }

  size_t idx = 0U;
  for (size_t i = 0U; i < y_dim_num; ++i) {
    if (y_shape->GetDim(i) != 1) {
      y_shape->SetDim(i, x_shape->GetDim(idx));
      ++idx;
    }
  }
  return ge::GRAPH_SUCCESS;
}
}  // namespace

static ge::graphStatus InferShapeForUnsqueezeV2(gert::InferShapeContext* context) {
  const auto x_shape = context->GetInputShape(0U);
  auto y_shape = context->GetOutputShape(0U);
  const gert::RuntimeAttrs* attrs = context->GetAttrs();
  if (attrs == nullptr) {
    OP_LOGE(context->GetNodeName(), "attrs pointer must not be nullptr!");
    return ge::GRAPH_FAILED;
  }
  auto axes = attrs->GetListInt(0U);
  OP_CHECK_IF(x_shape == nullptr, OP_LOGE(context, "x_shape is null."), return ge::GRAPH_FAILED);
  OP_CHECK_IF(y_shape == nullptr, OP_LOGE(context, "y_shape is null."), return ge::GRAPH_FAILED);
  OP_CHECK_IF(axes == nullptr, OP_LOGE(context, "axes is null."), return ge::GRAPH_FAILED);
  // solve unknown_rank
  const auto x_dim_num = x_shape->GetDimNum();
  if (x_dim_num == 1U && x_shape->GetDim(0U) == ge::UNKNOWN_DIM_NUM) {
    OP_LOGD(context->GetNodeName(), "Input shape is unkown rank, set output shape as unknown rank.");
    return SetOutputAsUnknownRank(y_shape);
  }
  return InferShapeMainImplForUnsqueeze(x_shape, y_shape, axes);
}

IMPL_OP_INFERSHAPE(UnsqueezeV2).InferShape(InferShapeForUnsqueezeV2);
}  // namespace ops