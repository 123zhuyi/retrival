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
 * \file squeezev2_infershape.cpp
 * \brief
 */
#include "runtime/infer_shape_context.h"
#include "register/op_impl_registry.h"
#include "log/log.h"

using namespace ge;
namespace ops {
namespace {
ge::graphStatus SqueezeForEmptyAxes(const gert::Shape *x_shape, gert::Shape *y_shape) {
  y_shape->SetDimNum(0U);
  for (size_t idx = 0U; idx < x_shape->GetDimNum(); ++idx) {
    if (x_shape->GetDim(idx) != 1) {
      y_shape->AppendDim(x_shape->GetDim(idx));
    }
  }
  return ge::GRAPH_SUCCESS;
}

bool IsUnknownDim(const gert::Shape &shape) {
  for (size_t i = 0U; i < shape.GetDimNum(); ++i) {
    if (shape.GetDim(i) == ge::UNKNOWN_DIM) {
      return true;
    }
  }
  return false;
}

bool IsAxesRangeValid(bool *squeeze_dims, int64_t x_dim_num, const gert::TypedContinuousVector<int64_t> *axes) {
  for (size_t idx = 0U; idx < axes->GetSize(); ++idx) {
    auto raw_axis = axes->GetData()[idx];
    auto final_axis = (raw_axis < 0) ? (raw_axis + x_dim_num) : raw_axis;
    if (final_axis < 0 || final_axis >= x_dim_num) {
      OP_LOGW("Squeeze(v2)", "Squeeze axis[%ld] not in range [-%ld, %ld)", raw_axis, x_dim_num, x_dim_num);
      return false;
    }
    squeeze_dims[final_axis] = true;
  }
  return true;
}

ge::graphStatus SqueezeWithAxes(const gert::Shape *x_shape, const bool *squeeze_dims, gert::Shape *y_shape) {
  y_shape->SetDimNum(0U);
  for (size_t idx = 0U; idx < x_shape->GetDimNum(); ++idx) {
    if (!squeeze_dims[idx]) {
      y_shape->AppendDim(x_shape->GetDim(idx));
    } else {
      const auto x_shape_dim = x_shape->GetDim(idx);
      if (x_shape_dim != ge::UNKNOWN_DIM && x_shape_dim != 1) {
          OP_LOGE(
            "Squeeze op series",
            "SqueezeV2 failed, cannot squeeze dim was not 1 or -1! Axes is %zu, while x_shape[%zu]=%ld.", idx, idx,
            x_shape_dim);
        return ge::GRAPH_FAILED;
      }
    }
  }
  return ge::GRAPH_SUCCESS;
}
}  // namespace

static ge::graphStatus InferShapeForSqueezeV2(gert::InferShapeContext* context) {
  const auto x_shape = context->GetInputShape(0U);
  auto y_shape = context->GetOutputShape(0U);
  const gert::RuntimeAttrs* attrs = context->GetAttrs();
  OP_CHECK_IF(attrs == nullptr, OP_LOGE(context, "attrs is null."), return ge::GRAPH_FAILED);
  auto axes = attrs->GetListInt(0U);
  OP_CHECK_IF(x_shape == nullptr, OP_LOGE(context, "x_shape is null."), return ge::GRAPH_FAILED);
  OP_CHECK_IF(y_shape == nullptr, OP_LOGE(context, "y_shape is null."), return ge::GRAPH_FAILED);
  OP_CHECK_IF(axes == nullptr, OP_LOGE(context, "axes is null."), return ge::GRAPH_FAILED);
  // solve unknown_rank
  auto x_dim_num = x_shape->GetDimNum();
  if (x_dim_num == 1U && x_shape->GetDim(0U) == ge::UNKNOWN_DIM_NUM) {
    OP_LOGD(context, "Input shape is unknown rank!");
    *y_shape = *x_shape;
    return ge::SUCCESS;
  }
  // solve the operator bug for x_shape(three-dimensional) temporarily
  if (x_dim_num == 3U) {
    *y_shape = *x_shape;
    return ge::SUCCESS;
  }
  // solve empty axis
  if (axes->GetSize() == 0U) {
    if (IsUnknownDim(*x_shape)) {
      OP_LOGD(context, "Input shape is unknown and axis is empty. Set output shape as unknown rank.");
      y_shape->SetDimNum(1U);
      y_shape->SetDim(0, ge::UNKNOWN_DIM_NUM);
      return ge::GRAPH_SUCCESS;
    }
    OP_LOGD(context, "Input shape is known and axis is empty. Squeeze all dim 1.");
    return SqueezeForEmptyAxes(x_shape, y_shape);
  }
  // solve V2 special logic
  bool squeeze_dims[gert::Shape::kMaxDimNum] = {false};
  if (!IsAxesRangeValid(squeeze_dims, static_cast<int64_t>(x_shape->GetDimNum()), axes)) {
    OP_LOGW(context, "SqueezeV2 special logic when axis is not valid. Set output shape as input shape.");
    *y_shape = *x_shape;
    return ge::SUCCESS;
  }
  // solve normal squeeze
  return SqueezeWithAxes(x_shape, squeeze_dims, y_shape);
}

IMPL_OP_INFERSHAPE(SqueezeV2).InferShape(InferShapeForSqueezeV2);
}  // namespace ops