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
 * \file squeezev3_infershape.cpp
 * \brief
 */
#include "runtime/infer_shape_context.h"
#include "register/op_impl_registry.h"
#include "log/log.h"

using namespace ge;
namespace ops {
namespace {
bool IsUnknownRank(const gert::Shape &shape) {
  return (shape.GetDimNum() == 1U && shape.GetDim(0U) == ge::UNKNOWN_DIM_NUM);
}

bool IsUnknownDim(const gert::Shape &shape) {
  for (size_t i = 0U; i < shape.GetDimNum(); ++i) {
    if (shape.GetDim(i) == ge::UNKNOWN_DIM) {
      return true;
    }
  }
  return false;
}

ge::graphStatus SqueezeForEmptyAxes(const gert::Shape *x_shape, gert::Shape *y_shape) {
  y_shape->SetDimNum(0U);
  for (size_t idx = 0U; idx < x_shape->GetDimNum(); ++idx) {
    if (x_shape->GetDim(idx) != 1) {
      y_shape->AppendDim(x_shape->GetDim(idx));
    }
  }
  return ge::GRAPH_SUCCESS;
}

ge::graphStatus SqueezeWithAxes(const gert::Shape *x_shape, const bool *squeeze_dims, gert::Shape *y_shape) {
  y_shape->SetDimNum(0U);
  for (size_t idx = 0U; idx < x_shape->GetDimNum(); ++idx) {
    if (!squeeze_dims[idx]) {
      y_shape->AppendDim(x_shape->GetDim(idx));
    } else {
      const auto x_shape_dim = x_shape->GetDim(idx);
      if (x_shape_dim != 1 && x_shape_dim != ge::UNKNOWN_DIM) {
        OP_LOGE(
          "Squeeze op series",
          "SqueezeV3 failed, cannot squeeze dim was not 1 or -1! Axes is %zu, while x_shape[%zu]=%ld.", idx, idx,
          x_shape_dim);
        return ge::GRAPH_FAILED;
      }
    }
  }
  return ge::GRAPH_SUCCESS;
}

ge::graphStatus GetSqueezeDimsWhenAxesRangeValid(const gert::Shape* x_shape, const int64_t* axes_data,
                                                 const size_t axes_size, bool* squeeze_dims) {
  const auto x_dim_num = static_cast<int64_t>(x_shape->GetDimNum());
  for (size_t idx = 0UL; idx < axes_size; ++idx) {
    int64_t raw_axis = axes_data[idx];
    const auto rectified_axis = (raw_axis < 0) ? (raw_axis + x_dim_num) : raw_axis;
    if (rectified_axis < 0 || rectified_axis >= x_dim_num) {
      OP_LOGE("SqueezeV3", "Squeeze failed, as axes val[%ld] is out of range[-%ld, %ld).", raw_axis,
                          x_dim_num, x_dim_num);
      return ge::GRAPH_FAILED;
    }
    squeeze_dims[rectified_axis] = true;
  }
  return ge::GRAPH_SUCCESS;
}

ge::graphStatus SetOutputShapeAsUnknownRank(gert::Shape *y_shape) {
  y_shape->SetDimNum(1U);
  y_shape->SetDim(0U, ge::UNKNOWN_DIM_NUM);
  return ge::GRAPH_SUCCESS;
}
}  // namespace

static ge::graphStatus InferShapeForSqueezeV3(gert::InferShapeContext* context) {
  const auto x_shape = context->GetInputShape(0U);
  auto y_shape = context->GetOutputShape(0U);
  OP_CHECK_IF(x_shape == nullptr, OP_LOGE(context, "x_shape is null."), return ge::GRAPH_FAILED);
  OP_CHECK_IF(y_shape == nullptr, OP_LOGE(context, "y_shape is null."), return ge::GRAPH_FAILED);

  // case 1, unknown rank
  if (IsUnknownRank(*x_shape)) {
    OP_LOGD(context, "Input shape is unknown rank. Set output shape as unknown rank.");
    return SetOutputShapeAsUnknownRank(y_shape);
  }

  auto axes = context->GetOptionalInputTensor(1U);
  // case 2, unknown dims
  if (IsUnknownDim(*x_shape)) {
    if (axes == nullptr || axes->GetShapeSize() == 0 || axes->GetTensorData().GetAddr() == nullptr) {
      OP_LOGD(
          context,
          "Input shape has unknown dims, optional input axes is not set or empty. Set output shape as unknown rank.");
      return SetOutputShapeAsUnknownRank(y_shape);
    } else {
      OP_LOGD(context,
              "Input shape has unknown dims, optional input axes is not empty. Squeeze dim 1 and -1 by axes.");
      const auto axes_size = static_cast<size_t>(axes->GetShapeSize());
      const auto axes_data = static_cast<const int64_t *>(axes->GetTensorData().GetAddr());
      bool squeeze_dims[gert::Shape::kMaxDimNum] = {false};
      if (GetSqueezeDimsWhenAxesRangeValid(x_shape, axes_data, axes_size, squeeze_dims) != ge::GRAPH_SUCCESS) {
        OP_LOGE(context, "Squeeze failed, as axes val is out of range.");
        return GRAPH_FAILED;
      }
      return SqueezeWithAxes(x_shape, squeeze_dims, y_shape);
    }
  }

  // case 3, known shape
  if (axes == nullptr) {
    OP_LOGD(context, "Input shape is known and optional input axes is not set. Squeeze all dim 1.");
    return SqueezeForEmptyAxes(x_shape, y_shape);
  }
  const auto axes_size = static_cast<size_t>(axes->GetShapeSize());
  const auto axes_data = static_cast<const int64_t*>(axes->GetTensorData().GetAddr());
  if (axes_data == nullptr) {
    OP_LOGD(context, "Input shape is known but axes is not got on compile stage. Set output shape as unknown rank.");
    return SetOutputShapeAsUnknownRank(y_shape);
  } else if (axes_size == 0U) {
    OP_LOGD(context, "Input shape is known and optional input axes is empty. Squeeze all dim 1.");
    return SqueezeForEmptyAxes(x_shape, y_shape);
  } else {
    OP_LOGD(context, "Input shape is known and optional input axes is not empty. Squeeze dim 1 by axes.");
    bool squeeze_dims[gert::Shape::kMaxDimNum] = {false};
    if (GetSqueezeDimsWhenAxesRangeValid(x_shape, axes_data, axes_size, squeeze_dims) != ge::GRAPH_SUCCESS) {
      OP_LOGE(context, "Squeeze failed, as axes val is out of range.");
      return GRAPH_FAILED;
    }
    return SqueezeWithAxes(x_shape, squeeze_dims, y_shape);
  }
}

IMPL_OP_INFERSHAPE(SqueezeV3).InferShape(InferShapeForSqueezeV3).InputsDataDependency({1U});
}  // namespace ops