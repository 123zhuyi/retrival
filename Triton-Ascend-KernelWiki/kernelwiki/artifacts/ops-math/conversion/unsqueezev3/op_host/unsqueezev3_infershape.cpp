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
 * \file unsqueezev3_infershape.cc
 * \brief
 */

#include "runtime/infer_shape_context.h"
#include "register/op_impl_registry.h"
#include "log/log.h"
#include <algorithm>

using namespace ge;
namespace ops {
namespace {
ge::graphStatus UnsqueezeWithValidAxes(const gert::Shape *x_shape, gert::Shape *y_shape, const int64_t *axes_data,
                                       size_t axes_size) {
  size_t axes_pos = 0U;
  size_t x_pos = 0U;
  for (size_t idx = 0U; idx < y_shape->GetDimNum(); ++idx) {
    if (axes_pos < axes_size && static_cast<int64_t>(idx) == axes_data[axes_pos]) {
      y_shape->SetDim(idx, 1);
      ++axes_pos;
    } else {
      y_shape->SetDim(idx, x_shape->GetDim(x_pos));
      ++x_pos;
    }
  }
  return ge::GRAPH_SUCCESS;
}

ge::graphStatus SetOutputAsUnknownRank(gert::Shape *y_shape) {
  y_shape->SetDimNum(1U);
  y_shape->SetDim(0U, ge::UNKNOWN_DIM_NUM);
  return ge::GRAPH_SUCCESS;
}

bool IsAxesRangeValidAndNotRepeated(const gert::Shape *x_shape, int64_t *axes_data, const size_t axes_size) {
  const int64_t y_dim_num = static_cast<int64_t>(x_shape->GetDimNum() + axes_size);
  for (size_t idx = 0U; idx < axes_size; ++idx) {
    int64_t &axis = axes_data[idx];
    axis = axis < 0 ? axis + y_dim_num : axis;
    if (axis < 0 || axis >= y_dim_num) {
      OP_LOGE("UnsqueezeV3", "Unsqueeze failed. %s", 
              ("Axes val is out of range, expect to be in range of [-" + std::to_string(y_dim_num) + "," +
              std::to_string(y_dim_num - 1) + "], but got " + std::to_string(axis) + ".").c_str());
      return false;
    }
  }

  std::sort(axes_data, axes_data + axes_size);
  size_t unique_axis_data_size = std::unique(axes_data, axes_data + axes_size) - axes_data;
  if (unique_axis_data_size != axes_size) {
    OP_LOGE("UnsqueezeV3", "Unsqueezed failed, as unsqueeze axes should not contain duplicated values.");
    return false;
  }
  return true;
}

ge::graphStatus InferShapeMainImplForUnsqueezeV3(const gert::Shape *x_shape, gert::Shape *y_shape,
                                                 const gert::Tensor *axes) {
  // solve axes input buff is null,  check whether is dynamic phase
  auto axes_data = static_cast<int64_t*>(axes->GetTensorData().GetAddr());
  if (axes_data == nullptr) {
    // check whether input is unknown_rank
    auto x_dim_num = x_shape->GetDimNum();
    if (x_dim_num == 1U && x_shape->GetDim(0U) == ge::UNKNOWN_DIM_NUM) {
      return SetOutputAsUnknownRank(y_shape);
    }
    OP_LOGD("UnsqueezeV3", "Set output shape as dynamic shape, as can not get axes tensor data.");
    return SetOutputAsUnknownRank(y_shape);
  }
  // tensor input buff not null, static phase
  // solve empty axes
  const size_t axes_size = static_cast<size_t>(axes->GetShapeSize());
  if (axes_size == 0) {
    *y_shape = *x_shape;
    return ge::GRAPH_SUCCESS;
  }
  // solve normal axes
  int64_t copied_axes_data[gert::Shape::kMaxDimNum] = {0};
  if (memcpy_s(copied_axes_data, sizeof(int64_t) * gert::Shape::kMaxDimNum,
               axes_data, sizeof(int64_t) * axes_size) != 0) {
    OP_LOGE("UnsqueezeV3", "memcpy_s not success!");
    return ge::GRAPH_FAILED;
  }
  if (!IsAxesRangeValidAndNotRepeated(x_shape, copied_axes_data, axes_size)) {
    return ge::GRAPH_FAILED;
  }
  y_shape->SetDimNum(x_shape->GetDimNum() + axes_size);
  return UnsqueezeWithValidAxes(x_shape, y_shape, copied_axes_data, axes_size);
}
}  // namespace

static ge::graphStatus InferShapeForUnsqueezeV3(gert::InferShapeContext* context) {
  auto x_shape = context->GetInputShape(0U);
  auto y_shape = context->GetOutputShape(0U);
  auto axes = context->GetInputTensor(1U);
  OP_CHECK_IF(x_shape == nullptr, OP_LOGE(context, "x_shape is null."), return ge::GRAPH_FAILED);
  OP_CHECK_IF(y_shape == nullptr, OP_LOGE(context, "y_shape is null."), return ge::GRAPH_FAILED);
  OP_CHECK_IF(axes == nullptr, OP_LOGE(context, "axes is null."), return ge::GRAPH_FAILED);

  return InferShapeMainImplForUnsqueezeV3(x_shape, y_shape, axes);
}

IMPL_OP_INFERSHAPE(UnsqueezeV3).InferShape(InferShapeForUnsqueezeV3).InputsDataDependency({1U});
}  // namespace ops