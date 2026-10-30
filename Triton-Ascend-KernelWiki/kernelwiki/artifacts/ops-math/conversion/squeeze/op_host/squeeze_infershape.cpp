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
 * \file squeeze_infershape.cpp
 * \brief
 */
#include "runtime/infer_shape_context.h"
#include "register/op_impl_registry.h"
#include "log/log.h"

using namespace ge;
namespace ops {
static ge::graphStatus InferShapeForSqueeze(gert::InferShapeContext* context) {
  const auto x_shape = context->GetInputShape(0);
  auto y_shape = context->GetOutputShape(0);
  const gert::RuntimeAttrs *attrs = context->GetAttrs();
  OP_CHECK_IF(x_shape == nullptr, OP_LOGE(context, "x_shape is null."), return ge::GRAPH_FAILED);
  OP_CHECK_IF(y_shape == nullptr, OP_LOGE(context, "y_shape is null."), return ge::GRAPH_FAILED);
  OP_CHECK_IF(attrs == nullptr, OP_LOGE(context, "attrs is null."), return ge::GRAPH_FAILED);
  auto axis = attrs->GetAttrPointer<gert::TypedContinuousVector<int64_t>>(0);
  OP_CHECK_IF(axis == nullptr, OP_LOGE(context, "axis is null."), return ge::GRAPH_FAILED);

  // todo 增加编译期处理逻辑
  y_shape->SetDimNum(0);
  auto dim_size = x_shape->GetDimNum();
  if (axis->GetSize() == 0UL) {
    // squeeze all 1
    for (size_t index = 0UL; index < dim_size; ++index) {
      if (x_shape->GetDim(index) != 1) {
        y_shape->AppendDim(x_shape->GetDim(index));
      }
    }
    return ge::GRAPH_SUCCESS;
  }
  // squeeze by axes
  bool dim_index[gert::Shape::kMaxDimNum] = {false};
  for (size_t i = 0UL; i < axis->GetSize(); ++i) {
    const int64_t tmp_axis = (axis->GetData())[i];
    const int64_t real_axis = (tmp_axis < 0) ? (tmp_axis + static_cast<int64_t>(dim_size)) : tmp_axis;
    if ((real_axis < 0) || (real_axis >= static_cast<int64_t>(dim_size))) {
      OP_LOGE(context, "Squeeze axis[%ld] must be in range [-%ld, %ld)!", tmp_axis,
              static_cast<int64_t>(dim_size), static_cast<int64_t>(dim_size));
      return ge::GRAPH_FAILED;
    }
    if (x_shape->GetDim(real_axis) != 1) {
      OP_LOGE(context, "Cannot squeeze the shape whose dim was not 1!");
      return ge::GRAPH_FAILED;
    }

    dim_index[real_axis] = true;
  }

  for (size_t i = 0UL; i < dim_size; ++i) {
    if (dim_index[i] != true) {
      y_shape->AppendDim(x_shape->GetDim(i));
    }
  }
  return ge::GRAPH_SUCCESS;
}

IMPL_OP_INFERSHAPE(Squeeze).InferShape(InferShapeForSqueeze);
}  // namespace ops