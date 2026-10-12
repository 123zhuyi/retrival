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
 * \file unsqueeze_infershape.cc
 * \brief
 */
#include "runtime/infer_shape_context.h"
#include "register/op_impl_registry.h"
#include "log/log.h"

using namespace ge;
namespace ops {
static ge::graphStatus InferShapeForUnsqueeze(gert::InferShapeContext *context) {
  const auto x_shape = context->GetInputShape(0);
  auto y_shape = context->GetOutputShape(0);
  const gert::RuntimeAttrs* attrs = context->GetAttrs();
  if (attrs == nullptr) {
    OP_LOGE(context->GetNodeName(), "attrs pointer must not be nullptr!");
    return ge::GRAPH_FAILED;
  }
  auto axes = attrs->GetAttrPointer<gert::TypedContinuousVector<int64_t>>(0);

  if (x_shape == nullptr || axes == nullptr || y_shape == nullptr) {
    OP_LOGE(context->GetNodeName(), "X_shape or axes or y_shape pointer must not be nullptr!");
    return ge::GRAPH_FAILED;
  }

  const auto dim_size = x_shape->GetDimNum();
  const auto out_dim_size = dim_size + axes->GetSize();
  y_shape->SetDimNum(out_dim_size);
  if (out_dim_size > y_shape->kMaxDimNum) {
    OP_LOGE(context->GetNodeName(),
                           "DimNum of output shape is %zu, larger than kMaxDimNum which is %zu!",
                           out_dim_size, (y_shape->kMaxDimNum));
    return ge::GRAPH_FAILED;
  }
  for (size_t i = 0UL; i < out_dim_size; ++i) {
    y_shape->SetDim(i, 0);
  }

  for (size_t i = 0UL; i < axes->GetSize(); ++i) {
    const int64_t axis = (axes->GetData())[i];
    const int64_t real_axis = (axis < 0) ? (axis + static_cast<int64_t>(out_dim_size)) : axis;
    if ((real_axis < 0) || (real_axis >= static_cast<int64_t>(out_dim_size))) {
      OP_LOGE(context->GetNodeName(), "Unsqueeze axis must be in range [-%zu, %zu)!",
                             out_dim_size, out_dim_size);
      return ge::GRAPH_FAILED;
    }
    if (y_shape->GetDim(real_axis) == 1) {
      OP_LOGE(context->GetNodeName(), "Unsqueeze axis repeated!");
      return ge::GRAPH_FAILED;
    }
    y_shape->SetDim(real_axis, 1);
  }

  size_t idx = 0UL;
  for (size_t i = 0UL; i < out_dim_size; ++i) {
    if (y_shape->GetDim(i) != 1) {
      y_shape->SetDim(i, x_shape->GetDim(idx));
      ++idx;
    }
  }
  return ge::GRAPH_SUCCESS;
}

IMPL_OP_INFERSHAPE(Unsqueeze).InferShape(InferShapeForUnsqueeze);
}  // namespace ops