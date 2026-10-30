/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include "register/op_impl_registry.h"
#include "log/log.h"
#include "op_host/util/shape_util.h"
#include "op_api/op_util.h"
#include "util/const_util.h"

using namespace ge;
using namespace Ops::Base;

namespace ops {
static graphStatus InferShapeForUnpack(gert::InferShapeContext* context) {
  OP_LOGI(context->GetNodeName(), "UnpackInferShape function start!");
  const gert::RuntimeAttrs *attrs = context->GetAttrs();
  OP_CHECK_NULL_WITH_CONTEXT(context, attrs);
  const int64_t *num = attrs->GetAttrPointer<int64_t>(0);
  OP_CHECK_NULL_WITH_CONTEXT(context, num);
  const int64_t *axis = attrs->GetAttrPointer<int64_t>(1);
  OP_CHECK_NULL_WITH_CONTEXT(context, axis);
  if (!context->GetOutputShape(*num - 1) || context->GetOutputShape(*num)) {
    OP_LOGE(context->GetNodeName(), "Get num value err.");
    return GRAPH_FAILED;
  }

  const gert::Shape *input_x_shape = context->GetInputShape(0);
  size_t input_x_dim_num = input_x_shape->GetDimNum();
  int64_t real_axis = (*axis >= 0) ? *axis : *axis + input_x_dim_num;
  if (real_axis < 0 || real_axis >= static_cast<int64_t>(input_x_dim_num)) {
    OP_LOGE(context->GetNodeName(), "%s", ConcatString("Axis exceeding the prescribed range. Axis is ", real_axis,
                                                   " and x_shape's size is ", input_x_dim_num).c_str());
    return GRAPH_FAILED;
  }

  for (size_t i0 = 0; static_cast<int64_t>(i0) < *num; i0++) {
    gert::Shape *output_y_i0_shape = context->GetOutputShape(i0);
    OP_CHECK_NULL_WITH_CONTEXT(context, output_y_i0_shape);
    output_y_i0_shape->SetDimNum(input_x_dim_num - 1);
    for (size_t i1 = 0; i1 < input_x_dim_num; i1++) {
      if (static_cast<int64_t>(i1) < real_axis) {
        output_y_i0_shape->SetDim(i1, input_x_shape->GetDim(i1));
      } else if (static_cast<int64_t>(i1) > real_axis) {
        output_y_i0_shape->SetDim(i1 - 1, input_x_shape->GetDim(i1));
      }
    }
  }
  OP_LOGI(context->GetNodeName(), "UnpackInferShape function End!");
  return GRAPH_SUCCESS;
}

IMPL_OP_INFERSHAPE(Unpack).InferShape(InferShapeForUnpack);
}  // namespace ops