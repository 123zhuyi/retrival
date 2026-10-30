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
 * \file reduce_infer.cc
 * \brief
 */
#include "log/log.h"
#include "register/op_impl_registry.h"
#include "op_host/util/shape_util.h"
#include "op_api/op_util.h"
#include "util/math_util.h"
#include "op_host/infershape_reduce_util.h"

using namespace ge;
using namespace Ops::Base;
namespace ops {
static ge::graphStatus InferShape4ReduceCommon(gert::InferShapeContext* context) {
  auto in_shape = context->GetInputShape(0);
  OP_CHECK_NULL_WITH_CONTEXT(context, in_shape);
  auto axes_tensor = context->GetInputTensor(1);
  OP_CHECK_NULL_WITH_CONTEXT(context, axes_tensor);
  auto out_shape = context->GetOutputShape(0);
  OP_CHECK_NULL_WITH_CONTEXT(context, out_shape);
  auto attrs = context->GetAttrs();
  OP_CHECK_NULL_WITH_CONTEXT(context, attrs);

  const bool* keep_dims = attrs->GetAttrPointer<bool>(0);
  OP_CHECK_NULL_WITH_CONTEXT(context, keep_dims);

  auto axes_size = static_cast<int32_t>(axes_tensor->GetShapeSize());

  OP_CHECK_IF(axes_size < 0,
           OP_LOGE(context->GetNodeName(), "axes num cannot be less than 0!"),
           return ge::GRAPH_FAILED);

  if (axes_size == 0) {
    *out_shape = *in_shape;
    OP_LOGD(context->GetNodeName(), "axes is empty tensor, will ignore infer, set output shape = input shape");
    return ge::GRAPH_SUCCESS;
  }

  auto dtype = axes_tensor->GetDataType();
  OP_CHECK_IF(dtype != ge::DT_INT32 && dtype != ge::DT_INT64,
           OP_LOGE(
               context->GetNodeName(), "axes datatype %s must in (int32, int64)", ToString(dtype).c_str()),
           return ge::GRAPH_FAILED);
  if (dtype == ge::DT_INT32) {
    return ReduceDims<int32_t>(in_shape, axes_tensor, axes_size, *keep_dims, out_shape);
  }
  return ReduceDims<int64_t>(in_shape, axes_tensor, axes_size, *keep_dims, out_shape);
}

IMPL_OP_INFERSHAPE(ReduceLogSumExp).InferShape(InferShape4ReduceCommon).InputsDataDependency({1});
}  // namespace ops