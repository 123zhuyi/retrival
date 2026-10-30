/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License")
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

/*!
 * \file select_infershape.cpp
 * \brief
 */
#include "log/log.h"
#include "register/op_impl_registry.h"

using namespace ge;
namespace ops {
static graphStatus InferShape4Select(gert::InferShapeContext* context) {
  const gert::Shape* x1_shape = context->GetInputShape(1);
  OP_CHECK_NULL_WITH_CONTEXT(context, x1_shape);

  gert::Shape* y_shape = context->GetOutputShape(0);
  OP_CHECK_NULL_WITH_CONTEXT(context, y_shape);
  *y_shape = *x1_shape;

  return GRAPH_SUCCESS;
}
IMPL_OP_INFERSHAPE(Select).InferShape(InferShape4Select);
}
