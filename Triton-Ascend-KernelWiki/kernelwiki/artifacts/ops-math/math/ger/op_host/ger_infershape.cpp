/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

/*!
 * \file mul_infershape.cpp
 * \brief
 */

#include "log/log.h"
#include "infershape_broadcast_util.h"
#include "register/op_impl_registry.h"

using namespace ge;
namespace ops {
static constexpr size_t EXPECT_DIM = 2;

static ge::graphStatus InferShape4Broadcast(gert::InferShapeContext* context) {
  auto in_shape1 = context->GetInputShape(0);
  OP_CHECK_NULL_WITH_CONTEXT(context, in_shape1);
  auto in_shape2 = context->GetInputShape(1);
  OP_CHECK_NULL_WITH_CONTEXT(context, in_shape2);
  auto out_shape = context->GetOutputShape(0);
  OP_CHECK_NULL_WITH_CONTEXT(context, out_shape);

  // 校验输入shape必须只有一根轴
  if (in_shape1->GetDimNum() != 1 || in_shape2->GetDimNum() != 1) {
        OP_LOGE(
            context->GetNodeName(),
            "Expected 1-D argument for both input, but got %zu-D for input0 and  %zu-D for input1",
            in_shape1->GetDimNum(), in_shape2->GetDimNum());
        return ge::GRAPH_FAILED;
  }

  // 输出shape 0维取第一个输入的轴， 1维取第二个输入的轴
  out_shape->SetDimNum(EXPECT_DIM);
  out_shape->SetDim(0, in_shape1->GetDim(0));
  out_shape->SetDim(1, in_shape2->GetDim(0));

  return ge::GRAPH_SUCCESS;
}

IMPL_OP_INFERSHAPE(Ger).InferShape(InferShape4Broadcast);

}