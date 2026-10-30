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
 * \file fill.cc
 * \brief
 */
#include "register/op_impl_registry.h"
#include "log/log.h"
#include "op_host/util/const_util.h"

using namespace ge;
namespace ops {
constexpr size_t INPUT_INDEX_DIMS = 0;

static ge::graphStatus InferShape4Fill(gert::InferShapeContext* context) {
  auto out_shape = context->GetOutputShape(0);
  OP_CHECK_NULL_WITH_CONTEXT(context, out_shape);
  OP_CHECK_IF(
      (!Ops::Base::GetConstIntToShape<gert::InferShapeContext>(context, INPUT_INDEX_DIMS, *out_shape)),
      OP_LOGE(
          context->GetNodeName(), "GetConstIntToShape failed!"),
      return ge::GRAPH_FAILED);

  return GRAPH_SUCCESS;
}

IMPL_OP_INFERSHAPE(Fill).InferShape(InferShape4Fill).InputsDataDependency({INPUT_INDEX_DIMS});
}  // namespace ops
