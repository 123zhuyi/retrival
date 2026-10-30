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
 * \file adjacent_difference_graph_plugin.cpp
 * \brief
 */


#include "register/op_impl_registry.h"
#include "log/log.h"

using namespace ge;
namespace ops {
static constexpr int Y_IDX = 0;

graphStatus InferDataType4AdjacentDifference(gert::InferDataTypeContext* context) {
  OP_LOGI(context->GetNodeName(), "Begin to do InferDataType4AdjacentDifference");
  auto attrs = context->GetAttrs();
  OP_CHECK_NULL_WITH_CONTEXT(context, attrs);
  auto yDtypePtr = attrs->GetAttrPointer<int64_t>(0);
  OP_CHECK_NULL_WITH_CONTEXT(context, yDtypePtr);

  int64_t yDtype = *yDtypePtr;
  if (yDtype == static_cast<int64_t>(ge::DataType::DT_INT64)) {
    OP_LOGI(context->GetNodeName(), "yDtype is DT_INT64");
    context->SetOutputDataType(Y_IDX, DT_INT64);
  } else if (yDtype == static_cast<int64_t>(ge::DataType::DT_INT32)) {
    OP_LOGI(context->GetNodeName(), "yDtype is DT_INT32");
    context->SetOutputDataType(Y_IDX, DT_INT32);
  } else {
    OP_LOGE(context->GetNodeName(), "The yDtype only support int32 or int64.");
    return GRAPH_FAILED;
  }
  OP_LOGI(context->GetNodeName(), "End to do InferDataType4AdjacentDifference");

  return GRAPH_SUCCESS;
}

IMPL_OP(AdjacentDifference).InferDataType(InferDataType4AdjacentDifference);
}  // namespace ops