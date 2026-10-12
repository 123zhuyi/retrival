/* Copyright (c) Huawei Technologies Co., Ltd. 2025. All rights reserved.
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

using namespace ge;
namespace ops {
static ge::graphStatus InferDataTypeForAcos(gert::InferDataTypeContext* context) {
  OP_LOGI("Begin InferDataTypeForAcos");
  const ge::DataType xDataType = context->GetInputDataType(0);
  context->SetOutputDataType(0, xDataType);
  return ge::GRAPH_SUCCESS;
}

IMPL_OP(Acos).InferDataType(InferDataTypeForAcos);
}
// namespace ops
