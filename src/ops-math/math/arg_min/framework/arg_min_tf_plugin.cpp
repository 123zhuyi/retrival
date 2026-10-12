/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include "register/register.h"
#include "log/log.h"

namespace domi {
static Status AutoMappingByOpFnArgMin(const ge::Operator& op_src, ge::Operator& op) {
  Status ret = AutoMappingByOpFn(op_src, op);
  if (ret != SUCCESS) {
    OP_LOGE("ArgMin", "tensorflow plugin parser failed. auto mapping failed.");
    return FAILED;
  }
  ge::DataType dataType;
  if (op.GetAttr("output_type", dataType) != ge::GRAPH_SUCCESS) {
    OP_LOGE("ArgMin", "GetAttr DstT failed");
    return FAILED;
  }
  op.SetAttr("dtype", dataType);
  OP_LOGI("ArgMin", "op[ArgMin] tensorflow plugin parser[AutoMapping] success.");
  return SUCCESS;
}

REGISTER_CUSTOM_OP("ArgMin")
    .FrameworkType(TENSORFLOW)
    .OriginOpType("ArgMin")
    .ParseParamsByOperatorFn(AutoMappingByOpFnArgMin)
    .ImplyType(ImplyType::TVM);
}  // namespace domi
