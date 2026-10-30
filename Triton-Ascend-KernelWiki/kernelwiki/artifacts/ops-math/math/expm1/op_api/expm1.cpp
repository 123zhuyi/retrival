/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include "expm1.h"
#include "op_api/aclnn_check.h"
#include "opdev/op_dfx.h"
#include "opdev/shape_utils.h"
#include "opdev/make_op_executor.h"

using namespace op;
namespace l0op {
OP_TYPE_REGISTER(Expm1);

static const std::initializer_list<op::DataType> AICORE_DTYPE_SUPPORT_LIST = {
    op::DataType::DT_FLOAT, op::DataType::DT_FLOAT16};

static const std::initializer_list<op::DataType> ASCEND910B_AICORE_DTYPE_SUPPORT_LIST = {
    op::DataType::DT_FLOAT, op::DataType::DT_FLOAT16, op::DataType::DT_BF16};

static bool IsAiCoreSupport(const aclTensor *self) {
  if (GetCurrentPlatformInfo().GetSocVersion() == SocVersion::ASCEND910B ||
      GetCurrentPlatformInfo().GetSocVersion() == SocVersion::ASCEND910_93 ||
      IsRegBase()) {
    return CheckType(self->GetDataType(), ASCEND910B_AICORE_DTYPE_SUPPORT_LIST);
  }
  return CheckType(self->GetDataType(), AICORE_DTYPE_SUPPORT_LIST);
}

// AiCore逻辑
static void Expm1AiCore(const aclTensor *self, const aclTensor *out, aclOpExecutor *executor) {
  L0_DFX(Expm1AiCore, self, out);
  auto retAicore = ADD_TO_LAUNCHER_LIST_AICORE(Expm1, OP_INPUT(self), OP_OUTPUT(out));
  if (retAicore != ACLNN_SUCCESS) {
    OP_LOGE(ACLNN_ERR_INNER_NULLPTR, "Expm1 ADD_TO_LAUNCHER_LIST_AICORE failed.");
  }
}

const aclTensor *Expm1(const aclTensor *self, aclOpExecutor *executor) {
  L0_DFX(Expm1, self);
  // 判断走AiCore还是AiCPU，目前无AiCPU实现，默认走AiCore
  if (!IsAiCoreSupport(self)) {
    return nullptr;
  }
  // 根据输入shape申请输出tensor
  auto out = executor->AllocTensor(self->GetViewShape(), self->GetDataType());
  if (out == nullptr) {
    return out;
  }
  Expm1AiCore(self, out, executor);
  return out;
}
} // namespace l0op
