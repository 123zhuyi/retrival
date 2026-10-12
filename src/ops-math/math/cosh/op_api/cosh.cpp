/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include "cosh.h"
#include "opdev/make_op_executor.h"
#include "opdev/aicpu/aicpu_task.h"
#include "opdev/op_def.h"
#include "opdev/op_dfx.h"
#include "opdev/op_executor.h"
#include "opdev/op_log.h"
#include "opdev/shape_utils.h"
#include "op_api/aclnn_check.h"

using namespace op;

namespace l0op {
OP_TYPE_REGISTER(Cosh);

static const std::initializer_list<op::DataType> AICORE_DTYPE_SUPPORT_LIST_SELF = {
    op::DataType::DT_FLOAT, op::DataType::DT_FLOAT16, op::DataType::DT_BF16};

static const std::initializer_list<op::DataType> AICORE_910B_DTYPE_SUPPORT_LIST = {
    op::DataType::DT_FLOAT, op::DataType::DT_FLOAT16, op::DataType::DT_BF16};

static bool IsAiCoreSupport(const aclTensor *self) {
  if (GetCurrentPlatformInfo().GetSocVersion() == SocVersion::ASCEND910B ||
      GetCurrentPlatformInfo().GetSocVersion() == SocVersion::ASCEND910_93 ||
      IsRegBase()) {
    return CheckType(self->GetDataType(), AICORE_910B_DTYPE_SUPPORT_LIST);
  }
  return CheckType(self->GetDataType(), AICORE_DTYPE_SUPPORT_LIST_SELF);
}

const aclTensor *CoshAiCore(const aclTensor *self,
                            aclOpExecutor *executor, aclTensor *coshOut) {
  L0_DFX(CoshAiCore, self);
  auto ret = ADD_TO_LAUNCHER_LIST_AICORE(Cosh, OP_INPUT(self),OP_OUTPUT(coshOut));
  OP_CHECK(ret == ACLNN_SUCCESS, OP_LOGE(ACLNN_ERR_INNER_NULLPTR, "CoshAiCore ADD_TO_LAUNCHER_LIST_AICORE failed."), return nullptr);
  return coshOut;
}

const aclTensor *CoshAiCpu(const aclTensor *self,
                           aclOpExecutor *executor, aclTensor *coshOut) {
  L0_DFX(CoshAiCpu, self);
  static internal::AicpuTaskSpace space("Cosh");
  auto ret = ADD_TO_LAUNCHER_LIST_AICPU(Cosh,
                                        OP_ATTR_NAMES(),
                                        OP_INPUT(self),
                                        OP_OUTPUT(coshOut));
  OP_CHECK(ret ==  ACL_SUCCESS, OP_LOGE(ACLNN_ERR_INNER_NULLPTR, "CoshAiCpu ADD_TO_LAUNCHER_LIST_AICPU failed."), return nullptr);
  return coshOut;
}

const aclTensor *Cosh(const aclTensor *self,
                      aclOpExecutor *executor) {
  L0_DFX(Cosh, self);
  auto coshOut = executor->AllocTensor(self->GetStorageShape(), self->GetDataType());
  CHECK_RET(coshOut != nullptr, nullptr);
  if (IsAiCoreSupport(self)) {
    return CoshAiCore(self, executor, coshOut);
  } else {
    return CoshAiCpu(self, executor, coshOut);
  }
}
}
