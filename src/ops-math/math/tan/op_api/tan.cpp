/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include "tan.h"
#include "opdev/aicpu/aicpu_task.h"
#include "opdev/op_dfx.h"
#include "opdev/op_log.h"
#include "opdev/op_executor.h"
#include "opdev/make_op_executor.h"
#include "aclnn_kernels/common/op_error_check.h"
#include "op_api/aclnn_check.h"

using namespace op;
namespace l0op {
OP_TYPE_REGISTER(Tan);
static const std::initializer_list<op::DataType> AICORE_DTYPE_SUPPORT_LIST = {
  op::DataType::DT_FLOAT, op::DataType::DT_FLOAT16, op::DataType::DT_INT32
};

static const std::initializer_list<op::DataType> ASCEND910B_AICORE_DTYPE_SUPPORT_LIST = {
  op::DataType::DT_FLOAT, op::DataType::DT_FLOAT16, op::DataType::DT_INT32, op::DataType::DT_BF16
};

static bool IsAiCoreSupport(const aclTensor *x) {
  if (GetCurrentPlatformInfo().GetSocVersion() == SocVersion::ASCEND910B ||
      GetCurrentPlatformInfo().GetSocVersion() == SocVersion::ASCEND910_93 ||
      IsRegBase()) {
    return CheckType(x->GetDataType(), ASCEND910B_AICORE_DTYPE_SUPPORT_LIST);
  }
  return CheckType(x->GetDataType(), AICORE_DTYPE_SUPPORT_LIST);
}

static const aclTensor *TanAiCore(const aclTensor *x, aclTensor *y, aclOpExecutor *executor) {
  L0_DFX(TanAiCore, x, y);

  auto retAicore = ADD_TO_LAUNCHER_LIST_AICORE(Tan,
                                               OP_INPUT(x),
                                               OP_OUTPUT(y));
  OP_CHECK_ADD_TO_LAUNCHER_LIST_AICORE(retAicore != ACLNN_SUCCESS, return nullptr,
                                       "Tan ADD_TO_LAUNCHER_LIST_AICORE failed.");
  return y;
}

static const aclTensor *TanAiCpu(const aclTensor *x, aclTensor *y, aclOpExecutor *executor) {
  L0_DFX(TanAiCpu, x, y);

  static internal::AicpuTaskSpace space("Tan");
  auto ret = ADD_TO_LAUNCHER_LIST_AICPU(Tan,
                                        OP_ATTR_NAMES(),
                                        OP_INPUT(x),
                                        OP_OUTPUT(y));
  CHECK_RET(ret == ACLNN_SUCCESS, nullptr);
  return y;
}

const aclTensor *Tan(const aclTensor *x, aclOpExecutor *executor) {
  auto y = executor->AllocTensor(x->GetViewShape(), x->GetDataType());

  if (IsAiCoreSupport(x)) {
    return TanAiCore(x, y, executor);
  } else {
    return TanAiCpu(x, y, executor);
  }
}
}