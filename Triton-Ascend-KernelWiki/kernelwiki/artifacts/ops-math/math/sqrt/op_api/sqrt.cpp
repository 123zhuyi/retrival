/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include "sqrt.h"
#include "opdev/aicpu/aicpu_task.h"
#include "opdev/make_op_executor.h"
#include "opdev/op_def.h"
#include "opdev/op_dfx.h"
#include "opdev/op_executor.h"
#include "opdev/op_log.h"
#include "opdev/shape_utils.h"
#include "opdev/platform.h"
#include "aclnn_kernels/common/op_error_check.h"

using namespace op;

namespace l0op {

OP_TYPE_REGISTER(Sqrt);

// 根据芯片类型、dtype判断算子是否支持走AiCore
static bool IsAiCoreSupport(const aclTensor *self) {
  // 获取芯片类型,判断是1971还是1980
  if (GetCurrentPlatformInfo().GetSocVersion() >= SocVersion::ASCEND910B &&
      GetCurrentPlatformInfo().GetSocVersion() <= SocVersion::ASCEND910E) {
    return CheckType(self->GetDataType(), {op::DataType::DT_FLOAT,
                                           op::DataType::DT_FLOAT16, op::DataType::DT_BF16});
  }

  // 1980 & other
  return CheckType(self->GetDataType(), {op::DataType::DT_FLOAT, op::DataType::DT_FLOAT16});
}

const aclTensor *SqrtAiCpu(const aclTensor *self,
                           aclOpExecutor *executor, aclTensor *sqrt_out) {
  L0_DFX(SqrtAiCpu, self);
  // 使用框架宏ADD_TO_LAUNCHER_LIST_AICPU，将AiCPu Sqrt算子加入任务队列
  static internal::AicpuTaskSpace space("Sqrt");
  auto ret = ADD_TO_LAUNCHER_LIST_AICPU(Sqrt,
                                        OP_ATTR_NAMES(),
                                        OP_INPUT(self),
                                        OP_OUTPUT(sqrt_out));
  CHECK_RET(ret == ACLNN_SUCCESS, nullptr);
  return sqrt_out;
}

const aclTensor *SqrtAiCore(const aclTensor *self,
                            aclOpExecutor *executor, aclTensor *sqrt_out) {
  L0_DFX(SqrtAiCore, self);
  auto retAicore = ADD_TO_LAUNCHER_LIST_AICORE(Sqrt,
                                               OP_INPUT(self),
                                               OP_OUTPUT(sqrt_out));
  OP_CHECK_ADD_TO_LAUNCHER_LIST_AICORE(retAicore != ACLNN_SUCCESS, return nullptr,
                                       "Sqrt ADD_TO_LAUNCHER_LIST_AICORE failed.");
  return sqrt_out;
}

const aclTensor *Sqrt(const aclTensor *self,
                      aclOpExecutor *executor) {
  auto sqrt_out = executor->AllocTensor(self->GetStorageShape(), self->GetDataType());
  CHECK_RET(sqrt_out != nullptr, nullptr);
  if (IsAiCoreSupport(self)) {
    return SqrtAiCore(self, executor, sqrt_out);
  } else {
    return SqrtAiCpu(self, executor, sqrt_out);
  }
}

}
