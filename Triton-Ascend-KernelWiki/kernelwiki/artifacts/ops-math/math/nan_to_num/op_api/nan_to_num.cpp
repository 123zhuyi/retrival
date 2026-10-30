/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include "nan_to_num.h"
#include "opdev/op_dfx.h"
#include "opdev/shape_utils.h"
#include "opdev/make_op_executor.h"
#include "opdev/aicpu/aicpu_task.h"
#include "aclnn_kernels/common/op_error_check.h"

using namespace op;
namespace l0op {

OP_TYPE_REGISTER(NanToNum);

static const std::initializer_list<op::DataType> AICORE_DTYPE_SUPPORT_LIST = {
    op::DataType::DT_FLOAT, op::DataType::DT_FLOAT16, op::DataType::DT_BF16
};

static bool IsAiCoreSupport(const aclTensor *self) {
  return CheckType(self->GetDataType(), AICORE_DTYPE_SUPPORT_LIST);
}

static const aclTensor *NanToNumAiCore(const aclTensor *self, float nan, float posinf, float neginf,
                                       aclTensor *out, aclOpExecutor *executor) {
  L0_DFX(NanToNumAiCore, self, nan, posinf, neginf, out);

  auto retAicore = ADD_TO_LAUNCHER_LIST_AICORE(NanToNum,
                                               OP_INPUT(self),
                                               OP_OUTPUT(out),
                                               OP_ATTR(nan, posinf, neginf));
  OP_CHECK_ADD_TO_LAUNCHER_LIST_AICORE(retAicore != ACLNN_SUCCESS, return nullptr,
                                       "NanToNum ADD_TO_LAUNCHER_LIST_AICORE failed.");
  return out;
}

const aclTensor *NanToNum(const aclTensor *self, float nan, float posinf, float neginf, aclOpExecutor *executor) {
  auto out = executor->AllocTensor(self->GetViewShape(), self->GetDataType());
  CHECK_RET(out != nullptr, nullptr);

  if (IsAiCoreSupport(self)) {
    return NanToNumAiCore(self, nan, posinf, neginf, out, executor);
  } else {
    // 目前无AiCPU实现，如果存在aicore不支持的类型，则返回空指针
    return nullptr;
  }
}
} // namespace l0op
