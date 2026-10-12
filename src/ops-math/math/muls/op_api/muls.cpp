/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include "muls.h"
#include "opdev/op_log.h"
#include "opdev/op_executor.h"
#include "opdev/make_op_executor.h"
#include "opdev/shape_utils.h"
#include "opdev/shape_utils.h"
#include "opdev/op_def.h"
#include "opdev/op_dfx.h"
#include "aclnn_kernels/common/op_error_check.h"

using namespace op;

namespace l0op {

OP_TYPE_REGISTER(Muls);

const aclTensor *Muls(const aclTensor *self,
                      float alpha,
                      aclOpExecutor *executor) {
  L0_DFX(Muls, self, alpha);

  auto mulsOut = executor->AllocTensor(self->GetViewShape(), self->GetDataType());

  auto retAicore = ADD_TO_LAUNCHER_LIST_AICORE(Muls,
                                               OP_INPUT(self),
                                               OP_OUTPUT(mulsOut),
                                               OP_ATTR(alpha));
  OP_CHECK_ADD_TO_LAUNCHER_LIST_AICORE(retAicore != ACLNN_SUCCESS, return nullptr,
                                       "Muls ADD_TO_LAUNCHER_LIST_AICORE failed.");
  return mulsOut;
}

const aclTensor *MulsInplace(const aclTensor *self,
                      float alpha,
                      aclOpExecutor *executor) {
  L0_DFX(MulsInplace, self, alpha);

  auto mulsOut = self;

  auto retAicore = ADD_TO_LAUNCHER_LIST_AICORE(Muls,
                                               OP_INPUT(self),
                                               OP_OUTPUT(mulsOut),
                                               OP_ATTR(alpha));
  OP_CHECK_ADD_TO_LAUNCHER_LIST_AICORE(retAicore != ACLNN_SUCCESS, return nullptr,
                                       "Muls ADD_TO_LAUNCHER_LIST_AICORE failed.");
  return mulsOut;
}
}
