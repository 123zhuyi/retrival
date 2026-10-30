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
 * \file eye.cpp
 * \brief
 */

#include "eye.h"
#include "opdev/aicpu/aicpu_task.h"
#include "opdev/data_type_utils.h"
#include "opdev/make_op_executor.h"
#include "opdev/op_def.h"
#include "opdev/op_dfx.h"
#include "opdev/op_log.h"
#include "opdev/shape_utils.h"
#include "opdev/platform.h"
#include "op_api/aclnn_check.h"

using namespace op;

namespace l0op {
OP_TYPE_REGISTER(Eye);

// AICORE算子kernel
inline static const aclTensor *EyeAiCore(aclTensor *out, const int64_t n, const int64_t m, aclOpExecutor *executor) {
  L0_DFX(EyeAiCore, out, n, m);
  // 使用框架宏ADD_TO_LAUNCHER_LIST_AICORE，将AiCore Eye算子加入任务队列
  auto ret = ADD_TO_LAUNCHER_LIST_AICORE(Eye, OP_OUTPUT(out), OP_ATTR(n, m, static_cast<aclIntArray *>(nullptr), out->GetDataType()));
  OP_CHECK(ret ==  ACLNN_SUCCESS, OP_LOGE(ACLNN_ERR_INNER_NULLPTR, "EyeAiCcore ADD_TO_LAUNCHER_LIST_AICORE failed."),
    return nullptr);
  return out;
}

const aclTensor *Eye(aclTensor *out, const int64_t n, const int64_t m, aclOpExecutor *executor) {
  auto npuArch = op::GetCurrentPlatformInfo().GetCurNpuArch();
  if (IsRegBase(npuArch)) {
    auto eyeOut = executor->AllocTensor(out->GetViewShape(), out->GetDataType());
    CHECK_RET(eyeOut != nullptr, nullptr);
    return EyeAiCore(eyeOut, n, m, executor);
  }
  return EyeAiCore(out, n, m, executor);
}
}  // namespace l0op