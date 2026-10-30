/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License")
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include "reciprocal.h"
#include "opdev/platform.h"
#include "opdev/aicpu/aicpu_task.h"
#include "opdev/make_op_executor.h"
#include "opdev/op_def.h"
#include "opdev/op_dfx.h"
#include "opdev/op_executor.h"
#include "opdev/op_log.h"
#include "opdev/shape_utils.h"
#include "aclnn_kernels/common/op_error_check.h"

using namespace op;

namespace l0op {
OP_TYPE_REGISTER(Reciprocal);

static const std::initializer_list<op::DataType> AICORE_DTYPE_SUPPORT_LIST = {
    op::DataType::DT_FLOAT, op::DataType::DT_FLOAT16, op::DataType::DT_BF16};

// 根据芯片类型、dtype判断算子是否支持走aicore
static inline bool IsAiCoreSupport(const aclTensor *x) {
  // Reciprocal只需要判断dtype
  return CheckType(x->GetDataType(), AICORE_DTYPE_SUPPORT_LIST);
}

static const aclTensor *ReciprocalAiCore(const aclTensor *x, aclTensor *y, aclOpExecutor *executor) {
  L0_DFX(ReciprocalAiCore, x, y);

  auto retAicore = ADD_TO_LAUNCHER_LIST_AICORE(Reciprocal, OP_INPUT(x), OP_OUTPUT(y));
  OP_CHECK_ADD_TO_LAUNCHER_LIST_AICORE(retAicore != ACLNN_SUCCESS, return nullptr,
                                       "Reciprocal ADD_TO_LAUNCHER_LIST_AICORE failed.");
  return y;
}

static const aclTensor *ReciprocalAiCpu(const aclTensor *x, aclTensor *y, aclOpExecutor *executor) {
  L0_DFX(ReciprocalAiCpu, x, y);

  static internal::AicpuTaskSpace space("Reciprocal", ge::DEPEND_IN_SHAPE, true);
  auto ret = ADD_TO_LAUNCHER_LIST_AICPU(Reciprocal, OP_ATTR_NAMES(), OP_INPUT(x), OP_OUTPUT(y));
  CHECK_RET(ret == ACLNN_SUCCESS, nullptr);
  return y;
}

const aclTensor *Reciprocal(const aclTensor *x, aclOpExecutor *executor) {
  auto reciprocalOut = executor->AllocTensor(x->GetViewShape(), x->GetDataType());
  if (IsAiCoreSupport(x))
    return ReciprocalAiCore(x, reciprocalOut, executor);
  else
    return ReciprocalAiCpu(x, reciprocalOut, executor);

  return reciprocalOut;
}
}  // namespace l0op