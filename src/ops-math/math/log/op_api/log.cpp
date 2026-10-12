/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include "log.h"
#include "opdev/aicpu/aicpu_task.h"
#include "opdev/make_op_executor.h"
#include "opdev/op_def.h"
#include "opdev/op_dfx.h"
#include "opdev/op_executor.h"
#include "opdev/op_log.h"
#include "opdev/shape_utils.h"

using namespace op;

constexpr float kLog10Base = 10.0F;

namespace l0op {

OP_TYPE_REGISTER(Log);

static const std::initializer_list<op::DataType> AICORE_DTYPE_SUPPORT_LIST = {
    op::DataType::DT_FLOAT, op::DataType::DT_FLOAT16, op::DataType::DT_BF16};

static const std::initializer_list<op::DataType> UNCHANGED_DTYPE_LIST = {
    op::DataType::DT_BF16,   op::DataType::DT_FLOAT16,   op::DataType::DT_FLOAT,
    op::DataType::DT_DOUBLE, op::DataType::DT_COMPLEX64, op::DataType::DT_COMPLEX128};

// 根据芯片类型、dtype判断算子是否支持走aicore
static bool IsAiCoreSupport(const aclTensor *self) {
  return CheckType(self->GetDataType(), AICORE_DTYPE_SUPPORT_LIST);
}

// AICORE算子kernel, base属性：对数的底数
static const aclTensor *LogAiCore(const aclTensor *self, const float base, const float scale,
                                  const float shift, aclTensor *out, aclOpExecutor *executor) {
  L0_DFX(LogAiCore, self, base, scale, shift, out);
  auto ret = ADD_TO_LAUNCHER_LIST_AICORE(Log, OP_INPUT(self), OP_OUTPUT(out), OP_ATTR(base, scale, shift));
  OP_CHECK(ret == ACLNN_SUCCESS, OP_LOGE(ACLNN_ERR_INNER_NULLPTR, "LogAiCore ADD_TO_LAUNCHER_LIST_AICORE failed."), return nullptr);
  return out;
}

// AICPU算子自研kernel, 支持属性
static const aclTensor* LogAiCpu(const aclTensor* self, const float base, const float scale, const float shift,
                                 aclTensor* out, aclOpExecutor* executor) {
  L0_DFX(LogAiCpu, self, base, scale, shift, out);
  static internal::AicpuTaskSpace space("Log", ge::DEPEND_IN_SHAPE, false);
  auto ret = ADD_TO_LAUNCHER_LIST_AICPU(Log, OP_ATTR_NAMES({"base", "scale", "shift"}), OP_INPUT(self), OP_OUTPUT(out),
                                        OP_ATTR(base, scale, shift));
  OP_CHECK(ret == ACLNN_SUCCESS, OP_LOGE(ACLNN_ERR_INNER_NULLPTR, "LogAiCpu ADD_TO_LAUNCHER_LIST_AICPU failed."), return nullptr);
  return out;
}

const aclTensor *Log(const aclTensor *self, const float base, const float scale,
                     const float shift, aclOpExecutor *executor) {
  L0_DFX(Log, self);
  aclTensor* out = nullptr;
  if (CheckType(self->GetDataType(), UNCHANGED_DTYPE_LIST)) {
    out = executor->AllocTensor(self->GetViewShape(), self->GetDataType());
  } else {
    out = executor->AllocTensor(self->GetViewShape(), op::DataType::DT_FLOAT);
  }
  CHECK_RET(out != nullptr, nullptr);
  if (IsAiCoreSupport(self)) {
    return LogAiCore(self, base, scale, shift, out, executor);
  } else {
    return LogAiCpu(self, base, scale, shift, out, executor);
  }
}

}  // namespace l0op
