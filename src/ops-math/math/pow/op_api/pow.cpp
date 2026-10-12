/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include "pow.h"
#include "opdev/aicpu/aicpu_task.h"
#include "opdev/make_op_executor.h"
#include "opdev/op_def.h"
#include "opdev/op_dfx.h"
#include "opdev/shape_utils.h"
#include "opdev/platform.h"
#include "opdev/op_log.h"
#include "op_api/aclnn_check.h"

using namespace op;

namespace l0op {

OP_TYPE_REGISTER(Pow);
 
static const std::initializer_list<op::DataType> AICORE_DTYPE_SUPPORT_LIST = {
    op::DataType::DT_FLOAT,  op::DataType::DT_FLOAT16,  op::DataType::DT_INT32,
    op::DataType::DT_INT8,   op::DataType::DT_UINT8};

static const std::initializer_list<op::DataType> AICORE_DTYPE_950_SUPPORT_LIST = {
    op::DataType::DT_FLOAT,  op::DataType::DT_FLOAT16,  op::DataType::DT_BF16, 
    op::DataType::DT_INT32,  op::DataType::DT_INT16,  op::DataType::DT_INT8, 
    op::DataType::DT_UINT8};

// 根据芯片类型、dtype判断算子是否支持走AiCore
static bool IsAiCoreSupport(const aclTensor *self) {
  auto socVersion = GetCurrentPlatformInfo().GetSocVersion();
  if (IsRegBase()) {
    return (CheckType(self->GetDataType(), AICORE_DTYPE_950_SUPPORT_LIST));
  }
  // 910b至910e
  if (socVersion >= SocVersion::ASCEND910B && socVersion <= SocVersion::ASCEND910E) {
    return (CheckType(self->GetDataType(), AICORE_DTYPE_SUPPORT_LIST) ||
            self->GetDataType() == op::DataType::DT_BF16);
  }
  // 910、310及其他
  return CheckType(self->GetDataType(), AICORE_DTYPE_SUPPORT_LIST);
}
 
// AICORE算子kernel
static const aclTensor *PowAiCore(const aclTensor *self, const aclTensor *exponent,
                                  aclTensor *powOut, aclOpExecutor *executor) {
  L0_DFX(PowAiCore, self, exponent, powOut);

  auto ret = ADD_TO_LAUNCHER_LIST_AICORE(Pow,
                              OP_INPUT(self, exponent),
                              OP_OUTPUT(powOut));
  OP_CHECK(
      ret == ACL_SUCCESS, OP_LOGE(ACLNN_ERR_INNER_NULLPTR, "PowAiCore ADD_TO_LAUNCHER_LIST_AICORE failed."),
      return nullptr);
  return powOut;
}
 
// AICPU算子kernel
static const aclTensor *PowAiCpu(const aclTensor *self, const aclTensor *exponent,
                                 aclTensor *powOut, aclOpExecutor *executor) {
  L0_DFX(PowAiCpu, self, exponent, powOut);
  // 使用框架宏ADD_TO_LAUNCHER_LIST_AICORE，将AiCpu Pow算子加入任务队列
  // pow是算子的OpType，self、exponent是算子的输入，powOut是算子的输出
  static internal::AicpuTaskSpace space("Pow", ge::DEPEND_IN_SHAPE, false);
  auto ret = ADD_TO_LAUNCHER_LIST_AICPU(Pow,
                                        OP_ATTR_NAMES(),
                                        OP_INPUT(self, exponent),
                                        OP_OUTPUT(powOut));
  CHECK_RET(ret == ACLNN_SUCCESS, nullptr);
  return powOut;
}
 
const aclTensor *Pow(const aclTensor *self, const aclTensor *exponent, aclOpExecutor *executor) {
  op::Shape broadcastShape;
  if (!BroadcastInferShape(self->GetViewShape(), exponent->GetViewShape(), broadcastShape)) {
    OP_LOGE(ACLNN_ERR_PARAM_INVALID, "Broadcast %s and %s failed.", op::ToString(self->GetViewShape()).GetString(),
            op::ToString(exponent->GetViewShape()).GetString());
    return nullptr;
  }

  auto powOut = executor->AllocTensor(broadcastShape, self->GetDataType());
  CHECK_RET(powOut != nullptr, nullptr);
  if (IsAiCoreSupport(self)) {
    return PowAiCore(self, exponent, powOut, executor);
  } else {
    return PowAiCpu(self, exponent, powOut, executor);
  }
}

// logspace算子特定场景使用
const aclTensor* InplacePow(const aclTensor* self, const aclTensor* exponent, aclOpExecutor* executor)
{
    auto powOut = const_cast<aclTensor*>(exponent);
    CHECK_RET(powOut != nullptr, nullptr);
    if (IsAiCoreSupport(self)) {
        return PowAiCore(self, exponent, powOut, executor);
    } else {
        return PowAiCpu(self, exponent, powOut, executor);
    }
}
}
