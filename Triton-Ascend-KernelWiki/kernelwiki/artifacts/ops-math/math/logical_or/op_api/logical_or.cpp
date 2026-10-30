/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include "logical_or.h"
#include "opdev/aicpu/aicpu_task.h"
#include "opdev/make_op_executor.h"
#include "opdev/op_def.h"
#include "opdev/op_dfx.h"
#include "opdev/op_executor.h"
#include "opdev/op_log.h"
#include "opdev/shape_utils.h"
#include "op_api/aclnn_check.h"

using namespace op;

namespace l0op {

OP_TYPE_REGISTER(LogicalOr);
 
static const std::initializer_list<op::DataType> AICORE_DTYPE_SUPPORT_LIST = {
    op::DataType::DT_BOOL, op::DataType::DT_INT8, op::DataType::DT_UINT8,
    op::DataType::DT_INT16, op::DataType::DT_INT32, op::DataType::DT_INT64,
    op::DataType::DT_FLOAT16, op::DataType::DT_BF16, op::DataType::DT_FLOAT};

// 根据芯片类型、dtype判断算子是否支持走AiCore
static bool IsAiCoreSupport(const aclTensor *self) {
  // regbase版本支持AICORE_DTYPE_SUPPORT_LIST中的所有dtype
  // non-regbase版本仅支持bool类型走AiCore
  if (IsRegBase()) {
    return CheckType(self->GetDataType(), AICORE_DTYPE_SUPPORT_LIST);
  } else {
    return self->GetDataType() == op::DataType::DT_BOOL;
  }
}
 
// AICORE算子kernel
static const aclTensor *LogicalOrAiCore(const aclTensor *self, const aclTensor *other, aclTensor *logical_orOut,
                                         aclOpExecutor *executor) {
  L0_DFX(LogicalOrAiCore, self, other, logical_orOut);
  // 使用框架宏ADD_TO_LAUNCHER_LIST_AICORE，将AiCore LogicalOr算子加入任务队列
  // LogicalOr是算子的OpType，self、other是算子的输入，logical_orOut是算子的输出
  auto ret = ADD_TO_LAUNCHER_LIST_AICORE(LogicalOr, OP_INPUT(self, other), OP_OUTPUT(logical_orOut));
  OP_CHECK(ret == ACLNN_SUCCESS, OP_LOGE(ACLNN_ERR_INNER_NULLPTR, "LogicalOrAiCore ADD_TO_LAUNCHER_LIST_AICORE failed."), return nullptr);
  return logical_orOut;
}
 
// AICPU算子kernel
static const aclTensor *LogicalOrAiCpu(const aclTensor *self, const aclTensor *other, aclTensor *logical_orOut,
                                        aclOpExecutor *executor) {
  // 使用框架宏ADD_TO_LAUNCHER_LIST_AICPU，将AiCpu LogicalOr算子加入任务队列
  // LogicalOr是算子的OpType，self、other是算子的输入，logical_orOut是算子的输出
  L0_DFX(LogicalOrAiCpu, self, other);

  static internal::AicpuTaskSpace space("LogicalOr");
  auto ret = ADD_TO_LAUNCHER_LIST_AICPU(LogicalOr, OP_ATTR_NAMES(), OP_INPUT(self, other), OP_OUTPUT(logical_orOut));
  OP_CHECK(ret == ACLNN_SUCCESS, OP_LOGE(ACLNN_ERR_INNER_NULLPTR, "LogicalOrAiCpu ADD_TO_LAUNCHER_LIST_AICPU failed."), return nullptr);
  return logical_orOut;
}
 
const aclTensor *LogicalOr(const aclTensor *self, const aclTensor *other, aclOpExecutor *executor) {
  op::Shape broadcastShape;
  if (!BroadcastInferShape(self->GetViewShape(), other->GetViewShape(), broadcastShape)) {
    OP_LOGE(ACLNN_ERR_PARAM_INVALID, "Broadcast %s and %s failed.", op::ToString(self->GetViewShape()).GetString(),
            op::ToString(other->GetViewShape()).GetString());
    return nullptr;
  }

  auto logical_orOut = executor->AllocTensor(broadcastShape, op::DataType::DT_BOOL);
  if (IsAiCoreSupport(self)) {
    return LogicalOrAiCore(self, other, logical_orOut, executor);
  } else {
    return LogicalOrAiCpu(self, other, logical_orOut, executor);
  }
}

}  // namespace l0op

