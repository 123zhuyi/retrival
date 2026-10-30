/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include "atan2.h"
#include "opdev/aicpu/aicpu_task.h"
#include "opdev/make_op_executor.h"
#include "opdev/op_def.h"
#include "opdev/op_dfx.h"
#include "opdev/op_executor.h"
#include "opdev/op_log.h"
#include "opdev/shape_utils.h"

using namespace op;
namespace l0op {
OP_TYPE_REGISTER(Atan2);

static const std::initializer_list<DataType> AICORE_DTYPE_SUPPORT_LIST = {
  DataType::DT_FLOAT, DataType::DT_FLOAT16, DataType::DT_BF16};

// 根据芯片类型、dtype判断算子是否支持走aicore
static inline bool IsAiCoreSupport(DataType selfDtype) {
  // 只需要判断dtype
  return CheckType(selfDtype, AICORE_DTYPE_SUPPORT_LIST);
}

// AICORE算子kernel
static inline const aclTensor* Atan2AiCore(const aclTensor* self, const aclTensor* other, aclTensor* out,
                                           aclOpExecutor* executor) {
  L0_DFX(Atan2AiCore, self, other, out);
  auto ret = ADD_TO_LAUNCHER_LIST_AICORE(Atan2, OP_INPUT(self, other), OP_OUTPUT(out));
  OP_CHECK(ret == ACLNN_SUCCESS, OP_LOGE(ACLNN_ERR_INNER_NULLPTR, "Atan2 ADD_TO_LAUNCHER_LIST_AICORE failed."), return nullptr);
  return out;
}

// AICPU算子kernel
static inline const aclTensor* Atan2AiCpu(const aclTensor* self, const aclTensor* other, aclTensor* out,
                                          aclOpExecutor* executor) {
  L0_DFX(Atan2AiCpu, self, other, out);
  static internal::AicpuTaskSpace space("Atan2", ge::DEPEND_IN_SHAPE, true);
  auto ret = ADD_TO_LAUNCHER_LIST_AICPU(Atan2, OP_ATTR_NAMES(), OP_INPUT(self, other), OP_OUTPUT(out));
  OP_CHECK(ret ==  ACL_SUCCESS, OP_LOGE(ACLNN_ERR_INNER_NULLPTR, "Atan2AiCpu ADD_TO_LAUNCHER_LIST_AICPU failed."), return nullptr);
  return out;
}

const aclTensor* Atan2(const aclTensor* self, const aclTensor* other, aclOpExecutor* executor) {
  Shape broadcastShape;
  BroadcastInferShape(self->GetViewShape(), other->GetViewShape(), broadcastShape);
  auto out = executor->AllocTensor(broadcastShape, self->GetDataType());

  if (IsAiCoreSupport(self->GetDataType())) {
    return Atan2AiCore(self, other, out, executor);
  }
  return Atan2AiCpu(self, other, out, executor);
}
}  // namespace l0op
