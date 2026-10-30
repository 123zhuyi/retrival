/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include "digamma.h"
#include "opdev/aicpu/aicpu_task.h"
#include "opdev/op_log.h"
#include "opdev/op_dfx.h"
#include "opdev/shape_utils.h"
#include "opdev/make_op_executor.h"

using namespace op;

namespace l0op {

OP_TYPE_REGISTER(Digamma);

inline static bool DigammaInferShape(const op::Shape& selfShape, op::Shape& outShape) {
  outShape = selfShape;
  return true;
}

// AICPU算子kernel
static const aclTensor* DigammaAiCpu(const aclTensor* self, const aclTensor* out, aclOpExecutor* executor) {
  L0_DFX(DigammaAiCpu, self, out);

  // 使用框架宏ADD_TO_LAUNCHER_LIST_AICPU，将AiCpu Digamma算子加入任务队列
  // Digamma是算子的OpType，self是算子的输入，out是算子的输出
  static internal::AicpuTaskSpace space("Digamma", ge::DEPEND_IN_SHAPE, true);
  auto ret = ADD_TO_LAUNCHER_LIST_AICPU(Digamma, OP_ATTR_NAMES(), OP_INPUT(self), OP_OUTPUT(out));
  OP_CHECK(ret ==  ACL_SUCCESS, OP_LOGE(ACLNN_ERR_INNER_NULLPTR, "DigammaAiCpu ADD_TO_LAUNCHER_LIST_AICPU failed."), return nullptr);
  return out;
}

const aclTensor* Digamma(const aclTensor* self, aclOpExecutor* executor) {
  Shape outShape;
  if (!DigammaInferShape(self->GetViewShape(), outShape)) {
    OP_LOGE(ACL_ERROR_INVALID_PARAM, "infer shape failed.");
    return nullptr;
  }

  auto out = executor->AllocTensor(outShape, self->GetDataType());
  return DigammaAiCpu(self, out, executor);  
}

}  // namespace l0op