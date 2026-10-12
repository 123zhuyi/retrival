/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include "logaddexp.h"
#include "opdev/make_op_executor.h"
#include "opdev/op_def.h"
#include "opdev/op_dfx.h"
#include "opdev/op_executor.h"
#include "opdev/op_log.h"
#include "opdev/shape_utils.h"

using namespace op;

namespace l0op {

OP_TYPE_REGISTER(LogAddExp);

static const std::initializer_list<op::DataType> AICORE_DTYPE_SUPPORT_LIST = {
    op::DataType::DT_FLOAT, op::DataType::DT_FLOAT16, op::DataType::DT_BF16};

static bool IsAiCoreSupport(const aclTensor* self) {
  return CheckType(self->GetDataType(), AICORE_DTYPE_SUPPORT_LIST);
}

static const aclTensor* LogAddExpAiCore(const aclTensor* self, const aclTensor* other, const float base,
                                        const float scale, const float shift, aclTensor* out, aclOpExecutor* executor) {
  L0_DFX(LogAddExpAiCore, self, other, base, scale, shift, out);

  auto ret = ADD_TO_LAUNCHER_LIST_AICORE(LogAddExp, OP_INPUT(self, other), OP_OUTPUT(out), OP_ATTR(base, scale, shift));
  OP_CHECK(ret ==  ACL_SUCCESS, OP_LOGE(ACLNN_ERR_INNER_NULLPTR, "LogAddExpAiCore ADD_TO_LAUNCHER_LIST_AICORE failed."), return nullptr);
  return out;
}

const aclTensor* LogAddExp(const aclTensor* self, const aclTensor* other, const float base, const float scale,
                           const float shift, aclOpExecutor* executor) {
  L0_DFX(LogAddExp, self, other, base, scale, shift);

  aclTensor* out = nullptr;
  op::Shape broadcastShape;
  auto selfShape = self->GetViewShape();
  auto otherShape = other->GetViewShape();
  CHECK_RET(BroadcastInferShape(selfShape, otherShape, broadcastShape), nullptr);
  out = executor->AllocTensor(broadcastShape, self->GetDataType());
  CHECK_RET(out != nullptr, nullptr);

  if (IsAiCoreSupport(self)) {
    return LogAddExpAiCore(self, other, base, scale, shift, out, executor);
  } else {
    return nullptr;
  }
}

}  // namespace l0op