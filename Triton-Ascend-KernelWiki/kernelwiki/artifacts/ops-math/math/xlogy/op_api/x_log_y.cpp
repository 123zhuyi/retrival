/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include "x_log_y.h"
#include "opdev/make_op_executor.h"
#include "opdev/op_def.h"
#include "opdev/op_dfx.h"
#include "opdev/op_executor.h"
#include "opdev/op_log.h"
#include "opdev/shape_utils.h"

using namespace op;

namespace l0op {
OP_TYPE_REGISTER(Xlogy);

const aclTensor *XLogY(const aclTensor *self, const aclTensor *other, aclOpExecutor *executor) {
  L0_DFX(XLogY, self, other);
  // 根据XLogY算子语义，通过输入shape推导算子输出shape
  op::Shape broadcastShape;
  if (!BroadcastInferShape(self->GetViewShape(), other->GetViewShape(), broadcastShape)) {
    OP_LOGE(ACLNN_ERR_PARAM_INVALID, "Broadcast %s and %s failed.", op::ToString(self->GetViewShape()).GetString(),
            op::ToString(other->GetViewShape()).GetString());
    return nullptr;
  }

  auto out = executor->AllocTensor(broadcastShape, self->GetDataType());
  CHECK_RET(out != nullptr, nullptr);
  // Xlogy是算子的OpType，self是算子的输入，out是算子的输出
  ADD_TO_LAUNCHER_LIST_AICORE(Xlogy, OP_INPUT(self, other), OP_OUTPUT(out));
  return out;
}
}  // namespace l0op
