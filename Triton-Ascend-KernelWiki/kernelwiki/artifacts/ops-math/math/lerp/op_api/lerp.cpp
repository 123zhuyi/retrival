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
 * \file lerp.cpp
 * \brief
 */

#include "lerp.h"
#include "aclnn_kernels/reshape.h"
#include "opdev/make_op_executor.h"
#include "opdev/op_dfx.h"
#include "opdev/op_log.h"
#include "opdev/shape_utils.h"

using namespace op;
namespace l0op {
OP_TYPE_REGISTER(Lerp);

const aclTensor* Lerp(const aclTensor* self, const aclTensor* other, const aclTensor* weight, aclOpExecutor* executor) {
  L0_DFX(Lerp, self, other, weight);
  op::Shape broadcastShape;
  if (!BroadcastInferShape(self->GetViewShape(), other->GetViewShape(), broadcastShape)) {
    OP_LOGE(ACLNN_ERR_PARAM_INVALID, "Broadcast %s and %s failed.", op::ToString(self->GetViewShape()).GetString(),
            op::ToString(other->GetViewShape()).GetString());
    return nullptr;
  }
  op::Shape outBroadcastShape;
  if (!BroadcastInferShape(weight->GetViewShape(), broadcastShape, outBroadcastShape)) {
    OP_LOGE(ACLNN_ERR_PARAM_INVALID, "Broadcast %s and %s failed.", op::ToString(self->GetViewShape()).GetString(),
            op::ToString(broadcastShape).GetString());
    return nullptr;
  }

  auto out = executor->AllocTensor(outBroadcastShape, self->GetDataType());
  auto ret = ADD_TO_LAUNCHER_LIST_AICORE(Lerp, op::AI_CORE, OP_INPUT(self, other, weight), OP_OUTPUT(out));
  OP_CHECK(ret ==  ACL_SUCCESS, OP_LOGE(ACLNN_ERR_INNER_NULLPTR, "Lerp ADD_TO_LAUNCHER_LIST_AICORE failed."), return nullptr);
  return out;
}
}  // namespace l0op
