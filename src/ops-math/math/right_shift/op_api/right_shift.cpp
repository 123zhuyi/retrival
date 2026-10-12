/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include "right_shift.h"
#include "aclnn_kernels/transpose.h"
#include "opdev/make_op_executor.h"
#include "opdev/aicpu/aicpu_task.h"
#include "opdev/op_def.h"
#include "opdev/op_dfx.h"
#include "opdev/op_executor.h"
#include "opdev/op_log.h"
#include "opdev/shape_utils.h"
#include "opdev/platform.h"
#include "op_api/aclnn_check.h"

using namespace op;

namespace l0op {
OP_TYPE_REGISTER(RightShift);

static const std::initializer_list<op::DataType> DTYPE_SUPPORT_LIST = {
   op::DataType::DT_INT8, op::DataType::DT_INT16,
   op::DataType::DT_INT32, op::DataType::DT_INT64,
   op::DataType::DT_UINT8, op::DataType::DT_UINT16,
   op::DataType::DT_UINT32, op::DataType::DT_UINT64};

bool IsSupported(const aclTensor* x, const aclTensor* y) {
   if (!op::CheckType(x->GetDataType(), DTYPE_SUPPORT_LIST)) {
       return false;
   }
   if (!op::CheckType(y->GetDataType(), DTYPE_SUPPORT_LIST)) {
       return false;
   }
   return true;
}

const aclTensor* RightShiftAiCore(const aclTensor* x, const aclTensor* y, const aclTensor* z, aclOpExecutor* executor) {
   L0_DFX(RightShiftAiCore, x, y, z);

   auto ret = ADD_TO_LAUNCHER_LIST_AICORE(RightShift, OP_INPUT(x, y), OP_OUTPUT(z));
   OP_CHECK(
       ret == ACLNN_SUCCESS, OP_LOGE(ACLNN_ERR_INNER_NULLPTR, "RightShiftAiCore ADD_TO_LAUNCHER_LIST_AICORE failed."),
       return nullptr);
   return z;
}

const aclTensor* RightShiftAiCpu(const aclTensor* x, const aclTensor* y, const aclTensor* z, aclOpExecutor* executor) {
   L0_DFX(RightShiftAiCpu, x, y, z);

   static internal::AicpuTaskSpace space("RightShift");
   auto ret = ADD_TO_LAUNCHER_LIST_AICPU(RightShift, OP_ATTR_NAMES(), OP_INPUT(x, y), OP_OUTPUT(z));
   OP_CHECK(
       ret == ACLNN_SUCCESS, OP_LOGE(ACLNN_ERR_INNER_NULLPTR, "RightShiftAiCpu ADD_TO_LAUNCHER_LIST_AICPU failed."),
       return nullptr);
   return z;
}

const aclTensor* RightShift(const aclTensor* x, const aclTensor* y, aclOpExecutor* executor) {
   Shape broadcastShape;
   if (!BroadcastInferShape(x->GetViewShape(), y->GetViewShape(), broadcastShape)) {
        OP_LOGE(
            ACLNN_ERR_PARAM_INVALID, "Broadcast %s and %s failed.",
            op::ToString(x->GetViewShape()).GetString(), op::ToString(y->GetViewShape()).GetString());
        return nullptr;
   }
   auto z = executor->AllocTensor(broadcastShape, x->GetDataType());
   CHECK_RET(z != nullptr, nullptr);
   if (!IsSupported(x, y)) {
       OP_LOGE(ACLNN_ERR_PARAM_INVALID, "The input dtype for rightshift is not supported.");
       return nullptr;
   }
   if (!IsRegBase()) {
       return RightShiftAiCpu(x, y, z, executor);
   } else {
       return RightShiftAiCore(x, y, z, executor);
   }
}
}  // namespace l0op