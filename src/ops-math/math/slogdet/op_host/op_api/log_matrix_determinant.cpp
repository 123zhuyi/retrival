/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include "log_matrix_determinant.h"

#include "opdev/aicpu/aicpu_task.h"
#include "opdev/make_op_executor.h"
#include "opdev/op_def.h"
#include "opdev/op_dfx.h"
#include "opdev/op_executor.h"
#include "opdev/op_log.h"
#include "opdev/shape_utils.h"

using namespace op;

namespace l0op {
OP_TYPE_REGISTER(LogMatrixDeterminant);

static inline bool IsAiCoreSupport(const aclTensor *) {
   // LogMatrixDeterminant does not currently support AiCore
   return false;
}

// AICPU算子kernel
static inline const std::tuple<const aclTensor *, const aclTensor *> LogMatrixDeterminantAiCpu(
   const aclTensor *self, const aclTensor *signOut, const aclTensor *logOut, aclOpExecutor *executor) {
   // 使用框架宏ADD_TO_LAUNCHER_LIST_AICPU，将Less算子加入任务队列
   L0_DFX(LogMatrixDeterminantAiCpu, self, signOut, logOut);

   static internal::AicpuTaskSpace space("LogMatrixDeterminant", ge::DEPEND_IN_SHAPE, true);
   auto ret = ADD_TO_LAUNCHER_LIST_AICPU(LogMatrixDeterminant, OP_ATTR_NAMES(), OP_INPUT(self),
                                         OP_OUTPUT(signOut, logOut));
   OP_CHECK(ret == ACLNN_SUCCESS, OP_LOGE(ACLNN_ERR_INNER_NULLPTR, "LogMatrixDeterminantAiCpu ADD_TO_LAUNCHER_LIST_AICPU failed."),
            return std::tuple(nullptr, nullptr));
   return std::tuple(signOut, logOut);
}

const std::tuple<const aclTensor *, const aclTensor *> LogMatrixDeterminant(const aclTensor *self,
                                                                         aclOpExecutor *executor) {
   L0_DFX(LogMatrixDeterminant, self);
   auto shapeVec = ToShapeVector(self->GetViewShape());
   shapeVec.pop_back();
   shapeVec.pop_back();
   op::Shape targetShape;
   ToShape(shapeVec, targetShape);

   auto signOut = executor->AllocTensor(targetShape, self->GetDataType());
   CHECK_RET(signOut != nullptr, std::tuple(nullptr, nullptr));
   auto logOut = executor->AllocTensor(targetShape, self->GetDataType());
   CHECK_RET(logOut != nullptr, std::tuple(nullptr, nullptr));

   if (IsAiCoreSupport(self)) {
       return std::tuple(nullptr, nullptr);
   } else {
       return LogMatrixDeterminantAiCpu(self, signOut, logOut, executor);
   }
}
}  // namespace l0op
