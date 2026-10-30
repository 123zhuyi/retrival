/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include "axpy.h"
#include "opdev/aicpu/aicpu_task.h"
#include "opdev/data_type_utils.h"
#include "opdev/format_utils.h"
#include "opdev/make_op_executor.h"
#include "opdev/op_def.h"
#include "opdev/op_dfx.h"
#include "opdev/op_executor.h"
#include "opdev/op_log.h"
#include "opdev/shape_utils.h"

using namespace op;

namespace l0op {

OP_TYPE_REGISTER(Axpy);

static const std::initializer_list<op::DataType> AICORE_DTYPE_SUPPORT_LIST = {DataType::DT_FLOAT, DataType::DT_INT32,
                                                                              DataType::DT_FLOAT16, DataType::DT_BF16};
// 根据芯片类型、dtype判断算子是否支持走aicore
static bool IsAiCoreSupport(const aclTensor *self) {
  // Axpy只需要判断dtype
  return CheckType(self->GetDataType(), AICORE_DTYPE_SUPPORT_LIST);
}

// AICORE算子kernel
static const aclTensor *AxpyAiCore(const aclTensor *self, const aclTensor *other, float alpha, const aclTensor *axpyOut,
                                   aclOpExecutor *executor) {
  L0_DFX(AxpyAiCore, self, other, alpha, axpyOut);
  // 使用框架宏ADD_TO_KERNEL_OBJ_LIST，将Axpy算子加入任务队列
  // op::OP_TYPE_Axpy是算子的OpType，self、other是算子的输入，axpyOut是算子的输出，alpha是算子的属性
  auto ret = ADD_TO_LAUNCHER_LIST_AICORE(Axpy, OP_INPUT(self, other), OP_OUTPUT(axpyOut), OP_ATTR(alpha));
  OP_CHECK(ret == ACL_SUCCESS, OP_LOGE(ACLNN_ERR_INNER_NULLPTR, "AxpyAiCore ADD_TO_LAUNCHER_LIST_AICORE failed."), return nullptr);
  return axpyOut;
}

// AICPU算子kernel
static const aclTensor *AxpyAiCpu(const aclTensor *self, const aclTensor *other, float alpha, aclTensor *axpyOut,
                                  aclOpExecutor *executor) {
  // 使用框架宏ADD_AICPU_TO_KERNEL_OBJ_LIST，将Axpy算子加入任务队列
  // op::OP_TYPE_Axpy是算子的OpType，self、other是算子的输入，axpyOut是算子的输出，alpha是算子的属性
  L0_DFX(AxpyAiCpu, self, other, alpha);
  static internal::AicpuTaskSpace space("Axpy");
  auto ret = ADD_TO_LAUNCHER_LIST_AICPU(Axpy, OP_ATTR_NAMES({"alpha"}), OP_INPUT(self, other), OP_OUTPUT(axpyOut),
                                        OP_ATTR(alpha));
  OP_CHECK(ret ==  ACL_SUCCESS, OP_LOGE(ACLNN_ERR_INNER_NULLPTR, "AxpyAiCpu ADD_TO_LAUNCHER_LIST_AICPU failed."), return nullptr);
  return axpyOut;
}

const aclTensor *Axpy(const aclTensor *self, const aclTensor *other, float alpha, aclOpExecutor *executor) {
  op::Shape broadcastShape;
  if (!BroadcastInferShape(self->GetViewShape(), other->GetViewShape(), broadcastShape)) {
    OP_LOGE(ACLNN_ERR_PARAM_INVALID, "Broadcast %s and %s failed.", op::ToString(self->GetViewShape()).GetString(),
            op::ToString(other->GetViewShape()).GetString());
    return nullptr;
  }

  auto axpyOut = executor->AllocTensor(broadcastShape, self->GetDataType());

  if (IsAiCoreSupport(self)) {
    return AxpyAiCore(self, other, alpha, axpyOut, executor);
  } else {
    return AxpyAiCpu(self, other, alpha, axpyOut, executor);
  }
}

const aclTensor *AxpyInplace(const aclTensor *self, const aclTensor *other, float alpha, aclOpExecutor *executor) {
  op::Shape broadcastShape;
  if (!BroadcastInferShape(self->GetViewShape(), other->GetViewShape(), broadcastShape)) {
    OP_LOGE(ACLNN_ERR_PARAM_INVALID, "Broadcast %s and %s failed.", op::ToString(self->GetViewShape()).GetString(),
            op::ToString(other->GetViewShape()).GetString());
    return nullptr;
  }
  
  // 校验输出tensor的shape和other tensor一致
  if (broadcastShape != other->GetViewShape()) {
    OP_LOGE(ACLNN_ERR_PARAM_INVALID, "self and other broadcastShape [%s] not equal to other shape [%s], do no support inplace from the 'other' tensor!", 
      op::ToString(broadcastShape).GetString(), op::ToString(other->GetViewShape()).GetString());
      return nullptr;
  }

  auto axpyOut = const_cast<aclTensor *>(other);

  if (IsAiCoreSupport(self)) {
    return AxpyAiCore(self, other, alpha, axpyOut, executor);
  } else {
    return AxpyAiCpu(self, other, alpha, axpyOut, executor);
  }
}
}  // namespace l0op
