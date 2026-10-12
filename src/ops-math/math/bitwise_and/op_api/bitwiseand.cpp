/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include "bitwiseand.h"
#include "opdev/data_type_utils.h"
#include "opdev/make_op_executor.h"
#include "opdev/op_def.h"
#include "opdev/op_dfx.h"
#include "opdev/op_executor.h"
#include "opdev/op_log.h"
#include "opdev/shape_utils.h"
#include "opdev/aicpu/aicpu_task.h"
#include "op_api/aclnn_check.h"

using namespace op;

namespace l0op {

OP_TYPE_REGISTER(BitwiseAnd);

static const std::initializer_list<op::DataType> AICORE_DTYPE_SUPPORT_LIST = {
    op::DataType::DT_INT16, op::DataType::DT_UINT16, op::DataType::DT_INT32};

static const std::initializer_list<op::DataType> ASCEND910B_AICORE_DTYPE_SUPPORT_LIST = {
    op::DataType::DT_INT64, op::DataType::DT_UINT16, op::DataType::DT_INT32, op::DataType::DT_INT16};

// 根据芯片类型、dtype判断算子是否支持走aicore
static bool IsAiCoreSupport(const aclTensor *self) {
  // 910B、910_93、regbase
  auto curArch = GetCurrentPlatformInfo().GetCurNpuArch();
  if (curArch == NpuArch::DAV_2201 || IsRegBase(curArch)) {
    return CheckType(self->GetDataType(), ASCEND910B_AICORE_DTYPE_SUPPORT_LIST);
  }
  // 910及其他
  return CheckType(self->GetDataType(), AICORE_DTYPE_SUPPORT_LIST);
}

// AICORE算子kernel
static const aclTensor *BitwiseAndAiCore(const aclTensor *self, const aclTensor *other, const aclTensor *BitwiseAndOut,
                                  aclOpExecutor *executor) {
  L0_DFX(BitwiseAndAiCore);
  // 使用框架宏ADD_TO_LAUNCHER_LIST，将AiCore BitwiseAnd算子加入任务队列
  // BitwiseAnd是算子的OpType，self、other是算子的输入，BitwiseAndOut是算子的输出
  auto ret = ADD_TO_LAUNCHER_LIST_AICORE(BitwiseAnd, OP_INPUT(self, other), OP_OUTPUT(BitwiseAndOut));
  OP_CHECK(ret == ACL_SUCCESS, OP_LOGE(ACLNN_ERR_INNER_NULLPTR, "BitwiseAndAiCore ADD_TO_LAUNCHER_LIST_AICORE failed."), return nullptr);
  return BitwiseAndOut;
}

// AICPU算子kernel
static const aclTensor *BitwiseAndAiCpu(const aclTensor *self, const aclTensor *other, aclTensor *BitwiseAndOut,
                                 aclOpExecutor *executor) {
  // 使用框架宏ADD_TO_LAUNCHER_LIST，将AiCpu BitwiseAnd算子加入任务队列
  // BitwiseAnd是算子的OpType，self、other是算子的输入，BitwiseAndOut是算子的输出
  L0_DFX(BitwiseAndAiCpu);
  static internal::AicpuTaskSpace space("BitwiseAnd", ge::DEPEND_IN_SHAPE, true);
  auto ret = ADD_TO_LAUNCHER_LIST_AICPU(BitwiseAnd, OP_ATTR_NAMES(), OP_INPUT(self, other), OP_OUTPUT(BitwiseAndOut));
  OP_CHECK(ret ==  ACL_SUCCESS, OP_LOGE(ACLNN_ERR_INNER_NULLPTR, "BitwiseAndAiCpu ADD_TO_LAUNCHER_LIST_AICPU failed."), return nullptr);
  return BitwiseAndOut;
}

const aclTensor *BitwiseAnd(const aclTensor *self, const aclTensor *other, aclOpExecutor *executor) {
  // 根据输入shape推导输出shape
  op::Shape broadcastShape;
  if (!BroadcastInferShape(self->GetViewShape(), other->GetViewShape(), broadcastShape)) {
    OP_LOGE(ACLNN_ERR_PARAM_INVALID, "Broadcast %s and %s failed.", op::ToString(self->GetViewShape()).GetString(),
            op::ToString(other->GetViewShape()).GetString());
    return nullptr;
  }
  // 根据输出shape申请输出tensor
  auto BitwiseAndOut = executor->AllocTensor(broadcastShape, self->GetDataType(), op::Format::FORMAT_ND);
  if (IsAiCoreSupport(self)) {
    return BitwiseAndAiCore(self, other, BitwiseAndOut, executor);
  } else {
    return BitwiseAndAiCpu(self, other, BitwiseAndOut, executor);
  }
  return BitwiseAndOut;
}

}  // namespace l0op
