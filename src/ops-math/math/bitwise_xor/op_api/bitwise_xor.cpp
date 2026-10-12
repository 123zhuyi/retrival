/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include "bitwise_xor.h"
#include "op_api/aclnn_check.h"
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
OP_TYPE_REGISTER(BitwiseXor);

static const std::initializer_list<DataType> AICORE_DTYPE_SUPPORT_LIST = {
  DataType::DT_INT16, DataType::DT_INT32, DataType::DT_UINT16};

static const std::initializer_list<DataType> ASCEND910B_AICORE_DTYPE_SUPPORT_LIST = {
  DataType::DT_INT16, DataType::DT_INT32, DataType::DT_UINT16, DataType::DT_INT64};

static const std::initializer_list<DataType> ASCEND950_AICORE_DTYPE_SUPPORT_LIST = {
  DataType::DT_INT8, DataType::DT_INT16, DataType::DT_UINT8, DataType::DT_UINT16,
  DataType::DT_INT32, DataType::DT_UINT32, DataType::DT_INT64, DataType::DT_UINT64};
 
// 根据芯片类型、dtype判断算子是否支持走aicore
static inline bool IsAiCoreSupport(const aclTensor *self) {
  if (GetCurrentPlatformInfo().GetSocVersion() == SocVersion::ASCEND910B ||
      GetCurrentPlatformInfo().GetSocVersion() == SocVersion::ASCEND910_93) {
    return CheckType(self->GetDataType(), ASCEND910B_AICORE_DTYPE_SUPPORT_LIST);
  }
  if (IsRegBase()) {
    return CheckType(self->GetDataType(), ASCEND950_AICORE_DTYPE_SUPPORT_LIST);
  }
  return CheckType(self->GetDataType(), AICORE_DTYPE_SUPPORT_LIST);
}

// AICORE算子kernel
static inline const aclTensor *BitwiseXorAiCore(const aclTensor *self, const aclTensor *other, aclTensor *out,
                                                aclOpExecutor *executor) {
  L0_DFX(BitwiseXorAiCore, self, other, out);
  // 使用框架宏ADD_TO_LAUNCHER_LIST_AICORE，将AiCore BitwiseXor算子加入任务队列
  // BitwiseXor是算子的OpType，self, other是算子的输入，out是算子的输出
  auto ret = ADD_TO_LAUNCHER_LIST_AICORE(BitwiseXor, OP_INPUT(self, other), OP_OUTPUT(out));
  OP_CHECK(ret == ACL_SUCCESS, OP_LOGE(ACLNN_ERR_INNER_NULLPTR, "BitwiseXorAiCore ADD_TO_LAUNCHER_LIST_AICORE failed."), return nullptr);
  return out;
}

// AICPU算子kernel
static inline const aclTensor *BitwiseXorAiCpu(const aclTensor *self, const aclTensor *other, aclTensor *out,
                                               aclOpExecutor *executor) {
  L0_DFX(BitwiseXorAiCpu, self, other, out);

  static internal::AicpuTaskSpace space("BitwiseXor", ge::DEPEND_IN_SHAPE, true);

  // 使用框架宏ADD_TO_LAUNCHER_LIST_AICPU，将AiCpu BitwiseXor算子加入任务队列
  // BitwiseXor是算子的OpType，self, other是算子的输入，out是算子的输出
  auto ret = ADD_TO_LAUNCHER_LIST_AICPU(BitwiseXor, OP_ATTR_NAMES(), OP_INPUT(self, other), OP_OUTPUT(out));
  OP_CHECK(ret ==  ACL_SUCCESS, OP_LOGE(ACLNN_ERR_INNER_NULLPTR, "BitwiseXorAiCpu ADD_TO_LAUNCHER_LIST_AICPU failed."), return nullptr);
  return out;
}

const aclTensor *BitwiseXor(const aclTensor *self, const aclTensor *other, aclOpExecutor *executor) {
  // 根据输入shape推导输出shape
  Shape broadcastShape;
  if (!BroadcastInferShape(self->GetViewShape(), other->GetViewShape(), broadcastShape)) {
    OP_LOGE(ACLNN_ERR_PARAM_INVALID, "Broadcast %s and %s failed.", ToString(self->GetViewShape()).GetString(),
            ToString(other->GetViewShape()).GetString());
    return nullptr;
  }
  // 根据输出shape申请输出tensor
  auto out = executor->AllocTensor(broadcastShape, self->GetDataType(), Format::FORMAT_ND);
  if (out == nullptr) {
    OP_LOGE(ACLNN_ERR_INNER_NULLPTR, "alloc out tensor failed.");
    return nullptr;
  }
  if (IsAiCoreSupport(self)) {
    return BitwiseXorAiCore(self, other, out, executor);
  } else {
    return BitwiseXorAiCpu(self, other, out, executor);
  }
}
}  // namespace l0op