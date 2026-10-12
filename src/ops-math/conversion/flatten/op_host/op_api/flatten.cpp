/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include "flatten.h"
#include "opdev/aicpu/aicpu_task.h"
#include "opdev/make_op_executor.h"
#include "opdev/op_def.h"
#include "opdev/op_dfx.h"
#include "opdev/op_executor.h"
#include "opdev/op_log.h"
#include "opdev/shape_utils.h"
#include "opdev/platform.h"

using namespace op;

namespace l0op {

OP_TYPE_REGISTER(Flatten);

static const std::initializer_list<op::DataType> ASCEND910_AICORE_DTYPE_SUPPORT_LIST = {
    op::DataType::DT_FLOAT, op::DataType::DT_FLOAT16, op::DataType::DT_INT32, op::DataType::DT_INT8,
    op::DataType::DT_UINT8, op::DataType::DT_INT64, op::DataType::DT_INT16, op::DataType::DT_UINT16,
    op::DataType::DT_UINT32, op::DataType::DT_UINT64, op::DataType::DT_BOOL};

static const std::initializer_list<op::DataType> ASCEND910B_AICORE_DTYPE_SUPPORT_LIST = {
    op::DataType::DT_FLOAT, op::DataType::DT_FLOAT16, op::DataType::DT_INT32, op::DataType::DT_INT8,
    op::DataType::DT_UINT8, op::DataType::DT_INT64, op::DataType::DT_INT16, op::DataType::DT_UINT16,
    op::DataType::DT_UINT32, op::DataType::DT_UINT64, op::DataType::DT_BOOL, op::DataType::DT_BF16};

// 根据芯片类型、dtype判断算子是否支持走AiCore
static bool IsAiCoreSupport(const aclTensor *self) {
  SocVersion version = GetCurrentPlatformInfo().GetSocVersion();
  // 获取芯片类型，判断是910B还是910
  if (version == SocVersion::ASCEND910) {
    return CheckType(self->GetDataType(), ASCEND910_AICORE_DTYPE_SUPPORT_LIST);
  }
  // 910、310芯片
  return CheckType(self->GetDataType(), ASCEND910B_AICORE_DTYPE_SUPPORT_LIST);
}

// AICORE算子kernel
static const aclTensor* FlattenAiCore(const aclTensor *self, int64_t axis, aclTensor *out, aclOpExecutor *executor) {
  L0_DFX(FlattenAiCore, self, axis, out);
  // 使用框架宏ADD_TO_LAUNCHER_LIST_AICORE，将AiCore FlattenV2算子加入任务队列
  auto ret = ADD_TO_LAUNCHER_LIST_AICORE(Flatten,
                                         OP_INPUT(self),
                                         OP_OUTPUT(out),
                                         OP_ATTR(axis));
  OP_CHECK(ret ==  ACLNN_SUCCESS, OP_LOGE(ACLNN_ERR_INNER_NULLPTR, "FlattenAiCore ADD_TO_LAUNCHER_LIST_AICORE failed."),
    return nullptr);
  return out;
}

const aclTensor* Flatten(const aclTensor *self, aclOpExecutor *executor, int64_t axis) {
  // Flatten无aicpu,dtype不在support list内直接返回nullptr
  if (!IsAiCoreSupport(self)) {
    return nullptr;
  }

  // 处理axis为负数的场景,axis默认值为1，axis超出self范围，在l2校验
  // 构造out
  op::Shape outShape;
  int64_t dim0 = 1;
  int64_t dim1 = 1;
  size_t selfDim = self->GetViewShape().GetDimNum();
  for (int64_t i = 0; i < axis; i++) {
      dim0 *= self->GetViewShape().GetDim(i);
  }
  for (size_t i = axis; i < selfDim; i++) {
      dim1 *= self->GetViewShape().GetDim(i);
  }
  outShape.AppendDim(dim0);
  outShape.AppendDim(dim1);
  auto flattenOut = executor->AllocTensor(outShape, self->GetDataType(), op::Format::FORMAT_ND);
  if (flattenOut == nullptr) {
    OP_LOGE(ACLNN_ERR_INNER_NULLPTR, "[Flatten] alloc out tensor failed.");
    return flattenOut;
  }

  return FlattenAiCore(self, axis, flattenOut, executor);
}
}
