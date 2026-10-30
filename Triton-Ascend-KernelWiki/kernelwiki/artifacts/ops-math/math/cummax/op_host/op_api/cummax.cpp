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
 * \file cummax.cpp
 * \brief
 */

#include "cummax.h"
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
OP_TYPE_REGISTER(Cummax);

// AICPU算子kernel
static inline std::tuple<aclTensor*, aclTensor*> CummaxAiCpu(const aclTensor* self, int64_t dim, aclTensor* valuesOut,
                                                             aclTensor* indicesOut, aclOpExecutor* executor) {
  L0_DFX(CummaxAiCpu, self, dim, valuesOut, indicesOut);

  static internal::AicpuTaskSpace space("Cummax");
  // 使用框架宏ADD_TO_LAUNCHER_LIST_AICPU，将AiCpu Cummax算子加入任务队列
  // Cummax是算子的OpType，self是算子的输入，valuesOut、indicesOut是算子的输出，属性dim传入dim
  auto ret = ADD_TO_LAUNCHER_LIST_AICPU(Cummax, OP_ATTR_NAMES({"dim"}), OP_INPUT(self), OP_OUTPUT(valuesOut, indicesOut),
                                        OP_ATTR(dim));
  if (ret != ACL_SUCCESS) {
    OP_LOGE(ACLNN_ERR_INNER_NULLPTR, "CummaxAiCpu ADD_TO_LAUNCHER_LIST_AICPU failed.");
    return std::tuple<aclTensor*, aclTensor*>(nullptr, nullptr);
  }
  return {valuesOut, indicesOut};
}

static inline std::tuple<aclTensor*, aclTensor*> CummaxExec(const aclTensor* self, int64_t dim, aclTensor* valuesOut,
                                                            aclTensor* indicesOut, aclOpExecutor* executor) {
  if (valuesOut == nullptr || indicesOut == nullptr) {
    OP_LOGE(ACLNN_ERR_INNER_NULLPTR, "alloc out tensor failed.");
    return {nullptr, nullptr};
  }
  return CummaxAiCpu(self, dim, valuesOut, indicesOut, executor);
}

std::tuple<aclTensor*, aclTensor*> CummaxOutInt32(const aclTensor* self, int64_t dim, aclOpExecutor* executor) {
  // 根据输入shape申请输出tensor
  auto valuesOut = executor->AllocTensor(self->GetViewShape(), self->GetDataType(), self->GetViewFormat());
  auto indicesOut = executor->AllocTensor(self->GetViewShape(), DataType::DT_INT32, self->GetViewFormat());
  return CummaxExec(self, dim, valuesOut, indicesOut, executor);
}

std::tuple<aclTensor*, aclTensor*> CummaxOutInt64(const aclTensor* self, int64_t dim, aclOpExecutor* executor) {
  // 根据输入shape申请输出tensor
  auto valuesOut = executor->AllocTensor(self->GetViewShape(), self->GetDataType(), self->GetViewFormat());
  auto indicesOut = executor->AllocTensor(self->GetViewShape(), DataType::DT_INT64, self->GetViewFormat());
  return CummaxExec(self, dim, valuesOut, indicesOut, executor);
}
}  // namespace l0op