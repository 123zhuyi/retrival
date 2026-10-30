/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

/*!
 * \file cummin.cpp
 * \brief
 */
#include "cummin.h"
#include "opdev/aicpu/aicpu_task.h"
#include "opdev/data_type_utils.h"
#include "opdev/format_utils.h"
#include "opdev/make_op_executor.h"
#include "opdev/op_def.h"
#include "opdev/op_dfx.h"
#include "opdev/op_executor.h"
#include "opdev/op_log.h"
#include "opdev/shape_utils.h"
#include "op_api/aclnn_check.h"

using namespace op;

namespace l0op {
OP_TYPE_REGISTER(Cummin);

static const std::initializer_list<DataType> AICORE_DTYPE_SUPPORT_LIST = {
    DataType::DT_FLOAT, DataType::DT_FLOAT16, DataType::DT_INT8, DataType::DT_UINT8,
    DataType::DT_INT32, DataType::DT_BF16};

static const std::initializer_list<DataType> AICORE_DTYPE_SUPPORT_LIST_REG = {
    DataType::DT_FLOAT, DataType::DT_FLOAT16, DataType::DT_INT32, DataType::DT_BF16};

// 根据芯片类型、dtype判断算子是否支持走aicore
static inline bool IsAiCoreSupport(const aclTensor* self) {
  // Cummin只需要判断dtype
  auto DTYPE_SUPPORT_LIST = IsRegBase() ? AICORE_DTYPE_SUPPORT_LIST_REG : AICORE_DTYPE_SUPPORT_LIST;
  return CheckType(self->GetDataType(), DTYPE_SUPPORT_LIST);
}

// AICORE算子kernel
static inline std::tuple<aclTensor*, aclTensor*> CumminAiCore(const aclTensor* self, int64_t dim, aclTensor* valuesOut,
                                                              aclTensor* indicesOut, aclOpExecutor* executor) {
  L0_DFX(CumminAiCore, self, dim, valuesOut, indicesOut);
  // 使用框架宏ADD_TO_LAUNCHER_LIST_AICORE，将AiCore Cummin算子加入任务队列
  // Cummin是算子的OpType，self是算子的输入，valuesOut, indicesOut是算子的输出，属性dim传入默认值dim
  auto ret = ADD_TO_LAUNCHER_LIST_AICORE(Cummin, OP_INPUT(self), OP_OUTPUT(valuesOut, indicesOut), OP_ATTR(dim));
  if (ret != ACL_SUCCESS) {
    OP_LOGE(ACLNN_ERR_INNER_NULLPTR, "CumminAiCore ADD_TO_LAUNCHER_LIST_AICORE failed.");
    return std::tuple<aclTensor*, aclTensor*>(nullptr, nullptr);
  }
  return {valuesOut, indicesOut};
}

// AICPU算子kernel
static inline std::tuple<aclTensor*, aclTensor*> CumminAiCpu(const aclTensor* self, int64_t dim, aclTensor* valuesOut,
                                                             aclTensor* indicesOut, aclOpExecutor* executor) {
  L0_DFX(CumminAiCpu, self, dim, valuesOut, indicesOut);

  static internal::AicpuTaskSpace space("Cummin");
  // 使用框架宏ADD_TO_LAUNCHER_LIST_AICPU，将AiCpu Cummin算子加入任务队列
  // Cummin是算子的OpType，self是算子的输入，valuesOut、indicesOut是算子的输出，属性dim传入默认值dim
  auto ret = ADD_TO_LAUNCHER_LIST_AICPU(Cummin, OP_ATTR_NAMES({"axis"}), OP_INPUT(self),
                                        OP_OUTPUT(valuesOut, indicesOut), OP_ATTR(dim));
  if (ret != ACL_SUCCESS) {
    OP_LOGE(ACLNN_ERR_INNER_NULLPTR, "CumminAiCpu ADD_TO_LAUNCHER_LIST_AICPU failed.");
    return std::tuple<aclTensor*, aclTensor*>(nullptr, nullptr);
  }
  return {valuesOut, indicesOut};
}

std::tuple<aclTensor*, aclTensor*> CumminOutInt32(const aclTensor* self, int64_t dim, aclOpExecutor* executor) {
  // 根据输入shape申请输出tensor
  auto valuesOut = executor->AllocTensor(self->GetViewShape(), self->GetDataType(), self->GetViewFormat());
  auto indicesOut = executor->AllocTensor(self->GetViewShape(), DataType::DT_INT32, self->GetViewFormat());
  if (valuesOut == nullptr || indicesOut == nullptr) {
    OP_LOGE(ACLNN_ERR_INNER_NULLPTR, "alloc out tensor failed.");
    return {nullptr, nullptr};
  }

  if (IsAiCoreSupport(self)) {
    return CumminAiCore(self, dim, valuesOut, indicesOut, executor);
  } else {
    return CumminAiCpu(self, dim, valuesOut, indicesOut, executor);
  }
}

std::tuple<aclTensor*, aclTensor*> CumminOutInt64(const aclTensor* self, int64_t dim, aclOpExecutor* executor) {
  // 根据输入shape申请输出tensor
  auto valuesOut = executor->AllocTensor(self->GetViewShape(), self->GetDataType(), self->GetViewFormat());
  auto indicesOut = executor->AllocTensor(self->GetViewShape(), DataType::DT_INT64, self->GetViewFormat());
  if (valuesOut == nullptr || indicesOut == nullptr) {
    OP_LOGE(ACLNN_ERR_INNER_NULLPTR, "alloc out tensor failed.");
    return {nullptr, nullptr};
  }

  if (IsAiCoreSupport(self) && IsRegBase()) {
    return CumminAiCore(self, dim, valuesOut, indicesOut, executor);
  } else {
    return CumminAiCpu(self, dim, valuesOut, indicesOut, executor);
  }
}
}  // namespace l0op