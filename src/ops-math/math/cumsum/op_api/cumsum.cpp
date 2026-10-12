/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include "cumsum.h"
#include "opdev/aicpu/aicpu_task.h"
#include "opdev/data_type_utils.h"
#include "opdev/format_utils.h"
#include "opdev/make_op_executor.h"
#include "opdev/op_def.h"
#include "opdev/shape_utils.h"
#include "opdev/op_dfx.h"
#include "opdev/op_executor.h"
#include "opdev/op_log.h"
#include "opdev/platform.h"
#include "aclnn_kernels/common/op_error_check.h"
#include "op_api/aclnn_check.h"

using namespace op;

namespace l0op {

OP_TYPE_REGISTER(Cumsum);

static const std::initializer_list<DataType> AICORE910_DTYPE_SUPPORT_LIST = {
  DataType::DT_FLOAT, DataType::DT_FLOAT16, DataType::DT_INT32};

static const std::initializer_list<DataType> AICORE910B_DTYPE_SUPPORT_LIST = {
  DataType::DT_FLOAT, DataType::DT_FLOAT16, op::DataType::DT_BF16, DataType::DT_INT32};

static const std::initializer_list<DataType> REGBASE_DTYPE_SUPPORT_LIST = {
  DataType::DT_FLOAT, DataType::DT_FLOAT16,op::DataType::DT_BF16, DataType::DT_INT32,
  DataType::DT_UINT8, DataType::DT_INT8, DataType::DT_INT64};

// 根据芯片类型、dtype判断算子是否支持走aicore
static inline bool IsAiCoreSupport(const aclTensor *self) {
  auto npuArch = op::GetCurrentPlatformInfo().GetCurNpuArch();
  if (npuArch == NpuArch::DAV_2201) {
    return CheckType(self->GetDataType(), AICORE910B_DTYPE_SUPPORT_LIST);
  }
  if (IsRegBase(npuArch)) {
    return CheckType(self->GetDataType(), REGBASE_DTYPE_SUPPORT_LIST);
  }
  // 1980
  return CheckType(self->GetDataType(), AICORE910_DTYPE_SUPPORT_LIST);
}

// AICORE算子kernel
static inline const aclTensor *CumsumAiCore(const aclTensor *self, const aclTensor *dim, aclTensor *out,
                                            aclOpExecutor *executor, bool exclusive = false, bool reverse = false) {
  L0_DFX(CumsumAiCore, self, dim, out);
  // 使用框架宏ADD_TO_LAUNCHER_LIST_AICORE，将AiCore Cumsum算子加入任务队列
  auto ret = ADD_TO_LAUNCHER_LIST_AICORE(Cumsum, OP_INPUT(self, dim), OP_OUTPUT(out), OP_ATTR(exclusive, reverse));
  OP_CHECK(ret == ACL_SUCCESS, OP_LOGE(ACLNN_ERR_INNER_NULLPTR, "CumsumAiCore ADD_TO_LAUNCHER_LIST_AICORE failed."), return nullptr);
  return out;
}

// AICPU算子kernel
static inline const aclTensor *CumsumAiCpu(const aclTensor *self, const aclTensor *dim, aclTensor *out,
                                           aclOpExecutor *executor, bool exclusive = false, bool reverse = false) {
  L0_DFX(CumsumAiCpu, self, dim, out);

  static internal::AicpuTaskSpace space("Cumsum");
  // 使用框架宏ADD_TO_LAUNCHER_LIST_AICPU，将AiCpu Cumsum算子加入任务队列
  // Cumsum是算子的OpType，self、dim是算子的输入，out是算子的输出
  auto ret = ADD_TO_LAUNCHER_LIST_AICPU(Cumsum, OP_ATTR_NAMES({"exclusive", "reverse"}), OP_INPUT(self, dim),
                                        OP_OUTPUT(out), OP_ATTR(exclusive, reverse));
  OP_CHECK(ret ==  ACL_SUCCESS, OP_LOGE(ACLNN_ERR_INNER_NULLPTR, "CumsumAiCpu ADD_TO_LAUNCHER_LIST_AICPU failed."), return nullptr);
  return out;
}

const aclTensor *Cumsum(const aclTensor *self, const aclTensor *dim, aclOpExecutor *executor) {
  // 根据输入shape申请输出tensor
  auto out = executor->AllocTensor(self->GetViewShape(), self->GetDataType(), Format::FORMAT_ND);
  if (out == nullptr) {
    OP_LOGE(ACLNN_ERR_INNER_NULLPTR, "alloc out tensor failed.");
    return nullptr;
  }

  if (IsAiCoreSupport(self)) {
    return CumsumAiCore(self, dim, out, executor);
  } else {
    return CumsumAiCpu(self, dim, out, executor);
  }
}

const aclTensor *Cumsum(const aclTensor *self, const aclTensor *dim, bool exclusive, bool reverse, aclOpExecutor *executor) {
  // 根据输入shape申请输出tensor
  auto out = executor->AllocTensor(self->GetViewShape(), self->GetDataType(), Format::FORMAT_ND);
  if (out == nullptr) {
    OP_LOGE(ACLNN_ERR_INNER_NULLPTR, "alloc out tensor failed.");
    return nullptr;
  }

  if (IsAiCoreSupport(self)) {
    return CumsumAiCore(self, dim, out, executor, exclusive, reverse);
  } else {
    return CumsumAiCpu(self, dim, out, executor, exclusive, reverse);
  }
}

}  // namespace l0op