/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include "masked_fill.h"
#include "opdev/aicpu/aicpu_task.h"
#include "opdev/op_log.h"
#include "opdev/op_executor.h"
#include "opdev/make_op_executor.h"
#include "opdev/shape_utils.h"
#include "opdev/shape_utils.h"
#include "opdev/op_def.h"
#include "opdev/op_dfx.h"
#include "op_api/aclnn_check.h"

using namespace op;

namespace l0op {

OP_TYPE_REGISTER(MaskedFill);

static const std::initializer_list<op::DataType> AICORE_DTYPE_SUPPORT_LIST = {
    op::DataType::DT_FLOAT, op::DataType::DT_FLOAT16, op::DataType::DT_BOOL,
    op::DataType::DT_INT8,  op::DataType::DT_INT32, op::DataType::DT_BF16};

static const std::initializer_list<op::DataType> ASCEND910D_AICORE_DTYPE_SUPPORT_LIST = {
    op::DataType::DT_FLOAT, op::DataType::DT_FLOAT16, op::DataType::DT_BOOL, op::DataType::DT_INT64,
    op::DataType::DT_INT8,  op::DataType::DT_INT32, op::DataType::DT_BF16};

// 根据芯片类型、dtype判断算子是否支持走aicore
static bool IsAiCoreSupport(const aclTensor *self) {
  // 只需要判断dtype
  if (IsRegBase()) {
    return CheckType(self->GetDataType(), ASCEND910D_AICORE_DTYPE_SUPPORT_LIST);
  } else {
    return CheckType(self->GetDataType(), AICORE_DTYPE_SUPPORT_LIST);
  }
}

// AICORE算子kernel
static const aclTensor *MaskedFillAiCore(const aclTensor *self, const aclTensor *mask,
                                         const aclTensor *value, aclTensor *output,
                                         aclOpExecutor *executor) {
  L0_DFX(MaskedFillAiCore, self, mask, value, output);

  auto ret = ADD_TO_LAUNCHER_LIST_AICORE(MaskedFill,
                                         OP_INPUT(self, mask, value),
                                         OP_OUTPUT(output));
  OP_CHECK(ret == ACLNN_SUCCESS, OP_LOGE(ACLNN_ERR_INNER_NULLPTR, "MaskedFillAiCore ADD_TO_LAUNCHER_LIST_AICORE failed."), return nullptr);
  return output;
}

// AICPU算子kernel
static const aclTensor *MaskedFillAiCpu(const aclTensor *self, const aclTensor *mask,
                                        const aclTensor *value, aclTensor *output,
                                        aclOpExecutor *executor) {
  L0_DFX(MaskedFillAiCpu, self, mask, value, output);

  static internal::AicpuTaskSpace space("MaskedFill");
  auto ret = ADD_TO_LAUNCHER_LIST_AICPU(MaskedFill, OP_ATTR_NAMES(),
                             OP_INPUT(self, mask, value),
                             OP_OUTPUT(output));
  OP_CHECK(ret == ACLNN_SUCCESS, OP_LOGE(ACLNN_ERR_INNER_NULLPTR, "MaskedFillAiCpu ADD_TO_LAUNCHER_LIST_AICPU failed."), return nullptr);
  return output;
}

const aclTensor *MaskedFill(const aclTensor *self, const aclTensor *mask,
                            const aclTensor *value, aclOpExecutor *executor) {
  L0_DFX(MaskedFill, self, mask, value);
  auto output = executor->AllocTensor(self->GetViewShape(), self->GetDataType());

  if (IsAiCoreSupport(self)) {
    return MaskedFillAiCore(self, mask, value, output, executor);
  } else {
    return MaskedFillAiCpu(self, mask, value, output, executor);
  }
}

}  // namespace l0op
