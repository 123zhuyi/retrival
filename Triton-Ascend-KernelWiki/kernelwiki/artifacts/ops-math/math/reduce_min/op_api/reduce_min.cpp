/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */
#include "reduce_min.h"
#include "opdev/aicpu/aicpu_task.h"
#include "opdev/make_op_executor.h"
#include "opdev/op_def.h"
#include "opdev/op_dfx.h"
#include "opdev/op_executor.h"
#include "opdev/op_log.h"
#include "opdev/shape_utils.h"
#include "opdev/platform.h"
#include "aclnn_kernels/common/op_error_check.h"
#include "op_api/aclnn_check.h"

using namespace op;

namespace l0op {
OP_TYPE_REGISTER(ReduceMin);

static const std::initializer_list<op::DataType> AICORE_DTYPE_SUPPORT_LIST_910 = {
    op::DataType::DT_FLOAT16, op::DataType::DT_FLOAT, op::DataType::DT_UINT8, op::DataType::DT_INT32};

static const std::initializer_list<op::DataType> AICORE_DTYPE_SUPPORT_LIST_GE910B = {
    op::DataType::DT_FLOAT16, op::DataType::DT_FLOAT, op::DataType::DT_UINT8, op::DataType::DT_INT32,
    op::DataType::DT_INT64, op::DataType::DT_INT8, op::DataType::DT_BF16};

// 判断芯片类型是否大于等于910B
static inline bool CheckSocVersionGe910B(void) {
  if ((GetCurrentPlatformInfo().GetSocVersion() >= SocVersion::ASCEND910B &&
       GetCurrentPlatformInfo().GetSocVersion() <= SocVersion::ASCEND910E) ||
       IsRegBase()) {
    return true;
  }
  return false;
}

static bool IsAiCoreSupport(const aclTensor *self) {
    // 获取芯片类型
  if (CheckSocVersionGe910B()){
      return CheckType(self->GetDataType(), AICORE_DTYPE_SUPPORT_LIST_GE910B);
  }
  // 910
  return CheckType(self->GetDataType(), AICORE_DTYPE_SUPPORT_LIST_910);
}

// AICORE算子kernel
static const aclTensor *ReduceMinAiCore(const aclTensor *self, const aclTensor *dimList, bool keepDim,
                                        bool noopWithEmptyAxes, const aclTensor *minOut,
                                        aclOpExecutor *executor) {
  L0_DFX(ReduceMinAiCore, self, dimList, keepDim, noopWithEmptyAxes, minOut);

  auto retAicore = ADD_TO_LAUNCHER_LIST_AICORE(ReduceMin, OP_INPUT(self, dimList),
                              OP_OUTPUT(minOut), OP_ATTR(keepDim, noopWithEmptyAxes));
  OP_CHECK_ADD_TO_LAUNCHER_LIST_AICORE(retAicore != ACLNN_SUCCESS, return nullptr,
                                       "ReduceMin ADD_TO_LAUNCHER_LIST_AICORE failed.");

  return minOut;
}

// AICPU算子kernel
static const aclTensor *ReduceMinAiCpu(const aclTensor *self, const aclTensor *dimList, bool keepDim,
                                       const aclTensor *minOut, aclOpExecutor *executor) {
  L0_DFX(ReduceMinAiCpu, self, dimList, keepDim, minOut);

  static internal::AicpuTaskSpace space("Min", ge::DEPEND_IN_SHAPE, true);
  auto ret = ADD_TO_LAUNCHER_LIST_AICPU(ReduceMin, OP_ATTR_NAMES({"keep_dims", "Tidx"}), OP_INPUT(self, dimList),
                             OP_OUTPUT(minOut), OP_ATTR(keepDim, dimList->GetDataType()));
  CHECK_RET(ret == ACLNN_SUCCESS, nullptr);
  return minOut;
}

const aclTensor *ReduceMin(const aclTensor *self, const aclIntArray *dim, bool keepDim, aclOpExecutor *executor) {
  auto dimList = executor->ConvertToTensor(dim, op::DataType::DT_INT64);
  auto minOut = executor->AllocTensor(self->GetViewShape(), self->GetDataType());

  // dim为空时，默认保留所有轴
  bool noopWithEmptyAxes = true;
  auto ret = INFER_SHAPE(ReduceMin, OP_INPUT(self, dimList), OP_OUTPUT(minOut), OP_ATTR(keepDim, noopWithEmptyAxes));
  if (ret != ACLNN_SUCCESS) {
    OP_LOGE(ACLNN_ERR_PARAM_INVALID, "ReduceMin infer shape faild.");
    return nullptr;
  }

  if (IsAiCoreSupport(self)) {
    return ReduceMinAiCore(self, dimList, keepDim, noopWithEmptyAxes, minOut, executor);
  } else {
    return ReduceMinAiCpu(self, dimList, keepDim, minOut, executor);
  }
}
} // namespace l0op
