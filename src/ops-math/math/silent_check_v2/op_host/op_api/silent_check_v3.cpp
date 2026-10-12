/**
 * This program is free software, you can redistribute it and/or modify.
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This file is a part of the CANN Open Software.
 * Licensed under CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include "silent_check_v3.h"
#include "opdev/make_op_executor.h"
#include "opdev/op_dfx.h"
#include "opdev/aicpu/aicpu_task.h"

using namespace op;
namespace l0op {
OP_TYPE_REGISTER(SilentCheckV3);

const aclTensor* SilentCheckV3(const aclTensor *val, const aclTensor *max, const aclTensor *avgRef, const aclTensor *inputGradRef, const aclTensor *stepRef, 
                                const aclTensor *dstSize, const aclTensor *dstStride, const aclTensor *dstOffset, const float cThreshL1, 
                                const float cThreshL2, const float beta1, const int32_t npuAsdDetect, aclOpExecutor *executor) {
  auto result = executor->AllocTensor({op::Shape({1})}, DataType::DT_INT32);
  L0_DFX(SilentCheckV3, val, max, avgRef, inputGradRef, stepRef, dstSize, dstStride, dstOffset, cThreshL1, cThreshL2, beta1, npuAsdDetect, result);
  static internal::AicpuTaskSpace space("SilentCheckV3");

  auto ret = ADD_TO_LAUNCHER_LIST_AICPU(SilentCheckV3,
                                        OP_ATTR_NAMES({"c_thresh_l1", "c_thresh_l2", "beta1", "npu_asd_detect"}),
                                        OP_INPUT(val, max, avgRef, inputGradRef, stepRef, dstSize, dstStride, dstOffset),
                                        OP_OUTPUT(avgRef, inputGradRef, stepRef, result),
                                        OP_ATTR(cThreshL1, cThreshL2, beta1, npuAsdDetect));
  CHECK_RET(ret == ACLNN_SUCCESS, nullptr);
  return result;
}
}  // namespace l0op
