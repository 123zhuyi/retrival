/**
 * This program is free software, you can redistribute it and/or modify.
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This file is a part of the CANN Open Software.
 * Licensed under CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include "silent_check_v2.h"
#include "opdev/make_op_executor.h"
#include "opdev/op_dfx.h"
#include "opdev/aicpu/aicpu_task.h"

using namespace op;
namespace l0op {
OP_TYPE_REGISTER(SilentCheckV2);

const aclTensor* SilentCheckV2(const aclTensor *val, const aclTensor *inputGradRef, const aclTensor *sfdaRef, const aclTensor *stepRef, 
                                const int32_t cMinSteps, const float cThreshL1, const float cCoeffL1, const float cThreshL2, const float cCoeffL2,
                                const int32_t npuAsdDetect, aclOpExecutor *executor) {
  auto result = executor->AllocTensor({op::Shape({1})}, DataType::DT_INT32);
  L0_DFX(SilentCheckV2, val, inputGradRef, sfdaRef, stepRef, cMinSteps, cThreshL1, cCoeffL1, cThreshL2, cCoeffL2, npuAsdDetect, result);
  static internal::AicpuTaskSpace space("SilentCheckV2");

  auto ret = ADD_TO_LAUNCHER_LIST_AICPU(SilentCheckV2,
                                        OP_ATTR_NAMES({"c_min_steps", "c_thresh_l1", "c_coeff_l1", "c_thresh_l2",
                                                       "c_coeff_l2", "npu_asd_detect"}),
                                        OP_INPUT(val, inputGradRef, sfdaRef, stepRef),
                                        OP_OUTPUT(inputGradRef, sfdaRef, stepRef, result),
                                        OP_ATTR(cMinSteps, cThreshL1, cCoeffL1, cThreshL2, cCoeffL2, npuAsdDetect));
  CHECK_RET(ret == ACLNN_SUCCESS, nullptr);
  return result;
}
}  // namespace l0op
