/**
 * Copyright (c) 2025-2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include "linspace.h"
#include "opdev/aicpu/aicpu_task.h"
#include "opdev/make_op_executor.h"
#include "opdev/op_def.h"
#include "opdev/op_dfx.h"
#include "opdev/op_executor.h"
#include "opdev/op_log.h"
#include "opdev/platform.h"
#include "opdev/shape_utils.h"
#include "op_api/aclnn_check.h"

using namespace op;

namespace l0op {

OP_TYPE_REGISTER(LinSpace);

static const std::initializer_list<op::DataType> ASCEND910_AICORE_DTYPE_SUPPORT_LIST = {
    op::DataType::DT_INT32, op::DataType::DT_FLOAT16, op::DataType::DT_FLOAT,
    op::DataType::DT_INT8, op::DataType::DT_UINT8};

static const std::initializer_list<op::DataType> ASCEND910B_AICORE_DTYPE_SUPPORT_LIST = {
    op::DataType::DT_INT32, op::DataType::DT_FLOAT16, op::DataType::DT_FLOAT,
    op::DataType::DT_INT16, op::DataType::DT_INT8, op::DataType::DT_UINT8,
    op::DataType::DT_BF16};

// 根据芯片类型、dtype判断算子是否支持走AiCore
static bool IsAiCoreSupport(const aclTensor *start) {
  // 获取芯片类型
  auto npuArch = op::GetCurrentPlatformInfo().GetCurNpuArch();
  if (npuArch == NpuArch::DAV_2201 || IsRegBase(npuArch)) {
    return CheckType(start->GetDataType(), ASCEND910B_AICORE_DTYPE_SUPPORT_LIST);
  }
  return CheckType(start->GetDataType(), ASCEND910_AICORE_DTYPE_SUPPORT_LIST);
}

// AICPU算子kernel
static const aclTensor* LinspaceAiCpu(const aclTensor* start, const aclTensor* end, const aclTensor* steps,
                                      aclTensor* out, aclOpExecutor* executor) {
  L0_DFX(LinspaceAiCpu, start, end, steps, out);
  // 使用框架宏ADD_TO_LAUNCHER_LIST_AICPU，将AiCpu Linspace算子加入任务队列
  static internal::AicpuTaskSpace space("LinSpace");
  auto ret = ADD_TO_LAUNCHER_LIST_AICPU(LinSpace, OP_ATTR_NAMES({"Tidx"}), OP_INPUT(start, end, steps), OP_OUTPUT(out),
                                        OP_ATTR(start->GetDataType()));
  OP_CHECK(ret == ACLNN_SUCCESS, OP_LOGE(ACLNN_ERR_INNER_NULLPTR, "LinspaceAiCpu ADD_TO_LAUNCHER_LIST_AICPU failed."), return nullptr);
  return out;
}

// AICORE算子kernel
static const aclTensor* LinspaceAiCore(const aclTensor* start, const aclTensor* end, const aclTensor* steps,
                                       aclTensor* out, aclOpExecutor* executor) {
  L0_DFX(LinspaceAiCore, start, end, steps, out);
  // 使用框架宏ADD_TO_LAUNCHER_LIST_AICORE，将AiCore Linspace算子加入任务队列
  auto ret = ADD_TO_LAUNCHER_LIST_AICORE(LinSpace, OP_INPUT(start, end, steps), OP_OUTPUT(out));
  OP_CHECK(ret == ACLNN_SUCCESS, OP_LOGE(ACLNN_ERR_INNER_NULLPTR, "LinspaceAiCore ADD_TO_LAUNCHER_LIST_AICORE failed."), return nullptr);
  return out;
}

const aclTensor* Linspace(const aclTensor* start, const aclTensor* end, int64_t steps,
                          aclOpExecutor* executor) {
  auto out = executor->AllocTensor({op::Shape({steps})}, start->GetDataType());
  auto stepsTensor = executor->ConvertToTensor(executor->AllocScalar(steps), op::DataType::DT_INT32);
  if (IsAiCoreSupport(start)) {
    return LinspaceAiCore(start, end, stepsTensor, out, executor);
  } else {
    return LinspaceAiCpu(start, end, stepsTensor, out, executor);
  }
}
}  // namespace l0op
