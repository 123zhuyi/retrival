/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include "stateless_randperm.h"
#include "opdev/aicpu/aicpu_task.h"
#include "opdev/make_op_executor.h"
#include "opdev/op_def.h"
#include "opdev/op_dfx.h"
#include "opdev/op_executor.h"
#include "opdev/op_log.h"
#include "opdev/shape_utils.h"
#include "opdev/platform.h"

using namespace op;

static const std::initializer_list<op::DataType> DTYPE_SUPPORT_LIST_AICORE_3510 = {
    op::DataType::DT_FLOAT,     op::DataType::DT_FLOAT16,  op::DataType::DT_INT32,       op::DataType::DT_BF16,
    op::DataType::DT_INT64,      op::DataType::DT_INT8,     op::DataType::DT_UINT8,      op::DataType::DT_INT16};

namespace l0op {

OP_TYPE_REGISTER(StatelessRandperm);
 
// AICPU算子kernel
static const aclTensor *StatelessRandpermAiCpu(const aclTensor *n, const aclTensor *seed,
                                 const aclTensor *offset, int64_t layout, op::DataType dstDtype, aclTensor *out,
                                 aclOpExecutor *executor) {
  L0_DFX(StatelessRandpermAiCpu, n, seed, offset, layout, dstDtype, out);
  static internal::AicpuTaskSpace space("StatelessRandperm");
  auto ret = ADD_TO_LAUNCHER_LIST_AICPU(StatelessRandperm, OP_ATTR_NAMES({"layout", "dtype"}),
                                        OP_INPUT(n, seed, offset),
                                        OP_OUTPUT(out), OP_ATTR(layout, dstDtype));
  CHECK_RET(ret == ACLNN_SUCCESS, nullptr);
  return out;
}

// AICORE算子kernel
static const aclTensor *StatelessRandpermAiCore(const aclTensor *n, const aclTensor *seed,
                                 const aclTensor *offset, int64_t layout, op::DataType dstDtype, aclTensor *out,
                                 aclOpExecutor *executor) {
  L0_DFX(StatelessRandpermAiCore, n, seed, offset, layout, dstDtype, out);
  auto ret = ADD_TO_LAUNCHER_LIST_AICORE(StatelessRandperm, OP_ATTR_NAMES({"layout", "dtype"}),
                                        OP_INPUT(n, seed, offset),
                                        OP_OUTPUT(out), OP_ATTR(layout, dstDtype));
  CHECK_RET(ret == ACLNN_SUCCESS, nullptr);
  return out;
}

static inline bool CheckDtypeSupportAiCore(op::DataType dtype) {
  auto it = std::find(DTYPE_SUPPORT_LIST_AICORE_3510.begin(), DTYPE_SUPPORT_LIST_AICORE_3510.end(), dtype);
  return (it != DTYPE_SUPPORT_LIST_AICORE_3510.end());
}

static inline bool CheckAiCoreIsSupport(op::DataType dtype) {
  auto npuArch = op::GetCurrentPlatformInfo().GetCurNpuArch();
  return (npuArch == NpuArch::DAV_3510 && CheckDtypeSupportAiCore(dtype));
}
 
const aclTensor *StatelessRandperm(op::Shape shape, const aclTensor *n, const aclTensor *seed,
                                 const aclTensor *offset, int64_t layout, op::DataType dtype,
                                 aclOpExecutor *executor) {
  auto out = executor->AllocTensor(shape, dtype);
  if (CheckAiCoreIsSupport(dtype)) {
    return StatelessRandpermAiCore(n, seed, offset, layout, dtype, out, executor);
  } else {
    return StatelessRandpermAiCpu(n, seed, offset, layout, dtype, out, executor);
  }
}

}  // namespace l0op

