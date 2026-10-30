/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include "precision_compare.h"
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
OP_TYPE_REGISTER(PrecisionCompare);
static const std::initializer_list<op::DataType> INPUT_DTYPE_SUPPORT_LIST = {
    op::DataType::DT_FLOAT, op::DataType::DT_FLOAT16, op::DataType::DT_BF16};
static const aclTensor *PrecisionCompareAiCpu(const aclTensor *benchMark, const aclTensor *realData, aclTensor *out,
                                              const uint32_t detect_type, aclOpExecutor *executor) {
  static internal::AicpuTaskSpace space("DataCompare");
  auto ret = ADD_TO_LAUNCHER_LIST_AICPU(PrecisionCompare, OP_ATTR_NAMES({"detect_type"}), OP_INPUT(benchMark, realData),
                                        OP_OUTPUT(out), OP_ATTR(detect_type));
  CHECK_RET(ret == ACLNN_SUCCESS, nullptr);
  return out;
}

const aclTensor *PrecisionCompare(const aclTensor *benchMark, const aclTensor *realData, aclTensor *out,
                                  const uint32_t detect_type, aclOpExecutor *executor) {
  L0_DFX(PrecisionCompare, benchMark, realData);
  if (!CheckType(benchMark->GetDataType(), INPUT_DTYPE_SUPPORT_LIST) ||
      !CheckType(realData->GetDataType(), INPUT_DTYPE_SUPPORT_LIST)) {
    OP_LOGE(ACLNN_ERR_PARAM_INVALID,
            "Level0 check input type failed, Unsupported input data type, benchmark type is [%s],"
            "real data type is [%s], Support list is %s",
            op::ToString(benchMark->GetDataType()).GetString(), op::ToString(realData->GetDataType()).GetString(),
            op::ToString(INPUT_DTYPE_SUPPORT_LIST).GetString());
    return nullptr;
  }
  return PrecisionCompareAiCpu(benchMark, realData, out, detect_type, executor);
}
}  // namespace l0op