/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include "searchsorted.h"
#include "opdev/aicpu/aicpu_task.h"
#include "opdev/make_op_executor.h"
#include "opdev/op_dfx.h"
#include "opdev/op_executor.h"
#include "opdev/op_log.h"

using namespace op;
namespace l0op {
OP_TYPE_REGISTER(SearchSorted);
const aclTensor *SearchSorted(const aclTensor *sortedSequence,
                              const aclTensor *self,
                              op::DataType dtype,
                              const bool right,
                              const aclTensor *sorter,
                              aclOpExecutor *executor) {
  L0_DFX(SearchSorted, sortedSequence, self, dtype, right, sorter);

  auto out = executor->AllocTensor(self->GetViewShape(), dtype);
  CHECK_RET(out != nullptr, nullptr);
  static internal::AicpuTaskSpace space("SearchSorted");
  if (sorter != nullptr) {
    auto ret = ADD_TO_LAUNCHER_LIST_AICPU(SearchSorted,
                                          OP_ATTR_NAMES({"dtype", "right"}),
                                          OP_INPUT(sortedSequence, self, sorter),
                                          OP_OUTPUT(out),
                                          OP_ATTR(dtype, right));
    CHECK_RET(ret == ACLNN_SUCCESS, nullptr);
  } else {
    auto ret = ADD_TO_LAUNCHER_LIST_AICPU(SearchSorted,
                                          OP_ATTR_NAMES({"dtype", "right"}),
                                          OP_INPUT(sortedSequence, self),
                                          OP_OUTPUT(out),
                                          OP_ATTR(dtype, right));
    CHECK_RET(ret == ACLNN_SUCCESS, nullptr);
  }
  return out;
}
}
