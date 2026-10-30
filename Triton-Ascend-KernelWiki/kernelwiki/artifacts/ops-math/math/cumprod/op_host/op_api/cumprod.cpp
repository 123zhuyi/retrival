/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include "cumprod.h"
#include "opdev/make_op_executor.h"
#include "opdev/op_dfx.h"
#include "opdev/aicpu/aicpu_task.h"

using namespace op;
namespace l0op {
OP_TYPE_REGISTER(Cumprod);

const aclTensor* Cumprod(const aclTensor *x, const aclScalar *axis, bool exclusive, bool reverse,
                         aclOpExecutor *executor) {
  auto out = executor->AllocTensor(x->GetViewShape(), x->GetDataType(), Format::FORMAT_ND);
  if (out == nullptr) {
    OP_LOGE(ACLNN_ERR_INNER_NULLPTR, "alloc out tensor allocation failed.");
    return nullptr;
  }
  L0_DFX(Cumprod, x, axis, exclusive, reverse, out);
  static internal::AicpuTaskSpace space("Cumprod", ge::DEPEND_IN_SHAPE, true);
  auto ret = ADD_TO_LAUNCHER_LIST_AICPU(Cumprod, 
                                        OP_ATTR_NAMES({"exclusive", "reverse", "Tidx"}), 
                                        OP_INPUT(x, axis),
                                        OP_OUTPUT(out), 
                                        OP_ATTR(exclusive, reverse, ge::DataType::DT_INT32));
  OP_LOGI("cumprod ret:%d, out:%p\n", ret, out);
  CHECK_RET(ret == ACLNN_SUCCESS, nullptr);
  return out;
}
}  // namespace l0op
