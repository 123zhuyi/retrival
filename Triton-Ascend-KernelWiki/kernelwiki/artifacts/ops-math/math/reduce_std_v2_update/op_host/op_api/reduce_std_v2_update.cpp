/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

/*!
 * \file reduce_std_v2_update.cpp
 * \brief
 */

#include "reduce_std_v2_update.h"
#include "opdev/aicpu/aicpu_task.h"
#include "opdev/data_type_utils.h"
#include "opdev/format_utils.h"
#include "opdev/make_op_executor.h"
#include "opdev/op_def.h"
#include "opdev/op_dfx.h"
#include "opdev/op_executor.h"
#include "opdev/op_log.h"
#include "opdev/shape_utils.h"

using namespace op;

namespace l0op {
OP_TYPE_REGISTER(ReduceStdV2Update);

const aclTensor* ReduceStdV2Update(const aclTensor* self, const aclTensor* mean, const aclIntArray* dim,
                                   bool unbiased, bool keepdim, aclOpExecutor* executor) {
  L0_DFX(ReduceStdV2Update, self, mean, dim, unbiased, keepdim);
  auto stdWithV2UpdateOut = executor->AllocTensor(self->GetDataType(),
                                                  self->GetStorageFormat(), self->GetOriginalFormat());
  CHECK_RET(stdWithV2UpdateOut != nullptr, nullptr);
  bool if_std = false;
  INFER_SHAPE(ReduceStdV2Update, OP_INPUT(self, mean), OP_OUTPUT(stdWithV2UpdateOut),
              OP_ATTR(dim, if_std, unbiased, keepdim));
  auto ret = ADD_TO_LAUNCHER_LIST_AICORE(ReduceStdV2Update, OP_INPUT(self, mean), OP_OUTPUT(stdWithV2UpdateOut),
                                         OP_ATTR(dim, if_std, unbiased, keepdim));
  OP_CHECK(ret ==  ACLNN_SUCCESS, OP_LOGE(ACLNN_ERR_INNER_NULLPTR, "ReduceStdV2UpdateAiCore ADD_TO_LAUNCHER_LIST_AICORE failed."),
    return nullptr);
  return stdWithV2UpdateOut;
}
const aclTensor* ReduceStdV2UpdateCorrection(const aclTensor* self, const aclTensor* mean, const aclIntArray* dim,
                                             int64_t correction, bool keepdim, aclOpExecutor* executor) {
  L0_DFX(ReduceStdV2UpdateCorrection, self, mean, dim, correction, keepdim);
  auto stdWithV2UpdateOut = executor->AllocTensor(self->GetDataType(),
                                                  self->GetStorageFormat(), self->GetOriginalFormat());
  CHECK_RET(stdWithV2UpdateOut != nullptr, nullptr);
  bool unbiased = true;
  if (correction == 0) {
    unbiased = false;
  }
  bool if_std = false;
  INFER_SHAPE(ReduceStdV2Update, OP_INPUT(self, mean), OP_OUTPUT(stdWithV2UpdateOut),
              OP_ATTR(dim, if_std, unbiased, keepdim, correction));
  auto ret = ADD_TO_LAUNCHER_LIST_AICORE(ReduceStdV2Update, OP_INPUT(self, mean), OP_OUTPUT(stdWithV2UpdateOut),
                                         OP_ATTR(dim, if_std, unbiased, keepdim, correction));
  OP_CHECK(ret ==  ACLNN_SUCCESS, OP_LOGE(ACLNN_ERR_INNER_NULLPTR, "ReduceStdV2UpdateAiCcore ADD_TO_LAUNCHER_LIST_AICORE failed."),
    return nullptr);
  return stdWithV2UpdateOut;
}
}  // namespace l0op