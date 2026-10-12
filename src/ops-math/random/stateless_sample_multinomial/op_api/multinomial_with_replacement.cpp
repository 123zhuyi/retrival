/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include "multinomial_with_replacement.h"
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
OP_TYPE_REGISTER(MultinomialWithReplacement);
OP_TYPE_REGISTER(MultinomialWithReplacementTensor);

static const aclTensor *MultinomialWithReplacementAiCpu(const aclTensor *self,
                                                        const aclTensor *seed,
                                                        const aclTensor *offset,
                                                        int64_t numsamples,
                                                        bool replacement,
                                                        aclTensor *out,
                                                        aclOpExecutor *executor) {
  L0_DFX(MultinomialWithReplacementAiCpu, self, seed, offset, numsamples, replacement, out);

  static internal::AicpuTaskSpace space("MultinomialWithReplacement");
  auto ret = ADD_TO_LAUNCHER_LIST_AICPU(MultinomialWithReplacement,
                                        OP_ATTR_NAMES({"numsamples", "replacement"}),
                                        OP_INPUT(self, seed, offset),
                                        OP_OUTPUT(out),
                                        OP_ATTR(numsamples, replacement));
  CHECK_RET(ret == ACLNN_SUCCESS, nullptr);
  return out;
}

const aclTensor *MultinomialWithReplacement(const aclTensor *self,
                                            int64_t numsamples,
                                            bool replacement,
                                            int64_t seed,
                                            int64_t offset,
                                            aclOpExecutor *executor) {
  int64_t size = 1;
  auto seedTensor = executor->ConvertToTensor(&seed, size, op::DataType::DT_INT64);
  auto offsetTensor = executor->ConvertToTensor(&offset, size, op::DataType::DT_INT64);
  auto shape = self->GetViewShape();
  auto dimNum = shape.GetDimNum();
  shape.SetDim(dimNum - 1, numsamples);
  auto out = executor->AllocTensor(shape, op::DataType::DT_INT64, self->GetViewFormat());
  return MultinomialWithReplacementAiCpu(self, seedTensor, offsetTensor, numsamples,
                                         replacement, out, executor);
}

const aclTensor *MultinomialWithReplacementTensor(const aclTensor *self,
                                            int64_t numsamples,
                                            bool replacement,
                                            const aclTensor *seedTensor,
                                            const aclTensor *offsetTensor,
                                            aclOpExecutor *executor) {
  auto shape = self->GetViewShape();
  auto dimNum = shape.GetDimNum();
  shape.SetDim(dimNum - 1, numsamples);
  auto out = executor->AllocTensor(shape, op::DataType::DT_INT64, self->GetViewFormat());
  return MultinomialWithReplacementAiCpu(self, seedTensor, offsetTensor, numsamples,
                                         replacement, out, executor);
}
}
