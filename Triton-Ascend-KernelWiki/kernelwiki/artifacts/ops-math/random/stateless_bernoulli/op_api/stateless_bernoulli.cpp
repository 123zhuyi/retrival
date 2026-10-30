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
 * \file stateless_bernoulli.cpp
 * \brief
 */

#include "stateless_bernoulli.h"
#include "op_api/aclnn_check.h"
#include "opdev/aicpu/aicpu_task.h"
#include "opdev/make_op_executor.h"
#include "opdev/op_def.h"
#include "opdev/op_dfx.h"
#include "opdev/op_executor.h"
#include "opdev/op_log.h"
#include "opdev/shape_utils.h"

using namespace op;

namespace l0op {

OP_TYPE_REGISTER(StatelessBernoulli);

static const std::initializer_list<DataType> AICORE_DTYPE_SUPPORT_LIST = {
  DataType::DT_INT8, DataType::DT_UINT8, DataType::DT_INT16, DataType::DT_UINT16,
  DataType::DT_INT32, DataType::DT_UINT32, DataType::DT_INT64, DataType::DT_UINT64,
  DataType::DT_FLOAT, DataType::DT_FLOAT16, DataType::DT_BF16, DataType::DT_BOOL
};

static const std::initializer_list<DataType> AICORE_PROB_DTYPE_SUPPORT_LIST = {
  DataType::DT_FLOAT, DataType::DT_FLOAT16, DataType::DT_BF16
};

// 根据芯片型号，dtype 判断AICore 是否支持
static inline bool IsAiCoreSupport(DataType yDtype, DataType pDtype)
{
  if (IsRegBase()) {
    return CheckType(yDtype, AICORE_DTYPE_SUPPORT_LIST) && CheckType(pDtype, AICORE_PROB_DTYPE_SUPPORT_LIST);
  }
  return false;
}

// AICPU算子kernel
static const aclTensor *StatelessBernoulliAiCpu(const aclTensor *shapeTensor, const aclTensor *prob,
                                                const aclTensor *seed, const aclTensor *offset, aclTensor *out,
                                                aclOpExecutor *executor) {
  L0_DFX(StatelessBernoulliAiCpu, shapeTensor, prob, seed, offset, out);

  static internal::AicpuTaskSpace space("StatelessBernoulli");
  ADD_TO_LAUNCHER_LIST_AICPU(StatelessBernoulli,
                             OP_ATTR_NAMES({"dtype"}),
                             OP_INPUT(shapeTensor, prob, seed, offset),
                             OP_OUTPUT(out),
                             OP_ATTR(out->GetDataType()));
  return out;
}

// AICORE算子kernel
static const aclTensor *StatelessBernoulliAiCore(const aclTensor *shapeTensor, const aclTensor *prob,
                                                 const aclTensor *seed, const aclTensor *offset, const aclTensor *out,
                                                 aclOpExecutor *executor) {
  L0_DFX(StatelessBernoulliAiCore, shapeTensor, prob, seed, offset, out);
  
  // 使用框架宏 ADD_TO_LAUNCHER_LIST_AICORE
  ADD_TO_LAUNCHER_LIST_AICORE(StatelessBernoulli,
                              OP_ATTR_NAMES({"dtype"}),
                              OP_INPUT(shapeTensor, prob, seed, offset),
                              OP_OUTPUT(out),
                              OP_ATTR(out->GetDataType()));
  return out;
}

const aclTensor *StatelessBernoulli(const aclTensor *input, const aclTensor *prob, int64_t seed, int64_t offset,
                                    aclOpExecutor *executor) {
  auto inputShape = op::ToShapeVector(input->GetViewShape());
  auto sizeArr = executor->AllocIntArray(inputShape.data(), inputShape.size());
  OP_CHECK(sizeArr != nullptr, OP_LOGE(ACLNN_ERR_INNER_NULLPTR, "StatelessBernoulli inputShape is null."), return nullptr);
  const aclTensor *shapeTensor = nullptr;

  int64_t size = 1;
  for (size_t index = 0; index < sizeArr->Size(); index++) {
      size *= (*sizeArr)[index];
  }
  if (size > INT32_MAX) {
    shapeTensor = executor->ConvertToTensor(sizeArr, DataType::DT_INT64);
  } else {
    shapeTensor = executor->ConvertToTensor(sizeArr, DataType::DT_INT32);
  }
  
  auto seedTensor = executor->ConvertToTensor(executor->AllocScalar(seed), op::DataType::DT_INT64);
  auto offsetTensor = executor->ConvertToTensor(executor->AllocScalar(offset), op::DataType::DT_INT64);

  auto out = executor->AllocTensor(input->GetViewShape(), input->GetDataType(), input->GetViewFormat());
  if (IsAiCoreSupport(out->GetDataType(), prob->GetDataType())) {
    return StatelessBernoulliAiCore(shapeTensor, prob, seedTensor, offsetTensor, out, executor);
  } else {
    return StatelessBernoulliAiCpu(shapeTensor, prob, seedTensor, offsetTensor, out, executor);
  }
}

}  // namespace l0op
