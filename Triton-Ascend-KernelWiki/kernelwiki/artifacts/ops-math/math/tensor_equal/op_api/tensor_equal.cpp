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
 * \file tensor_equal.cpp
 * \brief
 */

#include "tensor_equal.h"
#include "opdev/aicpu/aicpu_task.h"
#include "opdev/make_op_executor.h"
#include "opdev/op_def.h"
#include "opdev/op_dfx.h"
#include "opdev/op_executor.h"
#include "opdev/op_log.h"
#include "opdev/shape_utils.h"
#include "op_api/aclnn_check.h"

using namespace op;

namespace l0op {

OP_TYPE_REGISTER(TensorEqual);

// 1980
static const std::initializer_list<op::DataType> AICORE910_DTYPE_SUPPORT_LIST = {
    op::DataType::DT_FLOAT, op::DataType::DT_INT32, op::DataType::DT_FLOAT16, op::DataType::DT_INT8,
    op::DataType::DT_UINT8, op::DataType::DT_BOOL,  op::DataType::DT_BF16};

static const std::initializer_list<op::DataType> ARCH3510_DTYPE_SUPPORT_LIST = {
    op::DataType::DT_FLOAT, op::DataType::DT_FLOAT16, op::DataType::DT_BF16,  op::DataType::DT_BOOL,
    op::DataType::DT_INT8,  op::DataType::DT_UINT8,   op::DataType::DT_INT16, op::DataType::DT_UINT16,
    op::DataType::DT_INT32, op::DataType::DT_UINT32,  op::DataType::DT_INT64, op::DataType::DT_UINT64};

// 根据dtype判断算子是否支持走aicore
static bool IsAiCoreSupport(const aclTensor* self)
{
  if (IsRegBase()) {
    return CheckType(self->GetDataType(), ARCH3510_DTYPE_SUPPORT_LIST);
  } else {
    return CheckType(self->GetDataType(), AICORE910_DTYPE_SUPPORT_LIST);
  }
}

// AICORE算子kernel
static const aclTensor *TensorEqualAiCore(const aclTensor *self, const aclTensor *other,
                                   aclTensor *eqOut, aclOpExecutor *executor) {
  L0_DFX(TensorEqualAiCore, self, other, eqOut);

  ADD_TO_LAUNCHER_LIST_AICORE(TensorEqual,
                              OP_INPUT(self, other),
                              OP_OUTPUT(eqOut));
  return eqOut;
}

// AICPU算子kernel
static const aclTensor *TensorEqualAiCpu(const aclTensor *self, const aclTensor *other,
                                  aclTensor *eqOut, aclOpExecutor *executor) {
  L0_DFX(TensorEqualAiCpu, self, other);

  static internal::AicpuTaskSpace space("TensorEqual");
  auto ret = ADD_TO_LAUNCHER_LIST_AICPU(TensorEqual, OP_ATTR_NAMES(), OP_INPUT(self, other), OP_OUTPUT(eqOut));
  CHECK_RET(ret == ACLNN_SUCCESS, nullptr);

  return eqOut;
}

const aclTensor *TensorEqual(const aclTensor *self, const aclTensor *other, aclOpExecutor *executor) {
  // 输出一个tensor [True],一维一个bool元素
  op::Shape outShape;
  outShape.SetDimNum(1);
  outShape.SetDim(0, 1);

  auto eqOut = executor->AllocTensor(outShape, op::DataType::DT_BOOL);
  if (IsAiCoreSupport(self) && IsAiCoreSupport(other)) {
    return TensorEqualAiCore(self, other, eqOut, executor);
  } else {
    return TensorEqualAiCpu(self, other, eqOut, executor);
  }
}
}  // namespace l0op
