/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include "reduce_logsumexp.h"
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
OP_TYPE_REGISTER(ReduceLogSumExp);

static const std::initializer_list<op::DataType> aicore_dtype_support_list = {DataType::DT_FLOAT, DataType::DT_FLOAT16};

// 根据芯片类型、dtype判断算子是否支持走aicore
static bool IsAiCoreSupport(const aclTensor *self) {
    return CheckType(self->GetDataType(), aicore_dtype_support_list);
}

static const aclTensor *GenerateDimTensor(const aclTensor *self, const aclIntArray *dim, aclOpExecutor *executor) {
  if (dim->Size() == 0) {
    FVector<int64_t> dimVector;
    for (size_t i = 0; i < self->GetViewShape().GetDimNum(); i++) {
      dimVector.emplace_back(i);
    }
    return executor->ConvertToTensor(dimVector.data(), dimVector.size(), DataType::DT_INT64);
  }
  return executor->ConvertToTensor(dim, DataType::DT_INT64);
}

// AICORE算子kernel
static const aclTensor *ReduceLogSumExpAiCore(const aclTensor *self, const aclTensor *dim, bool keepDim,
                                              const aclTensor *logSumExpOut, aclOpExecutor *executor) {
    L0_DFX(ReduceLogSumExpAiCore, self, dim, keepDim, logSumExpOut);
    // 使用框架宏ADD_TO_KERNEL_OBJ_LIST，将Mean算子加入任务队列
    bool noopWithEmptyAxes = true; // useless attr to avoid dynamic reduce_mean_d compile error.
    ADD_TO_LAUNCHER_LIST_AICORE(ReduceLogSumExp, OP_INPUT(self, dim), OP_OUTPUT(logSumExpOut),
                                OP_ATTR(keepDim, noopWithEmptyAxes));

    return logSumExpOut;
}

const aclTensor* ReduceLogSumExp(const aclTensor* self, const aclIntArray* dim, bool keepDim, aclOpExecutor* executor) {
    // self如果为标量，不调用算子，直接返回
    if (self->GetViewShape().GetDimNum() == 0) {
        return self;
    }
    auto logSumExpOut = executor->AllocTensor(self->GetDataType(), op::Format::FORMAT_ND, op::Format::FORMAT_ND);
    auto dimTensor = GenerateDimTensor(self, dim, executor);
    INFER_SHAPE(ReduceLogSumExp, OP_INPUT(self, dimTensor), OP_OUTPUT(logSumExpOut), OP_ATTR(keepDim));
    
    if (IsAiCoreSupport(self)) {
      return ReduceLogSumExpAiCore(self, dimTensor, keepDim, logSumExpOut, executor);
    }
    return logSumExpOut;
}
}  // namespace l0op
