/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include "scale.h"
#include "opdev/data_type_utils.h"
#include "opdev/aicpu/aicpu_task.h"
#include "opdev/format_utils.h"
#include "opdev/make_op_executor.h"
#include "opdev/op_def.h"
#include "opdev/op_dfx.h"
#include "opdev/op_executor.h"
#include "opdev/op_log.h"
#include "opdev/shape_utils.h"

using namespace op;

namespace l0op {

OP_TYPE_REGISTER(Scale);

static const std::initializer_list<op::DataType> AICORE_DTYPE_SUPPORT_LIST = {
    op::DataType::DT_FLOAT16, op::DataType::DT_FLOAT, op::DataType::DT_BF16};

// 根据芯片类型、dtype判断算子是否支持走aicore
static inline bool IsAiCoreSupport(const aclTensor* self) {
  // Scale只需要判断dtype
  return CheckType(self->GetDataType(), AICORE_DTYPE_SUPPORT_LIST);
}

const aclTensor* ScaleAicore(const aclTensor* x, const aclTensor* scale, const aclTensor* bias, const int64_t axis,
                             const int64_t numAxes, const bool scaleFromBlob, aclTensor* out, aclOpExecutor* executor) {
  L0_DFX(ScaleAicore, x, scale, bias, axis, numAxes, scaleFromBlob, out);
  auto ret = ADD_TO_LAUNCHER_LIST_AICORE(Scale, OP_INPUT(x, scale, bias), OP_OUTPUT(out), OP_ATTR(axis, numAxes, scaleFromBlob));
  OP_CHECK(ret ==  ACLNN_SUCCESS, OP_LOGE(ACLNN_ERR_INNER_NULLPTR, "ScaleAiCore ADD_TO_LAUNCHER_LIST_AICORE failed."),
    return nullptr);
  return out;
}

const aclTensor* Scale(const aclTensor* x, const aclTensor* scale, const aclTensor* bias, const int64_t axis,
                       const int64_t numAxes, const bool scaleFromBlob, aclOpExecutor* executor) {
  op::Shape outShape = x->GetViewShape();
  auto out = executor->AllocTensor(outShape, x->GetDataType());

  if (!IsAiCoreSupport(x)) {
    OP_LOGE(ACL_ERROR_INVALID_PARAM, "dtype of input x is not supported");
    return nullptr;
  }

  return ScaleAicore(x, scale, bias, axis, numAxes, scaleFromBlob, out, executor);
}
}  // namespace l0op
