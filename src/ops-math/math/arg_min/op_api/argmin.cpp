/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include "argmin.h"
#include <climits>
#include "opdev/aicpu/aicpu_task.h"
#include "opdev/op_dfx.h"
#include "opdev/op_log.h"
#include "aclnn_kernels/common/op_error_check.h"
#include "op_api/aclnn_check.h"

using namespace op;
namespace l0op {
OP_TYPE_REGISTER(ArgMin);

static const std::initializer_list<op::DataType> AICORE_DTYPE_SUPPORT_LIST = {op::DataType::DT_FLOAT,
                                                                              op::DataType::DT_FLOAT16,
                                                                              op::DataType::DT_INT64};

static const std::initializer_list<op::DataType> AICORE_310P_DTYPE_SUPPORT_LIST = {op::DataType::DT_FLOAT,
                                                                                   op::DataType::DT_FLOAT16};

static const std::initializer_list<op::DataType> AICORE_910B_DTYPE_SUPPORT_LIST = {op::DataType::DT_FLOAT,
                                                                                   op::DataType::DT_FLOAT16,
                                                                                   op::DataType::DT_INT64,
                                                                                   op::DataType::DT_BF16};

static bool IsAiCoreSupport(const aclTensor *self) {
  auto curArch = GetCurrentPlatformInfo().GetCurNpuArch();
  if (curArch == NpuArch::DAV_2002) {
    return CheckType(self->GetDataType(), AICORE_310P_DTYPE_SUPPORT_LIST);
  }
  if (curArch == NpuArch::DAV_2201 || IsRegBase(curArch)) {
    return CheckType(self->GetDataType(), AICORE_910B_DTYPE_SUPPORT_LIST);
  }
  return CheckType(self->GetDataType(), AICORE_DTYPE_SUPPORT_LIST);
}

// AICORE算子kernel
static const aclTensor* ArgMinAiCore(const aclTensor* x, const aclTensor* dim, aclTensor* out,
                                     aclOpExecutor* executor) {
  L0_DFX(ArgMinAiCore, x, dim, out);
  auto ret = ADD_TO_LAUNCHER_LIST_AICORE(ArgMin, OP_INPUT(x, dim), OP_OUTPUT(out));
  OP_CHECK(ret ==  ACL_SUCCESS, OP_LOGE(ACLNN_ERR_INNER_NULLPTR, "ArgMinAiCore ADD_TO_LAUNCHER_LIST_AICORE failed."), return nullptr);
  return out;
}

// AICPU算子kernel
static const aclTensor* ArgMinAiCpu(const aclTensor* x, const aclTensor* dim, aclTensor* out, aclOpExecutor* executor) {
  L0_DFX(ArgMinAiCpu, x, dim, out);

  static internal::AicpuTaskSpace space("ArgMin", ge::DEPEND_IN_SHAPE, true);
  auto ret = ADD_TO_LAUNCHER_LIST_AICPU(ArgMin, OP_ATTR_NAMES({"Tidx", "output_type"}), OP_INPUT(x, dim),
                                        OP_OUTPUT(out), OP_ATTR(dim->GetDataType(), out->GetDataType()));
  OP_CHECK(ret ==  ACL_SUCCESS, OP_LOGE(ACLNN_ERR_INNER_NULLPTR, "ArgMinAiCpu ADD_TO_LAUNCHER_LIST_AICPU failed."), return nullptr);
  return out;
}

const aclTensor* ArgMin(const aclTensor* x, const int64_t dim, const bool keepdim,
                        op::DataType outDataType, aclOpExecutor* executor) {
  auto dimTensor = executor->ConvertToTensor(executor->AllocScalar(dim), op::DataType::DT_INT64);
  int64_t dimNum = x->GetViewShape().GetDim(dim);
  aclTensor* out;
  auto curArch = GetCurrentPlatformInfo().GetCurNpuArch();
  if (curArch == NpuArch::DAV_2002) {
    // 310p not support index int64_t
    out = executor->AllocTensor(op::ToOpDataType(ACL_INT32), op::Format::FORMAT_ND, op::Format::FORMAT_ND);
  } else {
    if (dimNum > INT_MAX || outDataType == op::DataType::DT_INT64) {
      out = executor->AllocTensor(op::ToOpDataType(ACL_INT64), op::Format::FORMAT_ND, op::Format::FORMAT_ND);
    } else {
      out = executor->AllocTensor(op::ToOpDataType(ACL_INT32), op::Format::FORMAT_ND, op::Format::FORMAT_ND);
    }
  }

  auto ret = INFER_SHAPE(ArgMin, OP_INPUT(x, dimTensor), OP_OUTPUT(out), OP_EMPTY_ARG);
  if (ret != ACLNN_SUCCESS) {
    OP_LOGE(ACLNN_ERR_PARAM_INVALID, "ArgMin InferShape failed.");
    return nullptr;
  }

  if (keepdim) {
    op::Shape inputShape = x->GetViewShape();
    size_t realDim = dim;
    if (dim < 0) {
      realDim = inputShape.GetDimNum() + dim;
    }
    inputShape.SetDim(realDim, 1);
    out->SetViewShape(inputShape);
    out->SetStorageShape(inputShape);
    out->SetOriginalShape(inputShape);
  }

  if (IsAiCoreSupport(x)) {
    return ArgMinAiCore(x, dimTensor, out, executor);
  } else {
    return ArgMinAiCpu(x, dimTensor, out, executor);
  }
}
}  // namespace l0op
