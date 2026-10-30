/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License")
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include "argmax_v2.h"
#include "opdev/aicpu/aicpu_task.h"
#include "opdev/op_dfx.h"
#include "opdev/op_log.h"
#include "opdev/platform.h"
#include "op_api/aclnn_check.h"
#include "aclnn_kernels/common/op_error_check.h"

using namespace op;
namespace l0op {

OP_TYPE_REGISTER(ArgMaxV2);

static const std::initializer_list<op::DataType> AICORE_DTYPE_SUPPORT_LIST_310P = {
    op::DataType::DT_FLOAT, op::DataType::DT_FLOAT16};

static const std::initializer_list<op::DataType> AICORE_DTYPE_SUPPORT_LIST_910 = {
    op::DataType::DT_FLOAT, op::DataType::DT_FLOAT16, op::DataType::DT_INT64};

static const std::initializer_list<op::DataType> AICORE_DTYPE_SUPPORT_LIST_910B = {
    op::DataType::DT_FLOAT, op::DataType::DT_FLOAT16, op::DataType::DT_BF16, op::DataType::DT_INT32,
    op::DataType::DT_INT64};

static const std::initializer_list<op::DataType> AICORE_DTYPE_SUPPORT_LIST_610LITE = {
    op::DataType::DT_FLOAT, op::DataType::DT_FLOAT16, op::DataType::DT_INT32};

static bool IsAiCoreSupport(const aclTensor *self) {
  auto curArch = GetCurrentPlatformInfo().GetCurNpuArch();
  if (curArch == NpuArch::DAV_2201) {
    return CheckType(self->GetDataType(), AICORE_DTYPE_SUPPORT_LIST_910B);
  } else if (curArch == NpuArch::DAV_1001) {
    return CheckType(self->GetDataType(), AICORE_DTYPE_SUPPORT_LIST_910);
  } else if (curArch == NpuArch::DAV_3102) {
    return CheckType(self->GetDataType(), AICORE_DTYPE_SUPPORT_LIST_610LITE);
  } else if (IsRegBase(curArch)) {
    return CheckType(self->GetDataType(), AICORE_DTYPE_SUPPORT_LIST_910B);
  }
  return CheckType(self->GetDataType(), AICORE_DTYPE_SUPPORT_LIST_310P);
}

// AICORE算子kernel
static const aclTensor *ArgMaxV2AiCore(const aclTensor *x, const aclTensor *dim, aclTensor *out,
                                       aclOpExecutor *executor) {
  L0_DFX(ArgMaxV2AiCore, x, dim, out);
  auto ret = ADD_TO_LAUNCHER_LIST_AICORE(ArgMaxV2, OP_INPUT(x, dim), OP_OUTPUT(out));
  OP_CHECK(ret ==  ACL_SUCCESS, OP_LOGE(ACLNN_ERR_INNER_NULLPTR, "ArgMaxV2AiCore ADD_TO_LAUNCHER_LIST_AICORE failed."), return nullptr);
  return out;
}

// AICPU算子kernel
static const aclTensor *ArgMaxV2AiCpu(const aclTensor *x, const aclTensor *dim, aclTensor *out,
                                      aclOpExecutor *executor) {
  L0_DFX(ArgMaxV2AiCpu, x, dim, out);
  static internal::AicpuTaskSpace space("ArgMax", ge::DEPEND_IN_SHAPE, true);
  auto ret = ADD_TO_LAUNCHER_LIST_AICPU(ArgMaxV2,
                                        OP_ATTR_NAMES({"Tidx", "output_type"}),
                                        OP_INPUT(x, dim),
                                        OP_OUTPUT(out),
                                        OP_ATTR(dim->GetDataType(), out->GetDataType()));
  OP_CHECK(ret ==  ACL_SUCCESS, OP_LOGE(ACLNN_ERR_INNER_NULLPTR, "ArgMaxV2AiCpu ADD_TO_LAUNCHER_LIST_AICPU failed."), return nullptr);
  return out;
}

const aclTensor *ArgMaxV2(const aclTensor *x, const int64_t dim, const bool keepdim, aclOpExecutor *executor) {
  auto dimTensor = executor->ConvertToTensor(executor->AllocScalar(dim), op::DataType::DT_INT64);
  auto out = executor->AllocTensor(op::ToOpDataType(ACL_INT32), op::Format::FORMAT_ND, op::Format::FORMAT_ND);
  auto ret = INFER_SHAPE(ArgMaxV2, OP_INPUT(x, dimTensor), OP_OUTPUT(out), OP_EMPTY_ARG);
  if (ret != ACLNN_SUCCESS) {
    OP_LOGE(ACLNN_ERR_PARAM_INVALID, "ArgMaxV2 InferShape failed.");
    return nullptr;
  }

  if (keepdim) {
    op::Shape inputShape = x->GetViewShape();
    inputShape.SetDim(dim, 1);
    out->SetViewShape(inputShape);
    out->SetStorageShape(inputShape);
  }

  if (IsAiCoreSupport(x)) {
    return ArgMaxV2AiCore(x, dimTensor, out, executor);
  } else {
    return ArgMaxV2AiCpu(x, dimTensor, out, executor);
  }
}
}
