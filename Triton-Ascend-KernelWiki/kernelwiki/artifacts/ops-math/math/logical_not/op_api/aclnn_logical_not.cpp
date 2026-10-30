/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License")
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include "logical_not.h"
#include "aclnn_kernels/contiguous.h"
#include "aclnn_kernels/cast.h"
#include "aclnn_kernels/reshape.h"
#include "op_api/op_api_def.h"
#include "aclnn_kernels/common/op_error_check.h"
#include "opdev/common_types.h"
#include "opdev/data_type_utils.h"
#include "opdev/format_utils.h"
#include "opdev/make_op_executor.h"
#include "opdev/op_dfx.h"
#include "opdev/op_log.h"
#include "opdev/shape_utils.h"
#include "opdev/tensor_view_utils.h"
#include "aclnn_logical_not.h"
using namespace op;

static const std::initializer_list<DataType> dtype_support_list = {
    op::DataType::DT_UINT8, op::DataType::DT_INT8,   op::DataType::DT_INT16,
    op::DataType::DT_INT32, op::DataType::DT_INT64,  op::DataType::DT_FLOAT16,
    op::DataType::DT_FLOAT, op::DataType::DT_DOUBLE, op::DataType::DT_BOOL, op::DataType::DT_BF16};


static bool CheckNotNull(const aclTensor* self, const aclTensor* out) {
  OP_CHECK_NULL(self, return false);
  OP_CHECK_NULL(out, return false);

  return true;
}

static bool CheckDtypeValid(const aclTensor* self, const aclTensor* out) {
  // 检查self的数据类型是否在支持列表内
  OP_CHECK_DTYPE_NOT_SUPPORT(self, dtype_support_list, return false);

  // 检查out的数据类型是否在支持列表内
  OP_CHECK_DTYPE_NOT_SUPPORT(out, dtype_support_list, return false);

  return true;
}

static bool CheckShape(const aclTensor* self, const aclTensor* out) {
  // self和out的shape必须一致
  OP_CHECK_SHAPE_NOT_EQUAL(self, out, return false);

  // 输入输出连续，不限制维度数
  if (IsContiguous(self) && IsContiguous(out)) {
    if(self->GetViewShape().GetDimNum() > static_cast<int64_t>(MAX_SUPPORT_DIMS_NUMS))  {
      OP_LOGW("The dimension of the self tensor is greater than 8");
    }
    return true;
  }

  OP_CHECK_MAX_DIM(self, MAX_SUPPORT_DIMS_NUMS, return false);
  OP_CHECK_MAX_DIM(out, MAX_SUPPORT_DIMS_NUMS, return false);

  return true;
}

static aclnnStatus CheckParams(const aclTensor* self, const aclTensor* out) {
  // 检查参数是否为空指针
  CHECK_RET(CheckNotNull(self, out), ACLNN_ERR_PARAM_NULLPTR);

  // 检查输入的数据类型是否在API支持的数据类型范围之内，需要根据api定义校验
  CHECK_RET(CheckDtypeValid(self, out), ACLNN_ERR_PARAM_INVALID);

  // 检查shape是否满足约束
  CHECK_RET(CheckShape(self, out), ACLNN_ERR_PARAM_INVALID);
  return ACLNN_SUCCESS;
}

static aclnnStatus CalculateResult(const aclTensor* self, aclTensor* out, aclOpExecutor* executor) {
  // 固定写法，参数检查
  auto ret = CheckParams(self, out);
  CHECK_RET(ret == ACLNN_SUCCESS, ret);
  
  // 空Tensor处理
  if (self->IsEmpty()) {
    return ACLNN_SUCCESS;
  }
  
  if(self->GetStorageFormat() != Format::FORMAT_ND){
    OP_LOGW("Format only support ND");
  }
  // self如果非连续，需要转连续
  auto selfContiguous = l0op::Contiguous(self, executor);
  CHECK_RET(selfContiguous != nullptr, ACLNN_ERR_INNER_NULLPTR);

  // 将输入self的数据类型转换成隐式数据类型，根据具体算子语义按需调用
  auto selfCasted = l0op::Cast(selfContiguous, DataType::DT_BOOL, executor);
  CHECK_RET(selfCasted != nullptr, ACLNN_ERR_INNER_NULLPTR);

  // 调用l0算子LogicalNot行计算
  auto result = l0op::LogicalNot(selfCasted, executor);
  CHECK_RET(result != nullptr, ACLNN_ERR_INNER_NULLPTR);

  auto resultCasted = l0op::Cast(result, out->GetDataType(), executor);
  CHECK_RET(resultCasted != nullptr, ACLNN_ERR_INNER_NULLPTR);

  // 如果出参out是非连续Tensor，需要把计算完的连续Tensor转非连续
  auto viewCopyResult = l0op::ViewCopy(resultCasted, out, executor);
  CHECK_RET(viewCopyResult != nullptr, ACLNN_ERR_INNER_NULLPTR);

  return ACLNN_SUCCESS;
}

aclnnStatus aclnnLogicalNotGetWorkspaceSize(const aclTensor* self, aclTensor* out, uint64_t* workspaceSize,
                                            aclOpExecutor** executor) {
  L2_DFX_PHASE_1(aclnnLogicalNot, DFX_IN(self), DFX_OUT(out));

  // 固定写法，创建OpExecutor
  auto uniqueExecutor = CREATE_EXECUTOR();
  CHECK_RET(uniqueExecutor.get() != nullptr, ACLNN_ERR_INNER_CREATE_EXECUTOR);
  
  auto ret = CalculateResult(self, out, uniqueExecutor.get());
  CHECK_RET(ret == ACLNN_SUCCESS, ret);

  // 固定写法，获取计算过程中需要使用的workspace大小
  *workspaceSize = uniqueExecutor->GetWorkspaceSize();
  uniqueExecutor.ReleaseTo(executor);
  return ACLNN_SUCCESS;
}

aclnnStatus aclnnInplaceLogicalNotGetWorkspaceSize(aclTensor* selfRef, uint64_t* workspaceSize,
                                                   aclOpExecutor** executor) {
  L2_DFX_PHASE_1(aclnnInplaceLogicalNot, DFX_IN(selfRef), DFX_OUT(selfRef));

  // 固定写法，创建OpExecutor
  auto uniqueExecutor = CREATE_EXECUTOR();
  CHECK_RET(uniqueExecutor.get() != nullptr, ACLNN_ERR_INNER_CREATE_EXECUTOR);

  auto ret = CalculateResult(selfRef, selfRef, uniqueExecutor.get());
  CHECK_RET(ret == ACLNN_SUCCESS, ret);

  // 固定写法，获取计算过程中需要使用的workspace大小
  *workspaceSize = uniqueExecutor->GetWorkspaceSize();
  uniqueExecutor.ReleaseTo(executor);
  return ACLNN_SUCCESS;
}

aclnnStatus aclnnLogicalNot(void* workspace, uint64_t workspaceSize, aclOpExecutor* executor,
                            aclrtStream stream) {
  L2_DFX_PHASE_2(aclnnLogicalNot);
  // 固定写法，调用框架能力，完成计算
  return CommonOpExecutorRun(workspace, workspaceSize, executor, stream);
}

aclnnStatus aclnnInplaceLogicalNot(void* workspace, uint64_t workspaceSize, aclOpExecutor* executor,
                                   aclrtStream stream) {
  L2_DFX_PHASE_2(aclnnInplaceLogicalNot);
  // 固定写法，调用框架能力，完成计算
  return CommonOpExecutorRun(workspace, workspaceSize, executor, stream);
}
