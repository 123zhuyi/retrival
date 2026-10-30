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
 * \file aclnn_lerp_scalar.cpp
 * \brief
 */

#include "aclnn_lerp_scalar.h"
#include "lerp.h"
#include "aclnn_kernels/contiguous.h"
#include "aclnn_kernels/cast.h"
#include "aclnn_kernels/reshape.h"
#include "aclnn_kernels/common/op_error_check.h"
#include "opdev/common_types.h"
#include "opdev/data_type_utils.h"
#include "opdev/format_utils.h"
#include "opdev/op_executor.h"
#include "opdev/op_dfx.h"
#include "opdev/op_log.h"
#include "opdev/shape_utils.h"
#include "opdev/tensor_view_utils.h"
#include "opdev/platform.h"
#include "op_api/op_api_def.h"
#include "op_api/aclnn_check.h"

using namespace op;

static const std::initializer_list<DataType> Ascend910_input_dtype_support_list = {op::DataType::DT_FLOAT16,
                                                                                   op::DataType::DT_FLOAT};

static const std::initializer_list<DataType> Ascend910B_input_dtype_support_list = {
    op::DataType::DT_FLOAT16, op::DataType::DT_FLOAT, op::DataType::DT_BF16};

static const std::initializer_list<DataType> Ascend910_out_dtype_support_list = {
    op::DataType::DT_FLOAT16, op::DataType::DT_FLOAT, op::DataType::DT_DOUBLE};

static const std::initializer_list<DataType> Ascend910B_out_dtype_support_list = {
    op::DataType::DT_FLOAT16, op::DataType::DT_FLOAT, op::DataType::DT_DOUBLE, op::DataType::DT_BF16};

static const std::initializer_list<DataType>& GetSelfDtypeSupportList() {
  auto npuArch = op::GetCurrentPlatformInfo().GetCurNpuArch();
  if (npuArch == NpuArch::DAV_2201 || IsRegBase(npuArch)) {
    return Ascend910B_input_dtype_support_list;
  }
  return Ascend910_input_dtype_support_list;
}

static const std::initializer_list<DataType>& GetOutDtypeSupportList() {
  auto npuArch = op::GetCurrentPlatformInfo().GetCurNpuArch();
  if (npuArch == NpuArch::DAV_2201 || IsRegBase(npuArch)) {
    return Ascend910B_out_dtype_support_list;
  }
  return Ascend910_out_dtype_support_list;
}

static bool CheckNotNull(const aclTensor* self, const aclTensor* end, const aclScalar* weight, const aclTensor* out) {
  OP_CHECK_NULL(self, return false);
  OP_CHECK_NULL(end, return false);
  OP_CHECK_NULL(weight, return false);
  OP_CHECK_NULL(out, return false);

  return true;
}

static bool CheckDtypeValid(const aclTensor* self, const aclTensor* end,
                            const aclTensor* out) {
  // 检查self的数据类型是否在支持列表内
  OP_CHECK_DTYPE_NOT_SUPPORT(self, GetSelfDtypeSupportList(), return false);

  // 检查end的数据类型是否在支持列表内
  OP_CHECK_DTYPE_NOT_SUPPORT(end, GetSelfDtypeSupportList(), return false);

  // 检查self, end的数据类型是否匹配
  OP_CHECK_DTYPE_NOT_MATCH(end, self->GetDataType(), return false);

  // 检查out的数据类型是否在支持列表内
  OP_CHECK_DTYPE_NOT_SUPPORT(out, GetOutDtypeSupportList(), return false);

  return true;
}

static bool CheckShape(const aclTensor* self, const aclTensor* end, const aclTensor* y) {
  op::Shape broadcastShape;
  if (!BroadcastInferShape(self->GetViewShape(), end->GetViewShape(), broadcastShape)) {
    OP_LOGE(ACLNN_ERR_PARAM_INVALID, "Shape of self and end can't broadcast.");
    return false;
  }

  if (broadcastShape != y->GetViewShape()) {
    OP_LOGE(ACLNN_ERR_PARAM_INVALID, "Shape of out should be %s, but current is %s.",
            op::ToString(broadcastShape).GetString(), op::ToString(y->GetViewShape()).GetString());
    return false;
  }

  if (IsContiguous(self) && IsContiguous(end) && IsContiguous(y)) {
    if(self->GetViewShape().GetDimNum() > static_cast<int64_t>(MAX_SUPPORT_DIMS_NUMS))  {
      OP_LOGW("The dimension of the self tensor is greater than 8");
    }
    return true;
  }

  OP_CHECK_MAX_DIM(self, MAX_SUPPORT_DIMS_NUMS, return false);
  OP_CHECK_MAX_DIM(end, MAX_SUPPORT_DIMS_NUMS, return false);
  return true;
}

static aclnnStatus CheckParams(const aclTensor* self, const aclTensor* end, const aclScalar* weight,
                               const aclTensor* out) {
  // 检查参数是否为空指针
  CHECK_COND(CheckNotNull(self, end, weight, out), ACLNN_ERR_PARAM_NULLPTR, "CheckNotNull failed!");

  // 检查输入的数据类型是否在API支持的数据类型范围之内，需要根据api定义校验
  CHECK_COND(CheckDtypeValid(self, end, out), ACLNN_ERR_PARAM_INVALID, "CheckDtypeValid failed!");

  // 检查shape是否满足约束
  CHECK_COND(CheckShape(self, end, out), ACLNN_ERR_PARAM_INVALID, "CheckShape failed!");
  return ACLNN_SUCCESS;
}

static inline op::DataType InferScalarTensorDtype(const aclTensor* self) {
  auto npuArch = op::GetCurrentPlatformInfo().GetCurNpuArch();
  if (IsRegBase(npuArch)) {
    return op::DataType::DT_FLOAT;
  }
  return self->GetDataType();
}

static aclnnStatus CalculateResult(const aclTensor* self, const aclTensor* end, const aclScalar* weight, aclTensor* out,
                           aclOpExecutor* executor) {
  // 固定写法，参数检查
  auto ret = CheckParams(self, end, weight, out);
  CHECK_RET(ret == ACLNN_SUCCESS, ret);

  // 空Tensor处理
  if (self->IsEmpty()) {
    return ACLNN_SUCCESS;
  }

  // self如果非连续，需要转连续
  auto selfContiguous = l0op::Contiguous(self, executor);
  CHECK_COND(selfContiguous != nullptr, ACLNN_ERR_INNER_NULLPTR, "InitializeTensor self failed!");

  if (self->GetStorageFormat() != Format::FORMAT_ND || end->GetStorageFormat() != Format::FORMAT_ND) {
    OP_LOGW("Format only support ND");
  }
  
  // end如果非连续，需要转连续
  auto endContiguous = l0op::Contiguous(end, executor);
  CHECK_COND(endContiguous != nullptr, ACLNN_ERR_INNER_NULLPTR, "InitializeTensor end failed!");

  // 将Scalar转成Tensor
  auto weightTensor = executor->ConvertToTensor(weight, InferScalarTensorDtype(self));
  CHECK_COND(weightTensor != nullptr, ACLNN_ERR_INNER_NULLPTR, "convert weight to tensor failed!");

  // 调用l0算子Lerp进行计算
  auto lerpResult = l0op::Lerp(selfContiguous, endContiguous, weightTensor, executor);
  CHECK_COND(lerpResult != nullptr, ACLNN_ERR_INNER_NULLPTR, "Lerp failed!");

  // 调用l0算子将输出转成指定类型
  auto lerpResultCasted = l0op::Cast(lerpResult, out->GetDataType(), executor);
  CHECK_COND(lerpResultCasted != nullptr, ACLNN_ERR_INNER_NULLPTR, "cast out failed!");

  // 如果出参out是非连续Tensor，需要把计算完的连续Tensor转非连续
  auto viewCopyResult = l0op::ViewCopy(lerpResultCasted, out, executor);
  CHECK_COND(viewCopyResult != nullptr, ACLNN_ERR_INNER_NULLPTR, "viewcopy out failed!");

  return ACLNN_SUCCESS;
}

aclnnStatus aclnnLerpsGetWorkspaceSize(const aclTensor* self, const aclTensor* end, const aclScalar* weight,
                                       aclTensor* out, uint64_t* workspaceSize, aclOpExecutor** executor) {
  OP_CHECK_COMM_INPUT(workspaceSize, executor);
  L2_DFX_PHASE_1(aclnnLerps, DFX_IN(self, end, weight), DFX_OUT(out));

  // 固定写法，创建OpExecutor
  auto uniqueExecutor = CREATE_EXECUTOR();
  CHECK_COND(uniqueExecutor.get() != nullptr, ACLNN_ERR_INNER_CREATE_EXECUTOR, "CREATE_EXECUTOR failed!");

  auto ret = CalculateResult(self, end, weight, out, uniqueExecutor.get());
  CHECK_RET(ret == ACLNN_SUCCESS, ret);

  // 固定写法，获取计算过程中需要使用的workspace大小
  *workspaceSize = uniqueExecutor->GetWorkspaceSize();
  uniqueExecutor.ReleaseTo(executor);
  return ACLNN_SUCCESS;
}

aclnnStatus aclnnInplaceLerpsGetWorkspaceSize(aclTensor* selfRef, const aclTensor* end, const aclScalar* weight,
                                              uint64_t* workspaceSize, aclOpExecutor** executor) {
  OP_CHECK_COMM_INPUT(workspaceSize, executor);
  L2_DFX_PHASE_1(aclnnInplaceLerps, DFX_IN(selfRef, end, weight), DFX_OUT(selfRef));

  // 固定写法，创建OpExecutor
  auto uniqueExecutor = CREATE_EXECUTOR();
  CHECK_COND(uniqueExecutor.get() != nullptr, ACLNN_ERR_INNER_CREATE_EXECUTOR, "CREATE_EXECUTOR failed!");

  auto ret = CalculateResult(selfRef, end, weight, selfRef, uniqueExecutor.get());
  CHECK_RET(ret == ACLNN_SUCCESS, ret);

  // 固定写法，获取计算过程中需要使用的workspace大小
  *workspaceSize = uniqueExecutor->GetWorkspaceSize();
  uniqueExecutor.ReleaseTo(executor);
  return ACLNN_SUCCESS;
}

aclnnStatus aclnnLerps(void* workspace, uint64_t workspaceSize, aclOpExecutor* executor, const aclrtStream stream) {
  L2_DFX_PHASE_2(aclnnLerps);
  // 固定写法，调用框架能力，完成计算
  return CommonOpExecutorRun(workspace, workspaceSize, executor, stream);
}

aclnnStatus aclnnInplaceLerps(void* workspace, uint64_t workspaceSize, aclOpExecutor* executor,
                              const aclrtStream stream) {
  L2_DFX_PHASE_2(aclnnInplaceLerps);
  // 固定写法，调用框架能力，完成计算
  return CommonOpExecutorRun(workspace, workspaceSize, executor, stream);
}
