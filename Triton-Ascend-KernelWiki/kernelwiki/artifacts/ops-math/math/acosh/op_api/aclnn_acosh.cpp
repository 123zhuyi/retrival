/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include "aclnn_acosh.h"
#include "acosh.h"
#include "aclnn_kernels/cast.h"
#include "aclnn_kernels/contiguous.h"
#include "aclnn/aclnn_base.h"
#include "aclnn_kernels/common/op_error_check.h"
#include "op_api/aclnn_check.h"
#include "op_api/op_api_def.h"
#include "opdev/common_types.h"
#include "opdev/data_type_utils.h"
#include "opdev/format_utils.h"
#include "opdev/op_dfx.h"
#include "opdev/op_executor.h"
#include "opdev/op_log.h"
#include "opdev/tensor_view_utils.h"
#include "opdev/platform.h"

using namespace op;
#ifdef __cplusplus
extern "C" {
#endif

// 根据API定义，需要列出所能支持的所有dtype
static const std::initializer_list<DataType> ASCEND910_DTYPE_DTYPE_SUPPORT_LIST = {
    DataType::DT_FLOAT, DataType::DT_FLOAT16, DataType::DT_DOUBLE, DataType::DT_INT8,
    DataType::DT_INT16, DataType::DT_INT32, DataType::DT_INT64, DataType::DT_BOOL,
    DataType::DT_COMPLEX64, DataType::DT_COMPLEX128, DataType::DT_UINT8};

static const std::initializer_list<DataType> ASCEND910B_DTYPE_DTYPE_SUPPORT_LIST = {
    DataType::DT_FLOAT, DataType::DT_FLOAT16, DataType::DT_DOUBLE, DataType::DT_INT8,
    DataType::DT_INT16, DataType::DT_INT32, DataType::DT_INT64, DataType::DT_BOOL,
    DataType::DT_COMPLEX64, DataType::DT_COMPLEX128, DataType::DT_UINT8, DataType::DT_BF16};

static const std::initializer_list<DataType> OUTPUT_DTYPE_SUPPORT_LIST = {
    DataType::DT_FLOAT, DataType::DT_FLOAT16, DataType::DT_DOUBLE,
    DataType::DT_COMPLEX64, DataType::DT_COMPLEX128, DataType::DT_BF16};

static const std::initializer_list<DataType>& GetDtypeSupportList() {
  if (GetCurrentPlatformInfo().GetSocVersion() == SocVersion::ASCEND910B ||
      GetCurrentPlatformInfo().GetSocVersion() == SocVersion::ASCEND910_93 ||
      IsRegBase()) {
    return ASCEND910B_DTYPE_DTYPE_SUPPORT_LIST;
  } else {
    return ASCEND910_DTYPE_DTYPE_SUPPORT_LIST;
  }
}

// 检查输入是否是空指针
static inline bool CheckNotNull(const aclTensor* self, const aclTensor* out) {
  OP_CHECK_NULL(self, return false);
  OP_CHECK_NULL(out, return false);
  return true;
}

static bool CheckDtypeValid(const aclTensor* self, const aclTensor* out) {
  // 检查输入和输出的数据类型是否在算子的支持列表内
  auto supportList = GetDtypeSupportList();
  OP_CHECK_DTYPE_NOT_SUPPORT(self, supportList, return false);
  OP_CHECK_DTYPE_NOT_SUPPORT(out, OUTPUT_DTYPE_SUPPORT_LIST, return false);
  return true;
}

static bool CheckShape(const aclTensor* self, const aclTensor* out) {
  // 输入和输出的shape必须一致
  OP_CHECK_SHAPE_NOT_EQUAL(out, self, return false);
  // 输入的维度必须小于等于8
  OP_CHECK_MAX_DIM(self, MAX_SUPPORT_DIMS_NUMS, return false);
  return true;
}

static bool CheckPromoteType(const aclTensor *self, const aclTensor *out) {
  OP_CHECK_RESULT_DTYPE_CAST_FAILED(self->GetDataType(), out->GetDataType(), return false);
  return true;
}

static aclnnStatus CheckParams(const aclTensor* self, const aclTensor* out) {
  // 1. 检查参数是否为空指针
  CHECK_RET(CheckNotNull(self, out), ACLNN_ERR_PARAM_NULLPTR);

  // 2. 检查输入和输出的数据类型是否满足约束，需要根据api定义校验
  CHECK_RET(CheckDtypeValid(self, out), ACLNN_ERR_PARAM_INVALID);

  // 3. 检查输入和输出的shape是否满足约束
  CHECK_RET(CheckShape(self, out), ACLNN_ERR_PARAM_INVALID);

  // 4. 检查self是否可以转成out数据类型
  CHECK_RET(CheckPromoteType(self, out), ACLNN_ERR_PARAM_INVALID);

  return ACLNN_SUCCESS;
}

aclnnStatus ExecAcoshGetWorkspaceSize(const aclTensor* self, aclTensor* out, uint64_t* workspaceSize,
                                      aclOpExecutor** executor) {
  // 创建OpExecutor
  auto uniqueExecutor = CREATE_EXECUTOR();
  CHECK_RET(uniqueExecutor.get() != nullptr, ACLNN_ERR_INNER_CREATE_EXECUTOR);

  // 参数检查
  auto ret = CheckParams(self, out);
  CHECK_RET(ret == ACLNN_SUCCESS, ret);

  // 空tensor支持
  if (self->IsEmpty()) {
    *workspaceSize = 0;
    uniqueExecutor.ReleaseTo(executor);
    return ACLNN_SUCCESS;
  }

  // 固定写法，将输入self转换成连续的tensor
  auto selfContiguous = l0op::Contiguous(self, uniqueExecutor.get());
  CHECK_RET(selfContiguous != nullptr, ACLNN_ERR_INNER_NULLPTR);

  // 调用cast算子将不支持的类型转化为float
  auto castDtype = selfContiguous->GetDataType();
  if (!CheckType(castDtype, OUTPUT_DTYPE_SUPPORT_LIST)) {
    castDtype = DataType::DT_FLOAT;
  }
  auto selfCast = l0op::Cast(selfContiguous, castDtype, uniqueExecutor.get());
  CHECK_RET(selfCast != nullptr, ACLNN_ERR_INNER_NULLPTR);

  // 调用Acosh算子
  auto acoshOutRet = l0op::Acosh(selfCast, uniqueExecutor.get());
  CHECK_RET(acoshOutRet != nullptr, ACLNN_ERR_INNER_NULLPTR);

  // 固定写法，将计算结果转换成输出out的数据类型
  auto castOut = l0op::Cast(acoshOutRet, out->GetDataType(), uniqueExecutor.get());
  CHECK_RET(castOut != nullptr, ACLNN_ERR_INNER_NULLPTR);

  // 固定写法，将计算结果拷贝到输出out上，out可能是非连续的tensor
  auto viewCopyResult = l0op::ViewCopy(castOut, out, uniqueExecutor.get());
  CHECK_RET(viewCopyResult != nullptr, ACLNN_ERR_INNER_NULLPTR);

  // 固定写法，获取计算过程中需要使用的workspace大小
  *workspaceSize = uniqueExecutor->GetWorkspaceSize();
  uniqueExecutor.ReleaseTo(executor);
  return ACLNN_SUCCESS;
}

aclnnStatus aclnnAcoshGetWorkspaceSize(const aclTensor *self, aclTensor *out, uint64_t *workspaceSize,
    aclOpExecutor **executor) {
    L2_DFX_PHASE_1(aclnnAcosh, DFX_IN(self), DFX_OUT(out));
    return ExecAcoshGetWorkspaceSize(self, out, workspaceSize, executor);
}

aclnnStatus aclnnInplaceAcoshGetWorkspaceSize(aclTensor *selfRef, uint64_t *workspaceSize,
    aclOpExecutor **executor) {
    L2_DFX_PHASE_1(aclnnInplaceAcosh, DFX_IN(selfRef), DFX_OUT(selfRef));
    return ExecAcoshGetWorkspaceSize(selfRef, selfRef, workspaceSize, executor);
}

aclnnStatus aclnnAcosh(void* workspace, uint64_t workspaceSize, aclOpExecutor* executor, aclrtStream stream) {
    // 固定写法，调用框架能力，完成计算
    L2_DFX_PHASE_2(aclnnAcosh);
    return CommonOpExecutorRun(workspace, workspaceSize, executor, stream);
}

aclnnStatus aclnnInplaceAcosh(void *workspace, uint64_t workspaceSize, aclOpExecutor *executor, aclrtStream stream) {
    // 固定写法，调用框架能力，完成计算
    L2_DFX_PHASE_2(aclnnInplaceAcosh);
    return CommonOpExecutorRun(workspace, workspaceSize, executor, stream);
}
 
#ifdef __cplusplus
}
#endif
