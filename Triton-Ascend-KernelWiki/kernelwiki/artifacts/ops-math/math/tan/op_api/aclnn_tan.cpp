/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include "aclnn_tan.h"
#include "tan.h"
#include "aclnn_kernels/cast.h"
#include "aclnn_kernels/contiguous.h"
#include "aclnn/aclnn_base.h"
#include "opdev/common_types.h"
#include "opdev/shape_utils.h"
#include "opdev/data_type_utils.h"
#include "opdev/format_utils.h"
#include "opdev/op_dfx.h"
#include "opdev/op_executor.h"
#include "opdev/op_log.h"
#include "opdev/tensor_view_utils.h"
#include "aclnn_kernels/common/op_error_check.h"
#include "op_api/aclnn_check.h"
#include "op_api/op_api_def.h"
#include "opdev/platform.h"

using namespace op;
#ifdef __cplusplus
extern "C" {
#endif

static const std::initializer_list<op::DataType> ASCEND910_DTYPE_DTYPE_SUPPORT_LIST = {
  op::DataType::DT_FLOAT, op::DataType::DT_FLOAT16, op::DataType::DT_DOUBLE, op::DataType::DT_INT8,
  op::DataType::DT_INT16, op::DataType::DT_INT32, op::DataType::DT_INT64, op::DataType::DT_UINT8,
  op::DataType::DT_BOOL, op::DataType::DT_COMPLEX64, op::DataType::DT_COMPLEX128};

static const std::initializer_list<op::DataType> ASCEND910B_DTYPE_DTYPE_SUPPORT_LIST = {
  op::DataType::DT_FLOAT, op::DataType::DT_FLOAT16, op::DataType::DT_DOUBLE, op::DataType::DT_INT8,
  op::DataType::DT_INT16, op::DataType::DT_INT32, op::DataType::DT_INT64, op::DataType::DT_UINT8,
  op::DataType::DT_BOOL, op::DataType::DT_COMPLEX64, op::DataType::DT_COMPLEX128, op::DataType::DT_BF16};

static const std::initializer_list<op::DataType> NEED_CAST_TO_FLOAT_DTYPE_LIST = {
  op::DataType::DT_INT8, op::DataType::DT_INT16, op::DataType::DT_INT32, op::DataType::DT_INT64,
  op::DataType::DT_UINT8, op::DataType::DT_BOOL};

static const std::initializer_list<DataType>& GetDtypeSupportList() {
  if (GetCurrentPlatformInfo().GetSocVersion() == SocVersion::ASCEND910B ||
      GetCurrentPlatformInfo().GetSocVersion() == SocVersion::ASCEND910_93 ||
      IsRegBase()) {
    return ASCEND910B_DTYPE_DTYPE_SUPPORT_LIST;
  } else {
    return ASCEND910_DTYPE_DTYPE_SUPPORT_LIST;
  }
}

static bool CheckDtypeValid(const aclTensor *self) {
  auto supportList = GetDtypeSupportList();
  OP_CHECK_DTYPE_NOT_SUPPORT(self, supportList, return false);
  return true;
}

static bool CheckDtypeCanCast(const aclTensor *self, const aclTensor *out) {
  auto selfDtype = self->GetDataType();
  auto outDtype = out->GetDataType();
  if (CheckType(selfDtype, NEED_CAST_TO_FLOAT_DTYPE_LIST)) {
    selfDtype = op::DataType::DT_FLOAT;
  }
  OP_CHECK_RESULT_DTYPE_CAST_FAILED(selfDtype, outDtype, return false);
  return true;
}

static bool CheckNotNull(const aclTensor *self, const aclTensor *out) {
  OP_CHECK_NULL(self, return false);
  OP_CHECK_NULL(out, return false);
  return true;
}

static bool CheckShape(const aclTensor *self, const aclTensor *out) {
  OP_CHECK_MAX_DIM(self, MAX_SUPPORT_DIMS_NUMS, return false);
  OP_CHECK_MAX_DIM(out, MAX_SUPPORT_DIMS_NUMS, return false);
  OP_CHECK_SHAPE_NOT_EQUAL(self, out, return false);
  return true;
}

static aclnnStatus Compute(const aclTensor *self, aclTensor *out, aclOpExecutor *executor) {
  auto selfContiguous = l0op::Contiguous(self, executor);
  CHECK_RET(selfContiguous != nullptr, ACLNN_ERR_INNER_NULLPTR);

  if (CheckType(self->GetDataType(), NEED_CAST_TO_FLOAT_DTYPE_LIST)) {
    selfContiguous = l0op::Cast(selfContiguous, op::DataType::DT_FLOAT, executor);
    CHECK_RET(selfContiguous != nullptr, ACLNN_ERR_INNER_NULLPTR);
  }

  auto tanOut = l0op::Tan(selfContiguous, executor);
  CHECK_RET(tanOut != nullptr, ACLNN_ERR_INNER_NULLPTR);

  auto castOut = l0op::Cast(tanOut, out->GetDataType(), executor);
  CHECK_RET(castOut != nullptr, ACLNN_ERR_INNER_NULLPTR);

  auto viewCopyOut = l0op::ViewCopy(castOut, out, executor);
  CHECK_RET(viewCopyOut != nullptr, ACLNN_ERR_INNER_NULLPTR);

  return ACLNN_SUCCESS;
}

static aclnnStatus CheckParams(const aclTensor *self, const aclTensor *out) {
  // 1. 检查参数是否为空指针
  CHECK_RET(CheckNotNull(self, out), ACLNN_ERR_PARAM_NULLPTR);

  // 2. 检查数据类型是否合法
  CHECK_RET(CheckDtypeValid(self), ACLNN_ERR_PARAM_INVALID);

  // 3. 检查shape是否支持
  CHECK_RET(CheckShape(self, out), ACLNN_ERR_PARAM_INVALID);

  // 4. 检查是否可以转成out数据类型
  CHECK_RET(CheckDtypeCanCast(self, out), ACLNN_ERR_PARAM_INVALID);
  if (self->GetStorageFormat() != Format::FORMAT_ND) {
      OP_LOGW("Only support ND format for tan operator.");
  }
  
  return ACLNN_SUCCESS;
}

aclnnStatus aclnnTanGetWorkspaceSize(const aclTensor *self, aclTensor *out,
                                     uint64_t *workspaceSize, aclOpExecutor **executor) {
  L2_DFX_PHASE_1(aclnnTan, DFX_IN(self), DFX_OUT(out));

  auto uniqueExecutor = CREATE_EXECUTOR();
  CHECK_RET(uniqueExecutor.get() != nullptr, ACLNN_ERR_INNER_CREATE_EXECUTOR);
  
  auto ret = CheckParams(self, out);
  CHECK_RET(ret == ACLNN_SUCCESS, ret);

  if (self->IsEmpty() || out->IsEmpty()) {
    *workspaceSize = 0;
    uniqueExecutor.ReleaseTo(executor);
    return ACLNN_SUCCESS;
  }

  ret = Compute(self, out, uniqueExecutor.get());
  CHECK_RET(ret == ACLNN_SUCCESS, ret);

  *workspaceSize = uniqueExecutor->GetWorkspaceSize();
  uniqueExecutor.ReleaseTo(executor);

  return ACLNN_SUCCESS;
}

static aclnnStatus CheckParamsInplace(const aclTensor *self, const aclTensor *out) {
  // 1. 检查参数是否为空指针
  OP_CHECK_NULL(self, return ACLNN_ERR_PARAM_NULLPTR);

  // 2. 检查数据类型是否合法
  CHECK_RET(CheckDtypeValid(self), ACLNN_ERR_PARAM_INVALID);

  // 3. 检查shape是否支持
  OP_CHECK_MAX_DIM(self, MAX_SUPPORT_DIMS_NUMS, return ACLNN_ERR_PARAM_INVALID);

  // 4. 检查是否可以转成out数据类型
  CHECK_RET(CheckDtypeCanCast(self, out), ACLNN_ERR_PARAM_INVALID);
  if (self->GetStorageFormat() != Format::FORMAT_ND) {
      OP_LOGW("Only support ND format for inplace tan operator.");
  }
  return ACLNN_SUCCESS;
}

aclnnStatus aclnnInplaceTanGetWorkspaceSize(const aclTensor *selfRef, uint64_t *workspaceSize,
                                            aclOpExecutor **executor) {
  L2_DFX_PHASE_1(aclnnInplaceTan, DFX_IN(selfRef), DFX_OUT(selfRef));

  auto uniqueExecutor = CREATE_EXECUTOR();
  CHECK_RET(uniqueExecutor.get() != nullptr, ACLNN_ERR_INNER_CREATE_EXECUTOR);
  
  auto ret = CheckParamsInplace(selfRef, selfRef);
  CHECK_RET(ret == ACLNN_SUCCESS, ret);

  if (selfRef->IsEmpty()) {
    *workspaceSize = 0;
    uniqueExecutor.ReleaseTo(executor);
    return ACLNN_SUCCESS;
  }

  auto out = const_cast<aclTensor*>(selfRef);
  ret = Compute(selfRef, out, uniqueExecutor.get());
  CHECK_RET(ret == ACLNN_SUCCESS, ret);

  *workspaceSize = uniqueExecutor->GetWorkspaceSize();
  uniqueExecutor.ReleaseTo(executor);

  return ACLNN_SUCCESS;
}

aclnnStatus aclnnTan(void *workspace, uint64_t workspaceSize, aclOpExecutor *executor, const aclrtStream stream) {
  L2_DFX_PHASE_2(aclnnTan);
  return CommonOpExecutorRun(workspace, workspaceSize, executor, stream);
}

aclnnStatus aclnnInplaceTan(void *workspace, uint64_t workspaceSize,
                            aclOpExecutor *executor, const aclrtStream stream) {
  L2_DFX_PHASE_2(aclnnInplaceTan);
  return CommonOpExecutorRun(workspace, workspaceSize, executor, stream);
}

#ifdef __cplusplus
}
#endif
