/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include "aclnn_sqrt.h"
#include "aclnn_kernels/contiguous.h"
#include "aclnn_kernels/cast.h"
#include "sqrt.h"
#include "aclnn_kernels/common/op_error_check.h"
#include "aclnn/aclnn_base.h"
#include "opdev/common_types.h"
#include "opdev/data_type_utils.h"
#include "opdev/shape_utils.h"
#include "opdev/format_utils.h"
#include "opdev/op_dfx.h"
#include "opdev/op_executor.h"
#include "opdev/op_log.h"
#include "opdev/tensor_view_utils.h"
#include "opdev/platform.h"
#include "op_api/level2_base.h"

using namespace op;
#ifdef __cplusplus
extern "C" {
#endif

static const std::initializer_list<op::DataType> DTYPE_SUPPORT_910_LIST = {
  // FLOAT16、FLOAT32、FLOAT64、COMPLEX64、COMPLEX128
  op::DataType::DT_FLOAT, op::DataType::DT_FLOAT16, op::DataType::DT_DOUBLE, op::DataType::DT_COMPLEX64,
  op::DataType::DT_COMPLEX128, op::DataType::DT_INT32, op::DataType::DT_INT64, op::DataType::DT_INT16,
  op::DataType::DT_INT8, op::DataType::DT_BOOL, op::DataType::DT_UINT8
};

static const std::initializer_list<op::DataType> DTYPE_SUPPORT_910B_LIST = {
  // FLOAT16、FLOAT32、FLOAT64、COMPLEX64、COMPLEX128
  op::DataType::DT_FLOAT, op::DataType::DT_FLOAT16, op::DataType::DT_DOUBLE, op::DataType::DT_COMPLEX64,
  op::DataType::DT_COMPLEX128, op::DataType::DT_INT32, op::DataType::DT_INT64, op::DataType::DT_INT16,
  op::DataType::DT_INT8, op::DataType::DT_BOOL, op::DataType::DT_BF16, op::DataType::DT_UINT8
};

static const std::initializer_list<op::DataType> DTYPE_CAST_LIST = {
  op::DataType::DT_INT32, op::DataType::DT_INT64, op::DataType::DT_INT16, op::DataType::DT_INT8, op::DataType::DT_BOOL,
  op::DataType::DT_UINT8
};

static const std::initializer_list<op::DataType> ASCEND910_DTYPE_OUT_LIST = {
    op::DataType::DT_FLOAT,      op::DataType::DT_FLOAT16,    op::DataType::DT_DOUBLE, op::DataType::DT_COMPLEX64,
    op::DataType::DT_COMPLEX128};

static const std::initializer_list<op::DataType> ASCEND910B_DTYPE_OUT_LIST = {
    op::DataType::DT_FLOAT,      op::DataType::DT_FLOAT16,    op::DataType::DT_DOUBLE, op::DataType::DT_COMPLEX64,
    op::DataType::DT_COMPLEX128, op::DataType::DT_BF16};

static bool CheckInplaceDtypeValid(aclTensor *selfRef) {
    auto inplaceSupportList = GetDtypeSupportListV2(ASCEND910B_DTYPE_OUT_LIST, ASCEND910_DTYPE_OUT_LIST);
    // 检查selfRef的数据类型是否在inplace sqrt算子的支持列表内
    OP_CHECK_DTYPE_NOT_SUPPORT(selfRef, inplaceSupportList, return false);

    return true;
}

static bool CheckDtypeValid(const aclTensor *self, const aclTensor *out) {
  const std::initializer_list<op::DataType> dtypeSupportList = GetDtypeSupportListV3(DTYPE_SUPPORT_910B_LIST, DTYPE_SUPPORT_910_LIST);
  // 检查self的数据类型是否在sqrt算子的支持列表内
  OP_CHECK_DTYPE_NOT_SUPPORT(self, dtypeSupportList, return false);
  auto castType = (IsIntegralType(self->GetDataType(), true)) ?
                   op::DataType::DT_FLOAT : self->GetDataType();
  OP_CHECK_RESULT_DTYPE_CAST_FAILED(castType, out->GetDataType(), return false);
  return true;
}

static aclnnStatus CheckParamsSqrt(const aclTensor *self, const aclTensor *out) {
  // 1. 检查参数是否为空指针
  CHECK_RET(CheckNotNull2Tensor(self, out), ACLNN_ERR_PARAM_NULLPTR);

  // 2. 检查输入的数据类型是否在API支持的数据类型范围之内，需要根据api定义校验
  CHECK_RET(CheckDtypeValid(self, out), ACLNN_ERR_PARAM_INVALID);

  // 3. 检查数据格式是否支持
  CHECK_RET(CheckSameShape1In1Out(self, out), ACLNN_ERR_PARAM_INVALID);

  return ACLNN_SUCCESS;
}

static aclnnStatus CheckInplaceParamsSqrt(aclTensor *selfRef) {
    OP_CHECK_NULL(selfRef, return ACLNN_ERR_PARAM_NULLPTR);
    // 检查selfRef的数据类型是否在inplace sqrt算子的支持列表内
    CHECK_RET(CheckInplaceDtypeValid(selfRef), ACLNN_ERR_PARAM_INVALID);

    return ACLNN_SUCCESS;
}

static void CheckFormat(const aclTensor* self) {
    ge::Format selfStorageFormat = self->GetStorageFormat();
    if (selfStorageFormat == ge::Format::FORMAT_FRACTAL_NZ) {
        OP_LOGW("aclnnSqrt/aclnnInplaceSqrt doesn't support format NZ.");
    }
}

aclnnStatus aclnnSqrtGetWorkspaceSize(const aclTensor *self, aclTensor *out,
                                      uint64_t *workspaceSize, aclOpExecutor **executor) {
  OP_CHECK_COMM_INPUT(workspaceSize, executor);

  L2_DFX_PHASE_1(aclnnSqrt, DFX_IN(self), DFX_OUT(out));

  // 固定写法，创建OpExecutor
  auto uniqueExecutor = CREATE_EXECUTOR();
  CHECK_RET(uniqueExecutor.get() != nullptr, ACLNN_ERR_INNER_CREATE_EXECUTOR);

  // 固定写法，参数检查
  auto ret = CheckParamsSqrt(self, out);
  CHECK_RET(ret == ACLNN_SUCCESS, ret);

  // 检查self的format是否支持
  CheckFormat(self);
  
  // 空tensor支持
  if (self->IsEmpty()) {
    *workspaceSize = 0;
    uniqueExecutor.ReleaseTo(executor);
    return ACLNN_SUCCESS;
  }

  // 固定写法，将输入self转换成连续的tensor
  auto selfContiguous = l0op::Contiguous(self, uniqueExecutor.get());
  CHECK_RET(selfContiguous != nullptr, ACLNN_ERR_INNER_NULLPTR);

  // cast算子INT64转float会出错，故转double
  auto castDtype = self->GetDataType() == op::DataType::DT_INT64 ? op::DataType::DT_DOUBLE : op::DataType::DT_FLOAT;

  // 调用cast转换int等类型
  auto selfCast = (CheckType(selfContiguous->GetDataType(), DTYPE_CAST_LIST)) ?
                  l0op::Cast(selfContiguous, castDtype, uniqueExecutor.get()) :
                  selfContiguous;
  CHECK_RET(selfCast != nullptr, ACLNN_ERR_INNER_NULLPTR);

  // 调用sqrt进行计算
  auto sqrtOpOut = l0op::Sqrt(selfCast, uniqueExecutor.get());
  CHECK_RET(sqrtOpOut != nullptr, ACLNN_ERR_INNER_NULLPTR);

  // 固定写法，将计算结果转换成输出out的数据类型
  auto castOut = l0op::Cast(sqrtOpOut, out->GetDataType(), uniqueExecutor.get());
  CHECK_RET(castOut != nullptr, ACLNN_ERR_INNER_NULLPTR);

  // 固定写法，将计算结果拷贝到输出out上，out可能是非连续的tensor
  auto viewCopyResult = l0op::ViewCopy(castOut, out, uniqueExecutor.get());
  CHECK_RET(viewCopyResult != nullptr, ACLNN_ERR_INNER_NULLPTR);

  // 固定写法，获取计算过程中需要使用的workspace大小
  *workspaceSize = uniqueExecutor->GetWorkspaceSize();
  uniqueExecutor.ReleaseTo(executor);
  return ACLNN_SUCCESS;
}

aclnnStatus aclnnSqrt(void *workspace, uint64_t workspaceSize, aclOpExecutor *executor, const aclrtStream stream) {
  L2_DFX_PHASE_2(aclnnSqrt);
  // 固定写法，调用框架能力，完成计算
  return CommonOpExecutorRun(workspace, workspaceSize, executor, stream);
}
aclnnStatus aclnnInplaceSqrtGetWorkspaceSize(aclTensor *self,
                                             uint64_t *workspaceSize,
                                             aclOpExecutor **executor) {
  auto out = const_cast<aclTensor*>(self);
  auto ret = CheckInplaceParamsSqrt(self);
  CHECK_RET(ret == ACLNN_SUCCESS, ret);
  return aclnnSqrtGetWorkspaceSize(self, out, workspaceSize, executor);
}

aclnnStatus aclnnInplaceSqrt(void *workspace, uint64_t workspaceSize,  aclOpExecutor *executor,
                             const aclrtStream stream) {
  L2_DFX_PHASE_2(aclnnInplaceSqrt);
  // 固定写法，调用框架能力，完成计算
  return CommonOpExecutorRun(workspace, workspaceSize, executor, stream);
}

#ifdef __cplusplus
}
#endif
