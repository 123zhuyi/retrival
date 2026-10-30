/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */
#include "aclnn_tanh.h"
#include "aclnn_kernels/contiguous.h"
#include "tanh.h"
#include "aclnn_kernels/cast.h"
#include "aclnn/aclnn_base.h"
#include "opdev/common_types.h"
#include "opdev/data_type_utils.h"
#include "opdev/format_utils.h"
#include "opdev/op_dfx.h"
#include "opdev/op_executor.h"
#include "opdev/op_log.h"
#include "opdev/shape_utils.h"
#include "opdev/tensor_view_utils.h"
#include "opdev/platform.h"
#include "aclnn_kernels/common/op_error_check.h"
#include "op_api/op_api_def.h"
#include "op_api/aclnn_check.h"

using namespace op;
#ifdef __cplusplus
extern "C" {
#endif

static const std::initializer_list<op::DataType> ASCEND910_DTYPE_SUPPORT_LIST = {
    op::DataType::DT_FLOAT, op::DataType::DT_FLOAT16, op::DataType::DT_BOOL,
    op::DataType::DT_UINT8, op::DataType::DT_INT8, op::DataType::DT_INT16,
    op::DataType::DT_INT32, op::DataType::DT_INT64
    };

static const std::initializer_list<op::DataType> ASCEND910B_DTYPE_SUPPORT_LIST = {
    op::DataType::DT_FLOAT, op::DataType::DT_FLOAT16, op::DataType::DT_BOOL,
    op::DataType::DT_UINT8, op::DataType::DT_INT8, op::DataType::DT_INT16,
    op::DataType::DT_INT32, op::DataType::DT_INT64, op::DataType::DT_BF16
    };

static const std::initializer_list<op::DataType> ASCEND910_DTYPE_OUT_LIST = {
    op::DataType::DT_FLOAT,      op::DataType::DT_FLOAT16};

static const std::initializer_list<op::DataType> ASCEND910B_DTYPE_OUT_LIST = {
    op::DataType::DT_FLOAT,      op::DataType::DT_FLOAT16,    op::DataType::DT_BF16};

static const std::initializer_list<op::DataType> DTYPE_CAST_LIST = {
    op::DataType::DT_BOOL, op::DataType::DT_UINT8, op::DataType::DT_INT8,
    op::DataType::DT_INT16, op::DataType::DT_INT32, op::DataType::DT_INT64
    };

static const std::initializer_list<DataType>& GetOutDtypeSupportList() {
  if (GetCurrentPlatformInfo().GetSocVersion() == SocVersion::ASCEND910B ||
      GetCurrentPlatformInfo().GetSocVersion() == SocVersion::ASCEND910_93 || 
      IsRegBase()) {
    return ASCEND910B_DTYPE_OUT_LIST;
  } else {
    return ASCEND910_DTYPE_OUT_LIST;
  }
}

static bool CheckNotNull(const aclTensor *self, const aclTensor *out) {
  OP_CHECK_NULL(self, return false);
  OP_CHECK_NULL(out, return false);
  return true;
}

static bool CheckDtypeValid(const aclTensor *self, const aclTensor *out) {
  // 检查self的数据类型是否在tanh算子的支持列表内
  bool isAscend910BSocVersion = (GetCurrentPlatformInfo().GetSocVersion() >= SocVersion::ASCEND910B &&
                                 GetCurrentPlatformInfo().GetSocVersion() <= SocVersion::ASCEND910E);
  const std::initializer_list<op::DataType> CURRENT_DTYPE_SUPPORT_LIST =
    isAscend910BSocVersion ? ASCEND910B_DTYPE_SUPPORT_LIST : ASCEND910_DTYPE_SUPPORT_LIST;
  OP_CHECK_DTYPE_NOT_SUPPORT(self, CURRENT_DTYPE_SUPPORT_LIST, return false);

  // 检查out的数据类型是否在tanh算子的支持列表内
  OP_CHECK_DTYPE_NOT_SUPPORT(out, CURRENT_DTYPE_SUPPORT_LIST, return false);

  auto castType = (IsIntegralType(self->GetDataType(), true)) ?
                   op::DataType::DT_FLOAT : self->GetDataType();
  OP_CHECK_RESULT_DTYPE_CAST_FAILED(castType, out->GetDataType(), return false);
  return true;
}

static bool CheckFormat(const aclTensor *self, const aclTensor *out) {
  if (self->GetViewFormat() != out->GetViewFormat()) {
    OP_LOGE(ACLNN_ERR_PARAM_INVALID, "Format of input and output should be equal, self [%s], out [%s].",
            op::ToString(self->GetViewFormat()).GetString(), op::ToString(out->GetViewFormat()).GetString());
    return false;
  }

  // 如果输入格式是私有格式，记录日志，直接报错
  if (op::IsPrivateFormat(self->GetViewFormat())) {
    OP_LOGE(ACLNN_ERR_PARAM_INVALID, "Format only support ND、NCHW、NHWC、HWCN、NDHWC、NCDHW, not support [%s].",
            ToString(self->GetViewFormat()).GetString());
    return false;
  }
  // 非资料中的格式，打印warning log
  if (self->GetStorageFormat() != Format::FORMAT_ND) {
    OP_LOGW("Only support ND format for tanh/inplaceTanh operator.");
  }
  return true;
}

static bool CheckShape(const aclTensor *self, const aclTensor *out) {
  OP_CHECK_MAX_DIM(self, MAX_SUPPORT_DIMS_NUMS, return false);
  OP_CHECK_SHAPE_NOT_EQUAL(self, out, return false);
  return true;
}

static aclnnStatus CheckParams(const aclTensor *self, const aclTensor *out) {
  // 1. 检查参数是否为空指针
  CHECK_RET(CheckNotNull(self, out), ACLNN_ERR_PARAM_NULLPTR);

  // 2. 检查输入的数据类型是否在API支持的数据类型范围之内，需要根据api定义校验
  CHECK_RET(CheckDtypeValid(self, out), ACLNN_ERR_PARAM_INVALID);

  // 3. 检查数据格式是否支持
  CHECK_RET(CheckFormat(self, out), ACLNN_ERR_PARAM_INVALID);

  // 4. 检查输入shape
  CHECK_RET(CheckShape(self, out), ACLNN_ERR_PARAM_INVALID);

  return ACLNN_SUCCESS;
}

static bool CheckInplaceDtypeValid(aclTensor *selfRef) {
    auto inplaceSupportList = GetOutDtypeSupportList();
    // 检查selfRef的数据类型是否在inplace tanh算子的支持列表内
    OP_CHECK_DTYPE_NOT_SUPPORT(selfRef, inplaceSupportList, return false);

    return true;
}

static aclnnStatus CheckInplaceParams(aclTensor *selfRef) {
    OP_CHECK_NULL(selfRef, return ACLNN_ERR_PARAM_NULLPTR);
    // 检查selfRef的数据类型是否在inplace tanh算子的支持列表内
    CHECK_RET(CheckInplaceDtypeValid(selfRef), ACLNN_ERR_PARAM_INVALID);

    return ACLNN_SUCCESS;
}

aclnnStatus ExecTanhGetWorkspaceSize(const aclTensor *self, aclTensor *out,
                                     uint64_t *workspaceSize, aclOpExecutor **executor) {
  // 固定写法，创建OpExecutor
  auto uniqueExecutor = CREATE_EXECUTOR();
  CHECK_RET(uniqueExecutor.get() != nullptr, ACLNN_ERR_INNER_CREATE_EXECUTOR);

  // 固定写法，参数检查
  auto ret = CheckParams(self, out);
  CHECK_RET(ret == ACLNN_SUCCESS, ret);
  
  // 空tensor支持
  if (self->IsEmpty() || out->IsEmpty()) {
    *workspaceSize = 0;
    uniqueExecutor.ReleaseTo(executor);
    return ACLNN_SUCCESS;
  }

  // 固定写法，将输入self转换成连续的tensor
  auto selfContiguous = l0op::Contiguous(self, uniqueExecutor.get());
  // BOOL类型转换成FLOAT32进行计算
  if (CheckType(self->GetDataType(), DTYPE_CAST_LIST)) {
    selfContiguous = l0op::Cast(selfContiguous, op::DataType::DT_FLOAT, uniqueExecutor.get());
  }
  CHECK_RET(selfContiguous != nullptr, ACLNN_ERR_INNER_NULLPTR);

  // 调用tanh进行计算
  auto tanhOut = l0op::Tanh(selfContiguous, uniqueExecutor.get());
  CHECK_RET(tanhOut != nullptr, ACLNN_ERR_INNER_NULLPTR);

  // 固定写法，将计算结果转换成输出out的数据类型
  auto castOut = l0op::Cast(tanhOut, out->GetDataType(), uniqueExecutor.get());
  CHECK_RET(castOut != nullptr, ACLNN_ERR_PARAM_NULLPTR);

  // 固定写法，将计算结果拷贝到输出out上，out可能是非连续的tensor
  auto viewCopyResult = l0op::ViewCopy(castOut, out, uniqueExecutor.get());
  CHECK_RET(viewCopyResult != nullptr, ACLNN_ERR_INNER_NULLPTR);

  *workspaceSize = uniqueExecutor->GetWorkspaceSize();
  uniqueExecutor.ReleaseTo(executor);

  return ACLNN_SUCCESS;
}

aclnnStatus aclnnTanhGetWorkspaceSize(const aclTensor *self,
                                      aclTensor *out,
                                      uint64_t *workspaceSize,
                                      aclOpExecutor **executor) {
  L2_DFX_PHASE_1(aclnnTanh, DFX_IN(self), DFX_OUT(out));
  return ExecTanhGetWorkspaceSize(self, out, workspaceSize, executor);
}

aclnnStatus aclnnTanh(void *workspace, uint64_t workspaceSize, aclOpExecutor *executor, const aclrtStream stream) {
  L2_DFX_PHASE_2(aclnnTanh);
  // 固定写法，调用框架能力，完成计算
  return CommonOpExecutorRun(workspace, workspaceSize, executor, stream);
}

aclnnStatus aclnnInplaceTanhGetWorkspaceSize(aclTensor *selfRef,
                                             uint64_t *workspaceSize,
                                             aclOpExecutor **executor) {
  L2_DFX_PHASE_1(aclnnInplaceTanh, DFX_IN(selfRef), DFX_OUT(selfRef));
  auto ret = CheckInplaceParams(selfRef);
  CHECK_RET(ret == ACLNN_SUCCESS, ret);
  auto out = const_cast<aclTensor*>(selfRef);
  
  return ExecTanhGetWorkspaceSize(selfRef, out, workspaceSize, executor);
}

aclnnStatus aclnnInplaceTanh(void *workspace, uint64_t workspaceSize,
                             aclOpExecutor *executor, const aclrtStream stream) {
  L2_DFX_PHASE_2(aclnnInplaceTanh);
  // 固定写法，调用框架能力，完成计算
  return CommonOpExecutorRun(workspace, workspaceSize, executor, stream);
}
#ifdef __cplusplus
}
#endif
