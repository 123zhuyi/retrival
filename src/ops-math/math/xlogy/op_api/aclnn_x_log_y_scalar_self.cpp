/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include "aclnn_x_log_y_scalar_self.h"
#include "x_log_y.h"
#include "aclnn_kernels/cast.h"
#include "aclnn_kernels/contiguous.h"
#include "aclnn_kernels/common/op_error_check.h"
#include "op_api/op_api_def.h"
#include "opdev/common_types.h"
#include "opdev/data_type_utils.h"
#include "opdev/format_utils.h"
#include "opdev/op_dfx.h"
#include "opdev/op_executor.h"
#include "opdev/op_log.h"
#include "opdev/shape_utils.h"
#include "opdev/tensor_view_utils.h"
#include "opdev/platform.h"
#include "op_api/aclnn_check.h"

using namespace op;
#ifdef __cplusplus
extern "C" {
#endif

// 根据API定义，需要列出所能支持的所有dtype
static const std::initializer_list<op::DataType> ASCEND910_DTYPE_DTYPE_SUPPORT_LIST = {
    op::DataType::DT_FLOAT, op::DataType::DT_INT32, op::DataType::DT_INT64, op::DataType::DT_FLOAT16,
    op::DataType::DT_INT8,  op::DataType::DT_UINT8, op::DataType::DT_BOOL, op::DataType::DT_INT16};

static const std::initializer_list<op::DataType> ASCEND910B_DTYPE_DTYPE_SUPPORT_LIST = {
    op::DataType::DT_FLOAT, op::DataType::DT_INT32, op::DataType::DT_INT64, op::DataType::DT_FLOAT16,
    op::DataType::DT_INT8,  op::DataType::DT_UINT8, op::DataType::DT_BOOL, op::DataType::DT_INT16,
    op::DataType::DT_BF16};

static const std::initializer_list<op::DataType> SCALAR_ASCEND910_DTYPE_DTYPE_SUPPORT_LIST = {
    op::DataType::DT_FLOAT, op::DataType::DT_INT32, op::DataType::DT_INT64, op::DataType::DT_FLOAT16,
    op::DataType::DT_INT8,  op::DataType::DT_UINT8,
    op::DataType::DT_BOOL, op::DataType::DT_INT16, op::DataType::DT_DOUBLE};

static const std::initializer_list<op::DataType> SCALAR_ASCEND910B_DTYPE_DTYPE_SUPPORT_LIST = {
    op::DataType::DT_FLOAT, op::DataType::DT_INT32, op::DataType::DT_INT64, op::DataType::DT_FLOAT16,
    op::DataType::DT_INT8,  op::DataType::DT_UINT8,
    op::DataType::DT_BOOL, op::DataType::DT_INT16, op::DataType::DT_DOUBLE, op::DataType::DT_BF16};

static bool CheckNotNull(const aclScalar *self, const aclTensor *other, const aclTensor *out) {
  OP_CHECK_NULL(self, return false);
  OP_CHECK_NULL(other, return false);
  OP_CHECK_NULL(out, return false);
  return true;
}

static const std::initializer_list<DataType>& GetDtypeSupportList() {
  if (GetCurrentPlatformInfo().GetSocVersion() == SocVersion::ASCEND910B ||
      GetCurrentPlatformInfo().GetSocVersion() == SocVersion::ASCEND910_93 ||
      IsRegBase()) {
    return ASCEND910B_DTYPE_DTYPE_SUPPORT_LIST;
  } else {
    return ASCEND910_DTYPE_DTYPE_SUPPORT_LIST;
  }
}

static const std::initializer_list<DataType>& ScalarGetDtypeSupportList() {
  if (GetCurrentPlatformInfo().GetSocVersion() == SocVersion::ASCEND910B ||
      GetCurrentPlatformInfo().GetSocVersion() == SocVersion::ASCEND910_93 ||
      IsRegBase()) {
    return SCALAR_ASCEND910B_DTYPE_DTYPE_SUPPORT_LIST;
  } else {
    return SCALAR_ASCEND910_DTYPE_DTYPE_SUPPORT_LIST;
  }
}

static bool CheckDtypeValid(const aclScalar *self, const aclTensor *other) {
  const auto& supportList = GetDtypeSupportList();
  const auto& scalarSupportList = ScalarGetDtypeSupportList();
  // 检查self的数据类型是否在xlogy算子的支持列表内
  OP_CHECK_DTYPE_NOT_SUPPORT(self, scalarSupportList, return false);

  // 检查other的数据类型是否在xlogy算子的支持列表内
  OP_CHECK_DTYPE_NOT_SUPPORT(other, supportList, return false);
  return true;
}

static bool CheckPromoteType(const aclScalar *self, const aclTensor *other, const aclTensor *out,
                             op::DataType &promoteType) {
  promoteType = op::PromoteType(self->GetDataType(), other->GetDataType());
  OP_CHECK(promoteType != DataType::DT_UNDEFINED,
           OP_LOGE(ACLNN_ERR_PARAM_INVALID, "Self dtype %s and other dtype %s can not promote dtype.",
                   op::ToString(self->GetDataType()).GetString(), op::ToString(other->GetDataType()).GetString()),
           return false);
  promoteType = IsFloatingType(promoteType) ? promoteType : op::DataType::DT_FLOAT;
  promoteType = (promoteType == op::DataType::DT_DOUBLE) ? op::DataType::DT_FLOAT : promoteType;
  // 检查推导后的数据类型能否转换为输出的数据类型
  OP_CHECK_RESULT_DTYPE_CAST_FAILED(promoteType, out->GetDataType(), return false);
  return true;
}

static bool CheckShape(const aclTensor *other, const aclTensor *out) {
  OP_CHECK_MAX_DIM(other, MAX_SUPPORT_DIMS_NUMS, return false);
  OP_CHECK_SHAPE_NOT_EQUAL(out, other, return false);
  return true;
}

static aclnnStatus CheckParams(const aclScalar *self, const aclTensor *other, const aclTensor *out,
                               op::DataType &promoteType) {
  // 1. 检查参数是否为空指针
  CHECK_RET(CheckNotNull(self, other, out), ACLNN_ERR_PARAM_NULLPTR);

  // 2. 检查输入的数据类型是否在API支持的数据类型范围之内，需要根据api定义校验
  CHECK_RET(CheckDtypeValid(self, other), ACLNN_ERR_PARAM_INVALID);

  // 3. 检查self和other能否做数据类型推导以及推导的数据类型能否转换为输出数据类型
  CHECK_RET(CheckPromoteType(self, other, out, promoteType), ACLNN_ERR_PARAM_INVALID);

  // 4. 检查Shape
  CHECK_RET(CheckShape(other, out), ACLNN_ERR_PARAM_INVALID);

  return ACLNN_SUCCESS;
}

aclnnStatus aclnnXLogYScalarSelfGetWorkspaceSize(const aclScalar *self, const aclTensor *other,
                                                 aclTensor *out, uint64_t *workspaceSize,
                                                 aclOpExecutor **executor) {
  L2_DFX_PHASE_1(aclnnXLogYScalarSelf, DFX_IN(self, other), DFX_OUT(out));

  // 固定写法，创建OpExecutor
  auto uniqueExecutor = CREATE_EXECUTOR();
  CHECK_RET(uniqueExecutor.get() != nullptr, ACLNN_ERR_INNER_CREATE_EXECUTOR);

  // 固定写法，参数检查
  op::DataType promoteType;
  auto ret = CheckParams(self, other, out, promoteType);
  CHECK_RET(ret == ACLNN_SUCCESS, ret);

  // xlogy算子的空tensor在kernel中支持，对标竞品根据算子实际情况补充
  if (other->IsEmpty()) {
    // 根据实际支持情况补充
    *workspaceSize = 0;
    uniqueExecutor.ReleaseTo(executor);
    return ACLNN_SUCCESS;
  }

  auto selfCasted = uniqueExecutor.get()->ConvertToTensor(self, promoteType);
  CHECK_RET(selfCasted != nullptr, ACLNN_ERR_INNER_NULLPTR);

  // 固定写法，将输入other转换成连续的tensor
  auto otherContiguous = l0op::Contiguous(other, uniqueExecutor.get());
  CHECK_RET(otherContiguous != nullptr, ACLNN_ERR_INNER_NULLPTR);

  // 将输入other的数据类型转换成隐式数据类型，根据具体算子语义按需调用
  auto otherCasted = l0op::Cast(otherContiguous, promoteType, uniqueExecutor.get());
  CHECK_RET(otherCasted != nullptr, ACLNN_ERR_INNER_NULLPTR);

  // 调用XLogY算子kernel
  auto OpOut = l0op::XLogY(selfCasted, otherCasted, uniqueExecutor.get());
  CHECK_RET(OpOut != nullptr, ACLNN_ERR_INNER_NULLPTR);

  // 固定写法，将计算结果转换成输出out的数据类型
  auto castOut = l0op::Cast(OpOut, out->GetDataType(), uniqueExecutor.get());
  CHECK_RET(castOut != nullptr, ACLNN_ERR_INNER_NULLPTR);

  // 固定写法，将计算结果拷贝到输出out上，out可能是非连续的tensor
  auto viewCopyResult = l0op::ViewCopy(castOut, out, uniqueExecutor.get());
  CHECK_RET(viewCopyResult != nullptr, ACLNN_ERR_INNER_NULLPTR);

  // 固定写法，获取计算过程中需要使用的workspace大小
  *workspaceSize = uniqueExecutor->GetWorkspaceSize();
  uniqueExecutor.ReleaseTo(executor);
  return ACLNN_SUCCESS;
}

aclnnStatus aclnnXLogYScalarSelf(void *workspace, uint64_t workspaceSize,
                                 aclOpExecutor *executor, aclrtStream stream) {
  L2_DFX_PHASE_2(aclnnXLogYScalarSelf);
  // 固定写法，调用框架能力，完成计算
  return CommonOpExecutorRun(workspace, workspaceSize, executor, stream);
}

#ifdef __cplusplus
}
#endif
