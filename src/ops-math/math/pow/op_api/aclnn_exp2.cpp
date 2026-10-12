/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include "aclnn_exp2.h"
#include "pow.h"
#include "aclnn_kernels/cast.h"
#include "aclnn_kernels/contiguous.h"
#include "aclnn_kernels/common/op_error_check.h"
#include "opdev/common_types.h"
#include "opdev/data_type_utils.h"
#include "opdev/format_utils.h"
#include "opdev/op_dfx.h"
#include "opdev/op_executor.h"
#include "opdev/shape_utils.h"
#include "opdev/tensor_view_utils.h"
#include "opdev/platform.h"
#include "opdev/op_log.h"
#include "op_api/aclnn_check.h"

using namespace op;
#ifdef __cplusplus
extern "C" {
#endif

static constexpr size_t MAX_DIM_LEN = 8;
// EXP2计算以2为底的指数
static constexpr int64_t BASE_FOR_EXP2 = 2;

static const std::initializer_list<op::DataType> DTYPE_SUPPORT_LIST = {
  DataType::DT_FLOAT, DataType::DT_INT32, DataType::DT_INT64, DataType::DT_FLOAT16,
  DataType::DT_INT8,  DataType::DT_UINT8, DataType::DT_DOUBLE, DataType::DT_INT16,
  DataType::DT_BOOL};

static const std::initializer_list<op::DataType> ASCEND910B_DTYPE_SUPPORT_LIST = {
  DataType::DT_FLOAT, DataType::DT_INT32, DataType::DT_INT64, DataType::DT_FLOAT16,
  DataType::DT_INT8,  DataType::DT_UINT8, DataType::DT_DOUBLE, DataType::DT_INT16,
  DataType::DT_BOOL,  DataType::DT_BF16};

static const std::initializer_list<op::DataType> INPLACE_DTYPE_SUPPORT_LIST = {
  DataType::DT_FLOAT, DataType::DT_FLOAT16, DataType::DT_DOUBLE};

static const std::initializer_list<op::DataType> INPLACE_ASCEND910B_DTYPE_SUPPORT_LIST = {
  DataType::DT_FLOAT, DataType::DT_FLOAT16, DataType::DT_DOUBLE, DataType::DT_BF16};

static inline bool CheckNotNull(const aclTensor* self, const aclTensor* out) {
  OP_CHECK_NULL(self, return false);

  OP_CHECK_NULL(out, return false);

  return true;
}

static inline bool CheckDtypeValid(const aclTensor* self, const aclTensor* out) {
  auto socVersion = GetCurrentPlatformInfo().GetSocVersion();
  bool isASCEND910BorASCEND910_93 = (socVersion == SocVersion::ASCEND910B ||
                                     socVersion == SocVersion::ASCEND910_93 ||
                                     IsRegBase());
  if (isASCEND910BorASCEND910_93) {
      OP_CHECK_DTYPE_NOT_SUPPORT(self, ASCEND910B_DTYPE_SUPPORT_LIST, return false);
  } else {
      OP_CHECK_DTYPE_NOT_SUPPORT(self, DTYPE_SUPPORT_LIST, return false);
  }

  // 类型检查前self的INT类型需要先进行转换
  auto selfDataType = self->GetDataType();
  if (IsIntegralType(selfDataType, true)) {
      selfDataType = DataType::DT_FLOAT;
  }

  OP_CHECK_RESULT_DTYPE_CAST_FAILED(selfDataType, out->GetDataType(), return false);

  return true;
}

static inline bool CheckInplaceDtypeValid(const aclTensor* self, const aclTensor* out) {
  auto socVersion = GetCurrentPlatformInfo().GetSocVersion();
  bool isASCEND910BorASCEND910_93 = (socVersion == SocVersion::ASCEND910B ||
                                     socVersion == SocVersion::ASCEND910_93 ||
                                     IsRegBase());
  if (isASCEND910BorASCEND910_93) {
      OP_CHECK_DTYPE_NOT_SUPPORT(self, INPLACE_ASCEND910B_DTYPE_SUPPORT_LIST, return false);
  } else {
      OP_CHECK_DTYPE_NOT_SUPPORT(self, INPLACE_DTYPE_SUPPORT_LIST, return false);
  }

  // 类型检查前self的INT类型需要先进行转换
  auto selfDataType = self->GetDataType();
  if (IsIntegralType(selfDataType, true)) {
      selfDataType = DataType::DT_FLOAT;
  }

  OP_CHECK_RESULT_DTYPE_CAST_FAILED(selfDataType, out->GetDataType(), return false);

  return true;
}

static inline bool CheckShape(const aclTensor* self, const aclTensor* out) {
  // self和out的shape必须一致
  OP_CHECK_SHAPE_NOT_EQUAL(self, out, return false);

  // self的数据维度不能超过8
  OP_CHECK_MAX_DIM(self, MAX_DIM_LEN, return false);

  return true;
}

static aclnnStatus CheckParams(const aclTensor* self, const aclTensor* out) {
  // 1. 检查参数是否为空指针
  CHECK_RET(CheckNotNull(self, out), ACLNN_ERR_PARAM_NULLPTR);

  // 2. 检查输入的数据类型是否在API支持的数据类型范围之内
  CHECK_RET(CheckDtypeValid(self, out), ACLNN_ERR_PARAM_INVALID);

  // 3. 检查self和out的shape是否一致
  CHECK_RET(CheckShape(self, out), ACLNN_ERR_PARAM_INVALID);

  return ACLNN_SUCCESS;
}

static aclnnStatus CheckInplaceParams(const aclTensor* self, const aclTensor* out) {
  // 1. 检查参数是否为空指针
  CHECK_RET(CheckNotNull(self, out), ACLNN_ERR_PARAM_NULLPTR);

  // 2. 检查输入的数据类型是否在API支持的数据类型范围之内
  CHECK_RET(CheckInplaceDtypeValid(self, out), ACLNN_ERR_PARAM_INVALID);

  // 3. 检查self和out的shape是否一致
  CHECK_RET(CheckShape(self, out), ACLNN_ERR_PARAM_INVALID);

  return ACLNN_SUCCESS;
}

static aclnnStatus GetWorkspaceSizeCommon(const aclTensor* self, aclTensor* out, uint64_t* workspaceSize,
                                          aclOpExecutor** executor) {
  // 固定写法，创建OpExecutor
  auto uniqueExecutor = CREATE_EXECUTOR();
  CHECK_RET(uniqueExecutor.get() != nullptr, ACLNN_ERR_INNER_CREATE_EXECUTOR);

  // 固定写法，参数检查
  auto ret = CheckParams(self, out);
  CHECK_RET(ret == ACLNN_SUCCESS, ret);

  // 如果self是空tensor，则out也是空tensor，直接返回
  if (self->IsEmpty()) {
      *workspaceSize = 0;
      uniqueExecutor.ReleaseTo(executor);
      return ACLNN_SUCCESS;
  }

  // 固定写法，将输入self转换成连续的Tensor
  auto contiguousSelf = l0op::Contiguous(self, uniqueExecutor.get());
  CHECK_RET(contiguousSelf != nullptr, ACLNN_ERR_INNER_NULLPTR);

  aclScalar* baseValue = uniqueExecutor.get()->AllocScalar(BASE_FOR_EXP2);
  auto dstDataType = contiguousSelf->GetDataType();
  auto selfAfterCast = contiguousSelf;

  // 只有在Ascend910B才支持BF16，同时l0 Cast在Ascend910B时也支持BF16Cast成FLOAT
  // BOOL类型也需要转换成FLOAT类型处理
  if (dstDataType == DataType::DT_BF16 || IsIntegralType(dstDataType, true)) {
      dstDataType = DataType::DT_FLOAT;
      selfAfterCast = l0op::Cast(contiguousSelf, DataType::DT_FLOAT, uniqueExecutor.get());
      CHECK_RET(selfAfterCast != nullptr, ACLNN_ERR_INNER_NULLPTR);
  }

  auto baseValueTensor = uniqueExecutor.get()->ConvertToTensor(baseValue, dstDataType);
  CHECK_RET(baseValueTensor != nullptr, ACLNN_ERR_INNER_NULLPTR);

  auto exp2Out = l0op::Pow(baseValueTensor, selfAfterCast, uniqueExecutor.get());
  CHECK_RET(exp2Out != nullptr, ACLNN_ERR_INNER_NULLPTR);

  // 固定写法，将计算结果转换成输出out的数据类型
  auto castOut = l0op::Cast(exp2Out, out->GetDataType(), uniqueExecutor.get());
  CHECK_RET(castOut != nullptr, ACLNN_ERR_INNER_NULLPTR);

  // 固定写法，将计算结果拷贝到输出out上，out可能是非连续的tensor
  auto viewCopyResult = l0op::ViewCopy(castOut, out, uniqueExecutor.get());
  CHECK_RET(viewCopyResult != nullptr, ACLNN_ERR_INNER_NULLPTR);

  // 固定写法，获取计算过程中需要使用的workspace大小
  *workspaceSize = uniqueExecutor->GetWorkspaceSize();
  uniqueExecutor.ReleaseTo(executor);

  return ACLNN_SUCCESS;
}

aclnnStatus aclnnExp2GetWorkspaceSize(const aclTensor* self, aclTensor* out, uint64_t* workspaceSize,
                                      aclOpExecutor** executor) {
  L2_DFX_PHASE_1(aclnnExp2, DFX_IN(self), DFX_OUT(out));
  return GetWorkspaceSizeCommon(self, out, workspaceSize, executor);
}

aclnnStatus aclnnExp2(void* workspace, uint64_t workspaceSize, aclOpExecutor* executor, aclrtStream stream) {
    L2_DFX_PHASE_2(aclnnExp2);
  // 固定写法，调用框架能力，完成计算
  return CommonOpExecutorRun(workspace, workspaceSize, executor, stream);
}

aclnnStatus aclnnInplaceExp2GetWorkspaceSize(aclTensor* selfRef, uint64_t* workspaceSize, aclOpExecutor** executor) {
  L2_DFX_PHASE_1(aclnnInplaceExp2, DFX_IN(selfRef), DFX_OUT(selfRef));
  // 固定写法，创建OpExecutor
  auto uniqueExecutor = CREATE_EXECUTOR();
  CHECK_RET(uniqueExecutor.get() != nullptr, ACLNN_ERR_INNER_CREATE_EXECUTOR);

  // 固定写法，参数检查
  auto ret = CheckInplaceParams(selfRef, selfRef);
  CHECK_RET(ret == ACLNN_SUCCESS, ret);

  // 如果self是空tensor，则out也是空tensor，直接返回
  if (selfRef->IsEmpty()) {
      *workspaceSize = 0;
      uniqueExecutor.ReleaseTo(executor);
      return ACLNN_SUCCESS;
  }

  // 固定写法，将输入self转换成连续的Tensor
  auto contiguousSelf = l0op::Contiguous(selfRef, uniqueExecutor.get());
  CHECK_RET(contiguousSelf != nullptr, ACLNN_ERR_INNER_NULLPTR);

  aclScalar* baseValue = uniqueExecutor.get()->AllocScalar(BASE_FOR_EXP2);
  auto dstDataType = contiguousSelf->GetDataType();
  auto selfAfterCast = contiguousSelf;

  // 只有在Ascend910B才支持BF16，同时l0 Cast在Ascend910B时也支持BF16Cast成FLOAT
  // BOOL类型也需要转换成FLOAT类型处理
  if (dstDataType == DataType::DT_BF16 || IsIntegralType(dstDataType, true)) {
      dstDataType = DataType::DT_FLOAT;
      selfAfterCast = l0op::Cast(contiguousSelf, DataType::DT_FLOAT, uniqueExecutor.get());
      CHECK_RET(selfAfterCast != nullptr, ACLNN_ERR_INNER_NULLPTR);
  }

  auto baseValueTensor = uniqueExecutor.get()->ConvertToTensor(baseValue, dstDataType);
  CHECK_RET(baseValueTensor != nullptr, ACLNN_ERR_INNER_NULLPTR);

  auto exp2Out = l0op::Pow(baseValueTensor, selfAfterCast, uniqueExecutor.get());
  CHECK_RET(exp2Out != nullptr, ACLNN_ERR_INNER_NULLPTR);

  // 固定写法，将计算结果转换成输出out的数据类型
  auto castOut = l0op::Cast(exp2Out, selfRef->GetDataType(), uniqueExecutor.get());
  CHECK_RET(castOut != nullptr, ACLNN_ERR_INNER_NULLPTR);

  // 固定写法，将计算结果拷贝到输出out上，out可能是非连续的tensor
  auto viewCopyResult = l0op::ViewCopy(castOut, selfRef, uniqueExecutor.get());
  CHECK_RET(viewCopyResult != nullptr, ACLNN_ERR_INNER_NULLPTR);

  // 固定写法，获取计算过程中需要使用的workspace大小
  *workspaceSize = uniqueExecutor->GetWorkspaceSize();
  uniqueExecutor.ReleaseTo(executor);

  return ACLNN_SUCCESS;
}

aclnnStatus aclnnInplaceExp2(void* workspace, uint64_t workspaceSize, aclOpExecutor* executor, aclrtStream stream) {
  L2_DFX_PHASE_2(aclnnInplaceExp2);
  // 固定写法，调用框架能力，完成计算
  return CommonOpExecutorRun(workspace, workspaceSize, executor, stream);
}

#ifdef __cplusplus
}
#endif