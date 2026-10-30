/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License")
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */
#include "aclnn_reciprocal.h"

#include "op_api/op_api_def.h"
#include "aclnn_kernels/common/op_error_check.h"
#include "opdev/common_types.h"
#include "opdev/data_type_utils.h"
#include "opdev/format_utils.h"
#include "opdev/op_dfx.h"
#include "opdev/op_executor.h"
#include "opdev/op_log.h"
#include "opdev/platform.h"
#include "opdev/shape_utils.h"
#include "opdev/tensor_view_utils.h"

#include "aclnn_kernels/cast.h"
#include "aclnn_kernels/contiguous.h"
#include "reciprocal.h"

using namespace op;
#ifdef __cplusplus
extern "C" {
#endif

/* Reciprocal 算子的完整计算流程如下:
 *                     self
 *                      |
 *            Contiguous(workspace_0)
 *                      |
 *               Cast(workspace_1)
 *                      |
 *            Reciprocal(workspace_2)
 *                      |
 *               Cast(workspace_3)
 *                      |
 *                   ViewCopy
 *                      |
 *                    result
 */

// 根据API定义，需要列出所能支持的所有dtype
static const std::initializer_list<op::DataType> INPUT_DTYPE_SUPPORT_LIST = {
    op::DataType::DT_FLOAT,      op::DataType::DT_FLOAT16,    op::DataType::DT_DOUBLE, op::DataType::DT_INT8,
    op::DataType::DT_INT16,      op::DataType::DT_INT32,      op::DataType::DT_INT64,  op::DataType::DT_BOOL,
    op::DataType::DT_COMPLEX64,  op::DataType::DT_COMPLEX128, op::DataType::DT_UINT8,  op::DataType::DT_BF16};

static const std::initializer_list<op::DataType> OUTPUT_DTYPE_SUPPORT_LIST = {
    op::DataType::DT_FLOAT,      op::DataType::DT_FLOAT16,    op::DataType::DT_DOUBLE, op::DataType::DT_COMPLEX64,
    op::DataType::DT_COMPLEX128, op::DataType::DT_BF16};

static const std::initializer_list<DataType> ASCEND910_DTYPE_SELFREF_LIST = {
    op::DataType::DT_FLOAT,      op::DataType::DT_FLOAT16,    op::DataType::DT_DOUBLE, op::DataType::DT_COMPLEX64,
    op::DataType::DT_COMPLEX128};

static inline bool CheckNotNull(const aclTensor *self, const aclTensor *out) {
  OP_CHECK_NULL(self, return false);
  OP_CHECK_NULL(out, return false);
  return true;
}

static inline bool CheckSocVersionIsSupportBf16(void)
{
  return GetCurrentPlatformInfo().GetSocVersion() >= SocVersion::ASCEND910B &&
         GetCurrentPlatformInfo().GetSocVersion() <= SocVersion::ASCEND910E;
}

static bool CheckDtypeValid(const aclTensor *self, const aclTensor *out) {
  // 检查self的数据类型是否在reciprocal算子的输入支持列表内
  OP_CHECK_DTYPE_NOT_SUPPORT(self, INPUT_DTYPE_SUPPORT_LIST, return false);

  // 检查out的数据类型是否在reciprocal算子的输出支持列表内
  OP_CHECK_DTYPE_NOT_SUPPORT(out, OUTPUT_DTYPE_SUPPORT_LIST, return false);

  // 检查self和out的数据类型，如果不符合芯片要求，则不允许BF16类型作为输入或者输出
  bool bf16Flag = CheckSocVersionIsSupportBf16();
  auto socVersion = GetCurrentPlatformInfo().GetSocVersion();
  if (!bf16Flag && self->GetDataType() == op::DataType::DT_BF16) {
      OP_LOGE(ACLNN_ERR_PARAM_INVALID, "Self dtype %s is unsupported by the current SOC version [%s].",
          op::ToString(self->GetDataType()).GetString(), op::ToString(socVersion).GetString());
      return false;
  }
  if (!bf16Flag && out->GetDataType() == op::DataType::DT_BF16) {
      OP_LOGE(ACLNN_ERR_PARAM_INVALID, "Out dtype %s is unsupported by the current SOC version [%s].",
          op::ToString(out->GetDataType()).GetString(), op::ToString(socVersion).GetString());
      return false;
  }

  return true;
}

static const std::initializer_list<DataType>& GetSelfRefDtypeList() {
  if (GetCurrentPlatformInfo().GetSocVersion() >= SocVersion::ASCEND910B &&
      GetCurrentPlatformInfo().GetSocVersion() <= SocVersion::ASCEND910E) {
    return OUTPUT_DTYPE_SUPPORT_LIST;
  } else {
    return ASCEND910_DTYPE_SELFREF_LIST;
  }
}

static bool CheckInplaceDtypeValid(const aclTensor *selfRef) {
  auto inplaceSupportList = GetSelfRefDtypeList();
  // 检查selfRef的数据类型是否在inplace reciprocal算子的支持列表内
  OP_CHECK_DTYPE_NOT_SUPPORT(selfRef, inplaceSupportList, return false);

  return true;
}

static bool CheckFormat(const aclTensor *self, const aclTensor *out) {
  // 需要根据算子实际情况添加校验
  if (self->GetStorageFormat() != out->GetStorageFormat()) {
    OP_LOGE(ACLNN_ERR_PARAM_INVALID, "Format of input and output should be equal, self [%s], out [%s].",
            op::ToString(self->GetStorageFormat()).GetString(), op::ToString(out->GetStorageFormat()).GetString());
    return false;
  }

  return true;
}

static inline bool CheckShape(const aclTensor *self, const aclTensor *out) {
  OP_CHECK_MAX_DIM(self, MAX_SUPPORT_DIMS_NUMS, return false);
  OP_CHECK_MAX_DIM(out, MAX_SUPPORT_DIMS_NUMS, return false);
  // 输入和输出的shape必须一致
  OP_CHECK_SHAPE_NOT_EQUAL(out, self, return false);

  return true;
}

static aclnnStatus CheckParams(const aclTensor *self, const aclTensor *out) {
  // 错误码等DFX方案细化后刷新，错误日志在check接口内打印
  // 1. 检查参数是否为空指针
  CHECK_RET(CheckNotNull(self, out), ACLNN_ERR_PARAM_NULLPTR);

  // 2. 检查输入的数据类型是否在API支持的数据类型范围之内，需要根据api定义校验
  CHECK_RET(CheckDtypeValid(self, out), ACLNN_ERR_PARAM_INVALID);

  // 3. 检查数据格式是否支持
  CHECK_RET(CheckFormat(self, out), ACLNN_ERR_PARAM_INVALID);

  // 4. 检查输入shape是否超过8维
  CHECK_RET(CheckShape(self, out), ACLNN_ERR_PARAM_INVALID);

  return ACLNN_SUCCESS;
}

static aclnnStatus CheckInplaceParams(const aclTensor *selfRef) {
  OP_CHECK_NULL(selfRef, return ACLNN_ERR_PARAM_NULLPTR);

  // 检查selfRef的数据类型是否在inplace reciprocal算子的支持列表内
  CHECK_RET(CheckInplaceDtypeValid(selfRef), ACLNN_ERR_PARAM_INVALID);
  return ACLNN_SUCCESS;
}

aclnnStatus aclnnReciprocalGetWorkspaceSize(const aclTensor *self, aclTensor *out,
                                            uint64_t *workspaceSize, aclOpExecutor **executor) {
  L2_DFX_PHASE_1(aclnnReciprocal, DFX_IN(self), DFX_OUT(out));
  // 固定写法，创建OpExecutor
  auto uniqueExecutor = CREATE_EXECUTOR();
  CHECK_RET(uniqueExecutor.get() != nullptr, ACLNN_ERR_INNER_CREATE_EXECUTOR);

  // 固定写法，参数检查
  auto ret = CheckParams(self, out);
  CHECK_RET(ret == ACLNN_SUCCESS, ret);

  // reciprocal算子的空tensor在kernel中支持
  if (self->IsEmpty()) {
    *workspaceSize = 0;
    uniqueExecutor.ReleaseTo(executor);
    return ACLNN_SUCCESS;
  }

  // 固定写法，将输入self转换成连续的tensor
  auto selfContiguous = l0op::Contiguous(self, uniqueExecutor.get());
  CHECK_RET(selfContiguous != nullptr, ACLNN_ERR_INNER_NULLPTR);

  // 若self的Dtype不在output支持的dtype范围内，转化为FLOAT类型
  auto selfCast = selfContiguous;
  if (!CheckType(selfContiguous->GetDataType(), OUTPUT_DTYPE_SUPPORT_LIST)) {
    selfCast = l0op::Cast(selfContiguous, op::DataType::DT_FLOAT, uniqueExecutor.get());
  }
  CHECK_RET(selfCast != nullptr, ACLNN_ERR_INNER_NULLPTR);

  // 进行Reciprocal计算
  auto reciprocalOpOut = l0op::Reciprocal(selfCast, uniqueExecutor.get());
  CHECK_RET(reciprocalOpOut != nullptr, ACLNN_ERR_INNER_NULLPTR);

  // 固定写法，将计算结果转换成输出out的数据类型
  auto castOut = l0op::Cast(reciprocalOpOut, out->GetDataType(), uniqueExecutor.get());
  CHECK_RET(castOut != nullptr, ACLNN_ERR_INNER_NULLPTR);

  // 固定写法，将计算结果拷贝到输出out上，out可能是非连续的tensor
  auto viewCopyResult = l0op::ViewCopy(castOut, out, uniqueExecutor.get());
  CHECK_RET(viewCopyResult != nullptr, ACLNN_ERR_INNER_NULLPTR);

  // 固定写法，获取计算过程中需要使用的workspace大小
  *workspaceSize = uniqueExecutor->GetWorkspaceSize();
  uniqueExecutor.ReleaseTo(executor);  // 需要把 uniqueExecutor持有executor转移给executor
  return ACLNN_SUCCESS;
}

aclnnStatus aclnnInplaceReciprocalGetWorkspaceSize(const aclTensor *selfRef,
                                                   uint64_t *workspaceSize, aclOpExecutor **executor) {
  auto out = const_cast<aclTensor *>(selfRef);
  auto ret = CheckInplaceParams(selfRef);
  CHECK_RET(ret == ACLNN_SUCCESS, ret);
  return aclnnReciprocalGetWorkspaceSize(selfRef, out, workspaceSize, executor);
}

aclnnStatus aclnnReciprocal(void *workspace, uint64_t workspaceSize,
                            aclOpExecutor *executor, aclrtStream stream) {
  L2_DFX_PHASE_2(aclnnReciprocal);
  // 固定写法，调用框架能力，完成计算
  return CommonOpExecutorRun(workspace, workspaceSize, executor, stream);
}

aclnnStatus aclnnInplaceReciprocal(void *workspace, uint64_t workspaceSize,
                                   aclOpExecutor *executor, aclrtStream stream) {
  L2_DFX_PHASE_2(aclnnInplaceReciprocal);
  // 固定写法，调用框架能力，完成计算
  return CommonOpExecutorRun(workspace, workspaceSize, executor, stream);
}

#ifdef __cplusplus
}
#endif
