/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include "aclnn_expm1.h"
#include "aclnn_kernels/contiguous.h"
#include "expm1.h"
#include "aclnn_kernels/cast.h"
#include "aclnn_kernels/common/op_error_check.h"
#include "opdev/common_types.h"
#include "opdev/data_type_utils.h"
#include "opdev/format_utils.h"
#include "opdev/make_op_executor.h"
#include "opdev/op_dfx.h"
#include "opdev/op_log.h"
#include "opdev/platform.h"
#include "op_api/aclnn_check.h"
#include "op_api/level2_base.h"

using namespace op;
#ifdef __cplusplus
extern "C" {
#endif

static const std::initializer_list<op::DataType> ASCEND910_DTYPE_DTYPE_SUPPORT_LIST = {
    op::DataType::DT_FLOAT, op::DataType::DT_FLOAT16, op::DataType::DT_INT64,
    op::DataType::DT_BOOL
};

static const std::initializer_list<op::DataType> ASCEND910B_DTYPE_DTYPE_SUPPORT_LIST = {
    op::DataType::DT_FLOAT, op::DataType::DT_FLOAT16, op::DataType::DT_INT64,
    op::DataType::DT_BOOL, op::DataType::DT_BF16
};

static const std::initializer_list<op::DataType> ASCEND910_DTYPE_SELFREF_LIST = {
    op::DataType::DT_FLOAT, op::DataType::DT_FLOAT16
};

static const std::initializer_list<op::DataType> ASCEND910B_DTYPE_SELFREF_LIST = {
    op::DataType::DT_FLOAT, op::DataType::DT_FLOAT16, op::DataType::DT_BF16
};

static const std::initializer_list<DataType> FLOAT_LIST = {DataType::DT_FLOAT16, DataType::DT_FLOAT, DataType::DT_BF16};

static bool CheckDtypeValid(const aclTensor *self, const aclTensor *out) {
  // self需要在support list内。
  auto supportList = GetDtypeSupportListV2(ASCEND910B_DTYPE_DTYPE_SUPPORT_LIST, ASCEND910_DTYPE_DTYPE_SUPPORT_LIST);
  OP_CHECK_DTYPE_NOT_SUPPORT(self, supportList, return false);

  // 检查expm1的计算结果能否转为out的类型。out为整型或者bool则直接返回false
  if (IsFloatingType(self->GetDataType())) {
    OP_CHECK_RESULT_DTYPE_CAST_FAILED(self->GetDataType(), out->GetDataType(), return false);
  } else {
    // self为整型或者bool，转换为float32进行计算，不影响精度。检查float32能否cast为out的dtype。
    OP_CHECK_RESULT_DTYPE_CAST_FAILED(op::DataType::DT_FLOAT, out->GetDataType(), return false);
  }

  return true;
}

static bool CheckInplaceDtypeValid(const aclTensor *selfRef) {
  auto inplaceSupportList = GetDtypeSupportListV2(ASCEND910B_DTYPE_SELFREF_LIST, ASCEND910_DTYPE_SELFREF_LIST);
  // 检查selfRef的数据类型是否在inplace Expm算子的支持列表内
  OP_CHECK_DTYPE_NOT_SUPPORT(selfRef, inplaceSupportList, return false);
  return true;
}

static aclnnStatus CheckParamsExpm1(const aclTensor *self, const aclTensor *out) {
  // 1. 检查参数是否为空指针
  CHECK_RET(CheckNotNull2Tensor(self, out), ACLNN_ERR_PARAM_NULLPTR);

  // 2. 检查self和out的维度匹配关系
  CHECK_RET(CheckSameShape1In1Out(self, out), ACLNN_ERR_PARAM_INVALID);

  // 3. 检查expm1计算的输出dtype能否cast为out的dtype
  CHECK_RET(CheckDtypeValid(self, out), ACLNN_ERR_PARAM_INVALID);

  return ACLNN_SUCCESS;
}

static aclnnStatus CheckInplaceParamsExpm1(aclTensor *selfRef) {
    OP_CHECK_NULL(selfRef, return ACLNN_ERR_PARAM_NULLPTR);

    // 检查selfRef的数据类型是否在inplace Expm1算子的支持列表内
    CHECK_RET(CheckInplaceDtypeValid(selfRef), ACLNN_ERR_PARAM_INVALID);
    return ACLNN_SUCCESS;
}

aclnnStatus aclnnExpm1Common(const aclTensor *self, aclTensor *out,
                             uint64_t *workspaceSize, aclOpExecutor **executor) {
  // 固定写法，创建OpExecutor
  auto uniqueExecutor = CREATE_EXECUTOR();
  CHECK_RET(uniqueExecutor.get() != nullptr, ACLNN_ERR_INNER_CREATE_EXECUTOR);

  // 固定写法，参数检查
  auto ret = CheckParamsExpm1(self, out);
  CHECK_RET(ret == ACLNN_SUCCESS, ret);

  // 空Tensor处理,此时self与out的shape一致，若self为空tensor，则out也为空，直接返回
  if (self->IsEmpty()) {
    *workspaceSize = 0;
    uniqueExecutor.ReleaseTo(executor);
    return ACLNN_SUCCESS;
  }
  if (self->GetStorageFormat() != Format::FORMAT_ND) {
      OP_LOGW("Format only support ND");
  }
  // self如果非empty，需要转连续
  auto selfContiguous = l0op::Contiguous(self, uniqueExecutor.get());
  CHECK_RET(selfContiguous != nullptr, ACLNN_ERR_INNER_NULLPTR);

  // self如果为整形，需要cast为float.self为complex，已在CheckDtypeValid里校验。
  auto selfCast = selfContiguous;
  if (!IsFloatingType(self->GetDataType())) {
    selfCast = l0op::Cast(selfContiguous, op::DataType::DT_FLOAT, uniqueExecutor.get());
    CHECK_RET(selfCast != nullptr, ACLNN_ERR_INNER_NULLPTR);
  }

  // 调用l0算子Expm1进行计算
  auto expm1Result = l0op::Expm1(selfCast, uniqueExecutor.get());
  CHECK_RET(expm1Result != nullptr, ACLNN_ERR_INNER_NULLPTR);

  // 将result的数据类型转换成out的数据类型，根据具体算子语义按需调用
  auto outCasted = l0op::Cast(expm1Result, out->GetDataType(), uniqueExecutor.get());
  CHECK_RET(outCasted != nullptr, ACLNN_ERR_INNER_NULLPTR);

  // 如果出参out是非连续Tensor，需要把计算完的连续Tensor转非连续
  auto viewCopyResult = l0op::ViewCopy(outCasted, out, uniqueExecutor.get());
  CHECK_RET(viewCopyResult != nullptr, ACLNN_ERR_INNER_NULLPTR);

  // 固定写法，获取计算过程中需要使用的workspace大小
  *workspaceSize = uniqueExecutor->GetWorkspaceSize();
  uniqueExecutor.ReleaseTo(executor);
  return ACLNN_SUCCESS;
}

aclnnStatus aclnnExpm1GetWorkspaceSize(const aclTensor *self, aclTensor *out,
                                       uint64_t *workspaceSize, aclOpExecutor **executor) {
  L2_DFX_PHASE_1(aclnnExpm1, DFX_IN(self), DFX_OUT(out));
  return aclnnExpm1Common(self, out, workspaceSize, executor);
}

aclnnStatus aclnnExpm1(void *workspace, uint64_t workspace_size, aclOpExecutor *executor, const aclrtStream stream) {
  L2_DFX_PHASE_2(aclnnExpm1);
  // 固定写法，调用框架能力，完成计算
  return CommonOpExecutorRun(workspace, workspace_size, executor, stream);
}

aclnnStatus aclnnInplaceExpm1GetWorkspaceSize(aclTensor *selfRef, uint64_t *workspaceSize,
                                              aclOpExecutor **executor) {
  L2_DFX_PHASE_1(aclnnInplaceExpm1, DFX_IN(selfRef), DFX_OUT(selfRef));
  auto ret = CheckInplaceParamsExpm1(selfRef);
  CHECK_RET(ret == ACLNN_SUCCESS, ret);
  return aclnnExpm1Common(selfRef, selfRef, workspaceSize, executor);
}

aclnnStatus aclnnInplaceExpm1(void *workspace, uint64_t workspace_size,
                              aclOpExecutor *executor, const aclrtStream stream) {
  L2_DFX_PHASE_2(aclnnInplaceExpm1);
  // 固定写法，调用框架能力，完成计算
  return CommonOpExecutorRun(workspace, workspace_size, executor, stream);
}

#ifdef __cplusplus
}
#endif
