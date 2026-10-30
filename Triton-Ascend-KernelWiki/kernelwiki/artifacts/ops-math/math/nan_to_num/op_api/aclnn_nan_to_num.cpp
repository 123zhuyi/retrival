/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include "aclnn_nan_to_num.h"
#include "nan_to_num.h"
#include "aclnn_kernels/contiguous.h"
#include "aclnn_kernels/common/op_error_check.h"
#include "opdev/op_dfx.h"
#include "opdev/make_op_executor.h"
#include "opdev/platform.h"
#include "op_api/level2_base.h"
#include "op_api/aclnn_check.h"

using namespace op;

#ifdef __cplusplus
extern "C" {
#endif

static const std::initializer_list<DataType> DTYPE_SUPPORT_LIST = {
    DataType::DT_FLOAT16, DataType::DT_FLOAT, DataType::DT_BF16, DataType::DT_INT8,
    DataType::DT_INT16, DataType::DT_INT32, DataType::DT_INT64, DataType::DT_UINT8,
    DataType::DT_BOOL};  // to support: DT_DOUBLE

static inline bool CheckDtypeValid(const aclTensor* self, const aclTensor* out) {
  // ascend910不支持inf和nan
  if (GetCurrentPlatformInfo().GetSocVersion() != SocVersion::ASCEND910B &&
      GetCurrentPlatformInfo().GetSocVersion() != SocVersion::ASCEND910_93 &&
      !IsRegBase()) {
    OP_LOGE(ACLNN_ERR_PARAM_INVALID, "NanToNum is not supported on this device.");
    return false;
  }

  // self和out数据类型必须一样
  OP_CHECK_DTYPE_NOT_MATCH(self, out->GetDataType(), return false);

  // 检查self的数据类型是否在支持列表内
  OP_CHECK_DTYPE_NOT_SUPPORT(self, DTYPE_SUPPORT_LIST, return false);

  return true;
}

static inline aclnnStatus CheckParamsNanToNum(const aclTensor* self, const aclTensor* out) {
  // 1. 检查参数是否为空指针
  CHECK_RET(CheckNotNull2Tensor(self, out), ACLNN_ERR_PARAM_NULLPTR);

  // 2. 检查输入的数据类型是否在API支持的数据类型范围之内，需要根据api定义校验
  CHECK_RET(CheckDtypeValid(self, out), ACLNN_ERR_PARAM_INVALID);

  // 3. 检查shape是否满足约束
  CHECK_RET(CheckSameShape1In1Out(self, out), ACLNN_ERR_PARAM_INVALID);

  return ACLNN_SUCCESS;
}

static void CheckFormat(const aclTensor* self) {
    ge::Format selfStorageFormat = self->GetStorageFormat();
    if (selfStorageFormat == ge::Format::FORMAT_FRACTAL_NZ) {
        OP_LOGW("aclnnNanToNum doesn't support format NZ.");
    }
}
 
static aclnnStatus ExecNanToNumGetWorkspaceSize(const aclTensor *self, float nan, float posinf, float neginf,
                                                aclTensor *out, uint64_t *workspaceSize, aclOpExecutor **executor) {
  // 固定写法，创建OpExecutor
  auto uniqueExecutor = CREATE_EXECUTOR();
  CHECK_RET(uniqueExecutor.get() != nullptr, ACLNN_ERR_INNER_CREATE_EXECUTOR);

  // 固定写法，参数检查
  auto ret = CheckParamsNanToNum(self, out);
  CHECK_RET(ret == ACLNN_SUCCESS, ret);
  
  // 检查self的format是否支持
  CheckFormat(self);
  
  // 空Tensor处理
  if (self->IsEmpty()) {
    *workspaceSize = 0;
    uniqueExecutor.ReleaseTo(executor);
    return ACLNN_SUCCESS;
  }

  // self如果非连续，需要转连续
  auto selfContiguous = l0op::Contiguous(self, uniqueExecutor.get());
  CHECK_RET(selfContiguous != nullptr, ACLNN_ERR_INNER_NULLPTR);

  // 如果self的dtype为整形，则返回self的值。
  if (IsIntegralType(DataType(self->GetDataType()), true)) {
    auto viewCopyResult = l0op::ViewCopy(selfContiguous, out, uniqueExecutor.get());
    CHECK_RET(viewCopyResult != nullptr, ACLNN_ERR_INNER_NULLPTR);
    *workspaceSize = uniqueExecutor->GetWorkspaceSize();
    uniqueExecutor.ReleaseTo(executor);
    return ACLNN_SUCCESS;
  }

  // 调用l0算子NanToNum进行计算
  auto outResult = l0op::NanToNum(selfContiguous, nan, posinf, neginf, uniqueExecutor.get());
  CHECK_RET(outResult != nullptr, ACLNN_ERR_INNER_NULLPTR);

  // 如果出参out是非连续Tensor，需要把计算完的连续Tensor转非连续
  auto viewCopyResult = l0op::ViewCopy(outResult, out, uniqueExecutor.get());
  CHECK_RET(viewCopyResult != nullptr, ACLNN_ERR_INNER_NULLPTR);

  // 固定写法，获取计算过程中需要使用的workspace大小
  *workspaceSize = uniqueExecutor->GetWorkspaceSize();
  uniqueExecutor.ReleaseTo(executor);
  return ACLNN_SUCCESS;
}

aclnnStatus aclnnNanToNumGetWorkspaceSize(const aclTensor *self, float nan, float posinf, float neginf,
                                          aclTensor *out, uint64_t *workspaceSize, aclOpExecutor **executor)
{
    L2_DFX_PHASE_1(aclnnNanToNum, DFX_IN(self, nan, posinf, neginf), DFX_OUT(out));
    return ExecNanToNumGetWorkspaceSize(self, nan, posinf, neginf, out, workspaceSize, executor);
}

aclnnStatus aclnnInplaceNanToNumGetWorkspaceSize(aclTensor *selfRef, float nan, float posinf, float neginf,
                                                 uint64_t *workspaceSize, aclOpExecutor **executor)
{
    L2_DFX_PHASE_1(aclnnInplaceNanToNum, DFX_IN(selfRef, nan, posinf, neginf), DFX_OUT(selfRef));

    return ExecNanToNumGetWorkspaceSize(selfRef, nan, posinf, neginf, selfRef, workspaceSize, executor);
}

aclnnStatus aclnnNanToNum(void* workspace, uint64_t workspaceSize, aclOpExecutor* executor, aclrtStream stream) {
  L2_DFX_PHASE_2(aclnnNanToNum);

  return CommonOpExecutorRun(workspace, workspaceSize, executor, stream);
}

aclnnStatus aclnnInplaceNanToNum(void *workspace, uint64_t workspaceSize, aclOpExecutor *executor, aclrtStream stream) {
  // 固定写法，调用框架能力，完成计算
  L2_DFX_PHASE_2(aclnnInplaceNanToNum);
  return CommonOpExecutorRun(workspace, workspaceSize, executor, stream);
}

#ifdef __cplusplus
}
#endif
