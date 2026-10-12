/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include "aclnn_masked_scale.h"
#include "masked_scale.h"
#include "aclnn_kernels/contiguous.h"
#include "opdev/op_log.h"
#include "opdev/op_dfx.h"
#include "opdev/common_types.h"
#include "opdev/data_type_utils.h"
#include "opdev/make_op_executor.h"
#include "opdev/platform.h"
#include "aclnn_kernels/common/op_error_check.h"
#include "op_api/op_api_def.h"
#include "op_api/aclnn_check.h"

using namespace op;
#ifdef __cplusplus
extern "C" {
#endif

static const std::initializer_list<DataType> NULL_SUPPORT_LIST = {};

static const std::initializer_list<DataType> REGBASE_X_DTYPE_SUPPORT_LIST = {
  DataType::DT_FLOAT, DataType::DT_FLOAT16, DataType::DT_BF16
};

static const std::initializer_list<DataType> REGBASE_MASK_DTYPE_SUPPORT_LIST = {
  DataType::DT_FLOAT, DataType::DT_FLOAT16, DataType::DT_UINT8, DataType::DT_INT8
};

static const std::initializer_list<DataType>& GetXDtypeSupportList() {
  if (IsRegBase()) {
    return REGBASE_X_DTYPE_SUPPORT_LIST;
  } else {
    return NULL_SUPPORT_LIST;
  }
}

static const std::initializer_list<DataType>& GetMaskDtypeSupportList() {
  if (IsRegBase()) {
    return REGBASE_MASK_DTYPE_SUPPORT_LIST;
  } else {
    return NULL_SUPPORT_LIST;
  }
}

static bool CheckNotNull(const aclTensor *self, const aclTensor *mask, const aclTensor *out) {
  OP_CHECK_NULL(self, return false);
  OP_CHECK_NULL(mask, return false);
  OP_CHECK_NULL(out, return false);

  return true;
}

static bool CheckDtypeValid(const aclTensor *self, const aclTensor *mask, const aclTensor *out) {
  // self和out数据类型必须一样
  OP_CHECK_DTYPE_NOT_SAME(self, out, return false);

  // 检查self和mask的数据类型是否在支持列表内，out类型随动检查
  OP_CHECK_DTYPE_NOT_SUPPORT(self, GetXDtypeSupportList(), return false);
  OP_CHECK_DTYPE_NOT_SUPPORT(mask, GetMaskDtypeSupportList(), return false);

  return true;
}

static bool CheckShape(const aclTensor *self, const aclTensor *mask, const aclTensor *out) {
  // self、mask和out的shape必须一致
  OP_CHECK_SHAPE_NOT_EQUAL(self, out, return false);
  OP_CHECK_SHAPE_NOT_EQUAL(mask, out, return false);
  // 校验self和mask shape维度是否小于8维
  OP_CHECK_MAX_DIM(self, MAX_SUPPORT_DIMS_NUMS, return false);
  OP_CHECK_MAX_DIM(mask, MAX_SUPPORT_DIMS_NUMS, return false);
  return true;
}

static bool CheckFormat(const aclTensor* self, const aclTensor* out) {
  // self、out的数据格式必须相同                     
  if (self->GetStorageFormat() != out->GetStorageFormat()) {
    OP_LOGE(ACLNN_ERR_PARAM_INVALID, "Format of self and out can't be different, self [%s], out [%s].",
        op::ToString(self->GetStorageFormat()).GetString(), op::ToString(out->GetStorageFormat()).GetString());
    return false;
  }
  return true;
}

static aclnnStatus CheckParams(const aclTensor *self, const aclTensor *mask, const aclTensor *out) {
  // 1. 检查参数是否为空指针
  CHECK_RET(CheckNotNull(self, mask, out), ACLNN_ERR_PARAM_NULLPTR);

  // 2. 检查输入的数据类型是否在API支持的数据类型范围之内，需要根据api定义校验
  CHECK_RET(CheckDtypeValid(self, mask, out), ACLNN_ERR_PARAM_INVALID);

  // 3. 检查shape
  CHECK_RET(CheckShape(self, mask, out), ACLNN_ERR_PARAM_INVALID);

  // 4. 检查gradOutput和self以及gradInput的format一致
  CHECK_RET(CheckFormat(self, out), ACLNN_ERR_PARAM_INVALID);

  return ACLNN_SUCCESS;
}

aclnnStatus aclnnMaskedScaleGetWorkspaceSize(const aclTensor* self, const aclTensor* mask, float scale,
                                             aclTensor* out, uint64_t *workspaceSize, aclOpExecutor **executor) {
  L2_DFX_PHASE_1(aclnnMaskedScale, DFX_IN(self, mask, scale), DFX_OUT(out));

    // 固定写法，创建OpExecutor
  auto uniqueExecutor = CREATE_EXECUTOR();
  CHECK_RET(uniqueExecutor.get() != nullptr, ACLNN_ERR_INNER_CREATE_EXECUTOR);

  // 固定写法，参数检查
  auto ret = CheckParams(self, mask, out);
  CHECK_RET(ret == ACLNN_SUCCESS, ret);

  // 空Tensor处理
  if (self->IsEmpty()) {
    *workspaceSize = 0;
    uniqueExecutor.ReleaseTo(executor);
    return ACLNN_SUCCESS;
  }

  // self如果非连续，需要转换
  auto selfContiguous = l0op::Contiguous(self, uniqueExecutor.get());
  CHECK_RET(selfContiguous != nullptr, ACLNN_ERR_INNER_NULLPTR);
  auto maskContiguous = l0op::Contiguous(mask, uniqueExecutor.get());
  CHECK_RET(maskContiguous != nullptr, ACLNN_ERR_INNER_NULLPTR);

  // 调用l0算子MaskedScale进行计算
  auto maskedScaleResult = l0op::MaskedScale(selfContiguous, maskContiguous, scale, uniqueExecutor.get());
  CHECK_RET(maskedScaleResult != nullptr, ACLNN_ERR_INNER_NULLPTR);

  // 如果出参out是非连续Tensor，需要把计算完的连续Tensor转非连续
  auto viewCopyResult = l0op::ViewCopy(maskedScaleResult, out, uniqueExecutor.get());
  CHECK_RET(viewCopyResult != nullptr, ACLNN_ERR_INNER_NULLPTR);

  // 固定写法，获取计算过程中需要使用的workspace大小
  *workspaceSize = uniqueExecutor->GetWorkspaceSize();
  uniqueExecutor.ReleaseTo(executor);
  return ACLNN_SUCCESS;
}

aclnnStatus aclnnMaskedScale(void *workspace, uint64_t workspaceSize, aclOpExecutor *executor, aclrtStream stream) {
  L2_DFX_PHASE_2(aclnnMaskedScale);

  return CommonOpExecutorRun(workspace, workspaceSize, executor, stream);
}

#ifdef __cplusplus
}
#endif