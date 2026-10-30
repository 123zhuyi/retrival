/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */
 
#include "aclnn_linalg_cholesky.h"
#include "cholesky.h"
#include "aclnn_kernels/contiguous.h"
#include "aclnn_kernels/cast.h"
#include "aclnn_kernels/transpose.h"
#include "aclnn_kernels/common/op_error_check.h"
#include "opdev/op_log.h"
#include "opdev/make_op_executor.h"
#include "opdev/op_executor.h"
#include "opdev/platform.h"
#include "opdev/op_dfx.h"
#include "opdev/tensor_view_utils.h"

using namespace op;
#ifdef __cplusplus
extern "C" {
#endif

const int64_t SECOND_LAST_DIM_OFFSET = 2;
const int64_t LAST_DIM_OFFSET = 1;
const int64_t MAX_LAST_DIM_SIZE = 8192;

static const std::initializer_list<op::DataType> ASCEND910B_DTYPE_DTYPE_SUPPORT_LIST = {
    op::DataType::DT_FLOAT, op::DataType::DT_BF16, op::DataType::DT_FLOAT16};

static const std::initializer_list<op::DataType> NULL_SUPPORT_LIST = {};

static const std::initializer_list<op::DataType>& GetDtypeSupportList() {
  if (op::GetCurrentPlatformInfo().GetCurNpuArch() == NpuArch::DAV_2201 ||
      op::GetCurrentPlatformInfo().GetCurNpuArch() == NpuArch::DAV_3510) {
    return ASCEND910B_DTYPE_DTYPE_SUPPORT_LIST;
  } else {
    return NULL_SUPPORT_LIST;
  }
}

static bool CheckNotNull(const aclTensor* self, aclTensor* out) {
  OP_CHECK_NULL(self, return false);
  OP_CHECK_NULL(out, return false);
  return true;
}

static bool CheckDtypeValid(const aclTensor *self, const aclTensor *out) {
  auto supportList = GetDtypeSupportList();

  //检查self与out的数据类型是否一致
  OP_CHECK_DTYPE_NOT_MATCH(out, self->GetDataType(), return false);
  // 检查self的数据类型是否支持，out和self数据类型一致，不需要额外检查
  OP_CHECK_DTYPE_NOT_SUPPORT(self, supportList, return false);

  return true;
}

static bool CheckFormat(const aclTensor *self, const aclTensor *out) {
  // 输入输出的格式需要一致
  if (self->GetStorageFormat() != out->GetStorageFormat()) {
    OP_LOGE(ACLNN_ERR_PARAM_INVALID, "Format of input and output should be equal. self [%s], out [%s].",
            ToString(self->GetStorageShape()).GetString(), ToString(out->GetStorageShape()).GetString());
    return false;
  }

  // self格式不能是私有格式
  if (IsPrivateFormat(self->GetStorageFormat())) {
    OP_LOGE(ACLNN_ERR_PARAM_INVALID, "Format only support ND.");
    return false;
  }

  return true;
}

static bool CheckShape(const aclTensor *self, const aclTensor *out) {
  // 维度不能超过8
  OP_CHECK_MAX_DIM(self, ACLNN_MAX_SHAPE_RANK, return false);

  // 维度最少为2维
  OP_CHECK_MIN_DIM(self, 2, return false);

  // self和out的shape必须一致
  OP_CHECK_SHAPE_NOT_EQUAL(self, out, return false); 

  return true;
}

static aclnnStatus CheckParams(const aclTensor *self, aclTensor *out) {
  // 1. 检查参数是否为空指针
  CHECK_RET(CheckNotNull(self, out), ACLNN_ERR_PARAM_NULLPTR);

  // 2. 检查输入的数据类型是否在API支持的数据类型范围内
  CHECK_RET(CheckDtypeValid(self, out), ACLNN_ERR_PARAM_INVALID);

  // 3. 检查数据格式是否支持
  CHECK_RET(CheckFormat(self, out), ACLNN_ERR_PARAM_INVALID);

  // 4. 检查shape是否满足约束
  CHECK_RET(CheckShape(self, out), ACLNN_ERR_PARAM_INVALID);

  return ACLNN_SUCCESS;
}

aclnnStatus aclnnLinalgCholeskyGetWorkspaceSize(const aclTensor *self, bool upper, aclTensor *out,
                                     uint64_t *workspaceSize, aclOpExecutor **executor) {
  L2_DFX_PHASE_1(aclnnLinalgCholesky, DFX_IN(self, upper), DFX_OUT(out));

  // 固定写法，创建OpExecutor
  auto uniqueExecutor = CREATE_EXECUTOR();
  CHECK_RET(uniqueExecutor.get() != nullptr, ACLNN_ERR_INNER_CREATE_EXECUTOR);

  // 固定写法，参数检查
  auto ret = CheckParams(self, out);
  CHECK_RET(ret == ACLNN_SUCCESS, ret);

  if (self->IsEmpty()) {
      // 根据实际支持情况补充
      *workspaceSize = 0;
      uniqueExecutor.ReleaseTo(executor);
      return ACLNN_SUCCESS;
  }

  // self最后两维必须为相同
  auto dims = static_cast<int64_t>(self->GetViewShape().GetDimNum());
  int64_t last_dim_size = self->GetViewShape().GetDim(dims -1);
  int64_t second_last_dim_size = self->GetViewShape().GetDim(dims -2);
  if (last_dim_size != second_last_dim_size) {
    OP_LOGE(ACLNN_ERR_PARAM_INVALID, "self must be batches of square matrices, but they are [%ld] by [%ld] matrices", second_last_dim_size, last_dim_size);
    return ACLNN_ERR_PARAM_INVALID;
  }
  // 尾轴超过8192可能会超时报错，提前拦截
  if (last_dim_size > MAX_LAST_DIM_SIZE) {
    OP_LOGE(ACLNN_ERR_PARAM_INVALID, "The last dim must be less than or equal to 8192.");
    return ACLNN_ERR_PARAM_INVALID;
  }

  // self如果非连续，需要转换
  auto selfContiguous = l0op::Contiguous(self, uniqueExecutor.get());
  CHECK_RET(selfContiguous != nullptr, ACLNN_ERR_INNER_NULLPTR);

  // 输入类型是bf16时，需要转换成fp32
  if (self->GetDataType() == op::DataType::DT_BF16 || self->GetDataType() == op::DataType::DT_FLOAT16) {
    selfContiguous = l0op::Cast(selfContiguous, op::DataType::DT_FLOAT, uniqueExecutor.get());
    CHECK_RET(selfContiguous != nullptr, ACLNN_ERR_INNER_NULLPTR);
  }

  // 调用l0算子Cholesky进行计算
  auto choleskyResult = l0op::Cholesky(selfContiguous, upper, uniqueExecutor.get());
  CHECK_RET(choleskyResult != nullptr, ACLNN_ERR_INNER_NULLPTR);

  // 固定写法，将计算结果转换成输出out的数据类型
  choleskyResult = l0op::Cast(choleskyResult, out->GetDataType(), uniqueExecutor.get());
  CHECK_RET(choleskyResult != nullptr, ACLNN_ERR_INNER_NULLPTR);

  // 将结果拷贝到out
  auto viewCopyResult = l0op::ViewCopy(choleskyResult, out, uniqueExecutor.get());
  CHECK_RET(viewCopyResult != nullptr, ACLNN_ERR_INNER_NULLPTR);

  // 固定写法，获取计算过程中需要使用的workspace大小
  *workspaceSize = uniqueExecutor->GetWorkspaceSize();
  uniqueExecutor.ReleaseTo(executor);
  return ACLNN_SUCCESS;
}

aclnnStatus aclnnLinalgCholesky(void *workspace, uint64_t workspaceSize, aclOpExecutor *executor, const aclrtStream stream) {
  L2_DFX_PHASE_2(aclnnLinalgCholesky);
  // 固定写法，调用框架能力，完成计算
  return CommonOpExecutorRun(workspace, workspaceSize, executor, stream);
}

#ifdef __cplusplus
}
#endif