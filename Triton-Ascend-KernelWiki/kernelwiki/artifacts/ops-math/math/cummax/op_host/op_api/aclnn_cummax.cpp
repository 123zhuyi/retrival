/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include "aclnn_cummax.h"
#include "cummax.h"
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
#include "op_api/level2_base.h"

using namespace op;
#ifdef __cplusplus
extern "C" {
#endif

static const std::initializer_list<op::DataType> ASCEND910_SELF_DTYPE_SUPPORT_LIST = {
    op::DataType::DT_FLOAT, op::DataType::DT_FLOAT16, op::DataType::DT_DOUBLE,
    op::DataType::DT_INT8,  op::DataType::DT_INT16,   op::DataType::DT_INT32,
    op::DataType::DT_INT64, op::DataType::DT_UINT8,   op::DataType::DT_BOOL};

static const std::initializer_list<op::DataType> ASCEND910B_SELF_DTYPE_SUPPORT_LIST = {
    op::DataType::DT_FLOAT, op::DataType::DT_FLOAT16, op::DataType::DT_DOUBLE,
    op::DataType::DT_INT8,  op::DataType::DT_INT16,   op::DataType::DT_INT32,
    op::DataType::DT_INT64, op::DataType::DT_UINT8,   op::DataType::DT_BOOL,
    op::DataType::DT_BF16};

static const std::initializer_list<op::DataType> INDICES_DTYPE_SUPPORT_LIST = {op::DataType::DT_INT32,
                                                                               op::DataType::DT_INT64};

static bool CheckDtypeValid(const aclTensor* self, const aclTensor* valuesOut, const aclTensor* indicesOut) {
  bool isAscend910BSocVersion = (GetCurrentPlatformInfo().GetSocVersion() >= SocVersion::ASCEND910B &&
                                   GetCurrentPlatformInfo().GetSocVersion() <= SocVersion::ASCEND910E);
  const std::initializer_list<op::DataType> selfDtypeSupportList =
      isAscend910BSocVersion ? ASCEND910B_SELF_DTYPE_SUPPORT_LIST : ASCEND910_SELF_DTYPE_SUPPORT_LIST;
  OP_CHECK_DTYPE_NOT_SUPPORT(self, selfDtypeSupportList, return false);
  OP_CHECK_DTYPE_NOT_SUPPORT(valuesOut, selfDtypeSupportList, return false);
  OP_CHECK_DTYPE_NOT_SUPPORT(indicesOut, INDICES_DTYPE_SUPPORT_LIST, return false);

  return true;
}

static bool CheckDimValid(const aclTensor* self, const int64_t dim) {
  const int64_t dimNum = self->GetViewShape().GetDimNum();
  int64_t minimum = dimNum * (-1);
  int64_t maximum = dimNum - 1;
  if (dimNum == 0) {
    OP_LOGE(ACLNN_ERR_PARAM_INVALID, "dimension specified as %ld but tensor has not dimensions.", dim);
    return false;
  }
  if (dim < minimum || dim > maximum) {
    OP_LOGE(ACLNN_ERR_PARAM_INVALID, "dim must be within the range of [%ld, %ld], but it is %ld.", minimum, maximum,
            dim);
    return false;
  }
  return true;
}

static aclnnStatus CheckParamsCummax(const aclTensor* self, const int64_t dim, const aclTensor* valuesOut,
                               const aclTensor* indicesOut) {
  // 1. 检查参数是否为空指针
  CHECK_RET(CheckNotNull3Tensor(self, valuesOut, indicesOut), ACLNN_ERR_PARAM_NULLPTR);

  // 2. 检查dim 是否合法
  CHECK_RET(CheckDimValid(self, dim), ACLNN_ERR_PARAM_INVALID);

  // 3. 检查输入的数据类型是否在API支持的数据类型范围之内，需要根据api定义校验
  CHECK_RET(CheckDtypeValid(self, valuesOut, indicesOut), ACLNN_ERR_PARAM_INVALID);

  // 4. 检查shape是否支持
  CHECK_RET(CheckShapeCumMinMax(self, valuesOut, indicesOut), ACLNN_ERR_PARAM_INVALID);

  return ACLNN_SUCCESS;
}

aclnnStatus aclnnCummaxGetWorkspaceSize(const aclTensor* self, int64_t dim, aclTensor* valuesOut, aclTensor* indicesOut,
                                        uint64_t* workspaceSize, aclOpExecutor** executor) {
  OP_CHECK_COMM_INPUT(workspaceSize, executor);
  
  L2_DFX_PHASE_1(aclnnCummax, DFX_IN(self, dim), DFX_OUT(valuesOut, indicesOut));

  auto ret = CheckParamsCummax(self, dim, valuesOut, indicesOut);
  CHECK_RET(ret == ACLNN_SUCCESS, ret);

  auto uniqueExecutor = CREATE_EXECUTOR();
  CHECK_RET(uniqueExecutor.get() != nullptr, ACLNN_ERR_INNER_CREATE_EXECUTOR);

  if (self->IsEmpty()) {
    *workspaceSize = 0;
    uniqueExecutor.ReleaseTo(executor);
    return ACLNN_SUCCESS;
  }

  auto selfContiguous = l0op::Contiguous(self, uniqueExecutor.get());
  CHECK_RET(selfContiguous != nullptr, ACLNN_ERR_INNER_NULLPTR);
  if (selfContiguous->GetDataType() == DataType::DT_BOOL) {
    selfContiguous = l0op::Cast(selfContiguous, DataType::DT_UINT8, uniqueExecutor.get());
    CHECK_RET(selfContiguous != nullptr, ACLNN_ERR_INNER_NULLPTR);
  }

  std::tuple<aclTensor*, aclTensor*> cummaxResult;
  if (indicesOut->GetDataType() == DataType::DT_INT64) {
    cummaxResult = l0op::CummaxOutInt64(selfContiguous, dim, uniqueExecutor.get());
  } else {
    cummaxResult = l0op::CummaxOutInt32(selfContiguous, dim, uniqueExecutor.get());
  }
  const aclTensor* valuesResult = std::get<0>(cummaxResult);
  CHECK_RET(valuesResult != nullptr, ACLNN_ERR_INNER_NULLPTR);
  const aclTensor* indicesResult = std::get<1>(cummaxResult);
  CHECK_RET(indicesResult != nullptr, ACLNN_ERR_INNER_NULLPTR);

  auto valuesCastResult = l0op::Cast(valuesResult, valuesOut->GetDataType(), uniqueExecutor.get());
  CHECK_RET(valuesCastResult != nullptr, ACLNN_ERR_INNER_NULLPTR);
  auto valuesViewCopyResult = l0op::ViewCopy(valuesCastResult, valuesOut, uniqueExecutor.get());
  CHECK_RET(valuesViewCopyResult != nullptr, ACLNN_ERR_INNER_NULLPTR);

  auto indicesViewCopyResult = l0op::ViewCopy(indicesResult, indicesOut, uniqueExecutor.get());
  CHECK_RET(indicesViewCopyResult != nullptr, ACLNN_ERR_INNER_NULLPTR);

  *workspaceSize = uniqueExecutor->GetWorkspaceSize();
  uniqueExecutor.ReleaseTo(executor);

  return ACLNN_SUCCESS;
}

aclnnStatus aclnnCummax(void* workspace, uint64_t workspaceSize, aclOpExecutor* executor, aclrtStream stream) {
  L2_DFX_PHASE_2(aclnnCummax);
  return CommonOpExecutorRun(workspace, workspaceSize, executor, stream);
}

#ifdef __cplusplus
}
#endif
