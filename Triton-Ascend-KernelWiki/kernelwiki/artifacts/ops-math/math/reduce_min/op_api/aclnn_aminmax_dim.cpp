/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include "aclnn_aminmax_dim.h"
#include "reduce_min.h"
#include "aclnn_kernels/contiguous.h"
#include "aclnn_kernels/cast.h"
#include "aclnn_kernels/common/op_error_check.h"
#include "opdev/common_types.h"
#include "opdev/data_type_utils.h"
#include "opdev/format_utils.h"
#include "opdev/op_dfx.h"
#include "opdev/op_executor.h"
#include "opdev/op_log.h"
#include "opdev/shape_utils.h"
#include "opdev/tensor_view_utils.h"
#include "opdev/platform.h"
#include "math/reduce_max/op_api/reduce_max.h"
#include "op_api/aclnn_check.h"

using namespace op;
#ifdef __cplusplus
extern "C" {
#endif

static const size_t MAX_DIM_LEN = 8;

// 根据API定义，需要列出所能支持的所有dtype
static const std::initializer_list<op::DataType> ASCEND910_DTYPE_DTYPE_SUPPORT_LIST = {
  op::DataType::DT_FLOAT, op::DataType::DT_FLOAT16, op::DataType::DT_UINT8, op::DataType::DT_INT32,   // AiCore
  op::DataType::DT_INT64, op::DataType::DT_INT16, op::DataType::DT_INT8, op::DataType::DT_DOUBLE,  // AiCpu
  op::DataType::DT_BOOL};

static const std::initializer_list<op::DataType> ASCEND910B_DTYPE_DTYPE_SUPPORT_LIST = {
  op::DataType::DT_FLOAT, op::DataType::DT_FLOAT16, op::DataType::DT_UINT8, op::DataType::DT_INT32,   // AiCore
  op::DataType::DT_INT64, op::DataType::DT_INT16, op::DataType::DT_INT8, op::DataType::DT_DOUBLE,  // AiCpu
  op::DataType::DT_BOOL, op::DataType::DT_BF16};

static bool CheckNotNull(const aclTensor *self, const aclTensor *minOut, const aclTensor *maxOut) {
  OP_CHECK_NULL(self, return false);
  OP_CHECK_NULL(minOut, return false);
  OP_CHECK_NULL(maxOut, return false);

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

static bool CheckDtypeValid(const aclTensor *self, const aclTensor *minOut, const aclTensor *maxOut) {
  const auto& supportList = GetDtypeSupportList();
  // 检查self的数据类型是否在算子的支持列表内
  OP_CHECK_DTYPE_NOT_SUPPORT(self, supportList, return false);
  OP_CHECK_DTYPE_NOT_SAME(self, minOut, return false);
  OP_CHECK_DTYPE_NOT_SAME(self, maxOut, return false);

  return true;
}

static bool CheckShape(const aclTensor *self, const aclTensor *minOut, const aclTensor *maxOut) {
  OP_CHECK_MAX_DIM(self, MAX_DIM_LEN, return false);
  OP_CHECK_MAX_DIM(minOut, MAX_DIM_LEN, return false);
  OP_CHECK_MAX_DIM(maxOut, MAX_DIM_LEN, return false);

  return true;
}

static bool CheckDimValid(const aclTensor *self, const int64_t dim) {
  auto selfViewShape = self->GetViewShape();
  auto selfDimNum = static_cast<int64_t>(selfViewShape.GetDimNum());
  // self为标量时，dim range [-1, 0]
  if (selfDimNum <= 0) {
      selfDimNum = 1;
  }
  if (dim >= selfDimNum || dim < (-selfDimNum)) {
      OP_LOGE(ACLNN_ERR_PARAM_INVALID, "Provided dim %ld must be in the range of [%ld, %ld].",
              dim, -selfDimNum, selfDimNum - 1);
      return false;
  }
  // dim指定的轴不能是空轴
  if (selfViewShape.GetDim((size_t)dim) == 0) {
    OP_LOGE(ACLNN_ERR_PARAM_INVALID, "Expected reduction dim %ld to have non-zero size, but check failed.", dim);
    return false;
  }
  return true;
}

static inline aclnnStatus CheckParams(const aclTensor *self, const int64_t dim,
                                      const aclTensor *minOut, const aclTensor *maxOut) {
  // 1. 检查参数是否为空指针
  CHECK_RET(CheckNotNull(self, minOut, maxOut), ACLNN_ERR_PARAM_NULLPTR);

  // 2. 检查输入的数据类型是否在API支持的数据类型范围之内，需要根据api定义校验
  CHECK_RET(CheckDtypeValid(self, minOut, maxOut), ACLNN_ERR_PARAM_INVALID);

  // 3. 检查tensor的维度
  CHECK_RET(CheckShape(self, minOut, maxOut), ACLNN_ERR_PARAM_INVALID);

  // 4. 检查dim是否在有效范围内
  CHECK_RET(CheckDimValid(self, dim), ACLNN_ERR_PARAM_INVALID);

  return ACLNN_SUCCESS;
}

aclnnStatus aclnnAminmaxDimGetWorkspaceSize(const aclTensor *self, const int64_t dim, bool keepDim,
                                            aclTensor *minOut, aclTensor *maxOut,
                                            uint64_t *workspaceSize, aclOpExecutor **executor) {
  L2_DFX_PHASE_1(aclnnAminmaxDim, DFX_IN(self, dim, keepDim), DFX_OUT(minOut, maxOut));
  // 固定写法，创建OpExecutor
  auto uniqueExecutor = CREATE_EXECUTOR();
  CHECK_RET(uniqueExecutor.get() != nullptr, ACLNN_ERR_INNER_CREATE_EXECUTOR);

  // 固定写法，参数检查
  auto ret = CheckParams(self, dim, minOut, maxOut);
  CHECK_RET(ret == ACLNN_SUCCESS, ret);

  // 空tensor处理
  if (self->IsEmpty()) {
    *workspaceSize = 0;
    uniqueExecutor.ReleaseTo(executor);
    return ACLNN_SUCCESS;
  }

  // 当输入tensor是0维时，直接将输入tensor作为最大值和最小值返回
  if (self->GetViewShape().GetDimNum() == 0) {
    auto viewCopyMinResult = l0op::ViewCopy(self, minOut, uniqueExecutor.get());
    CHECK_RET(viewCopyMinResult != nullptr, ACLNN_ERR_INNER_NULLPTR);

    auto viewCopyMaxResult = l0op::ViewCopy(self, maxOut, uniqueExecutor.get());
    CHECK_RET(viewCopyMaxResult != nullptr, ACLNN_ERR_INNER_NULLPTR);
    *workspaceSize = uniqueExecutor->GetWorkspaceSize();
    uniqueExecutor.ReleaseTo(executor);
    return ACLNN_SUCCESS;
  }

  // 固定写法，将输入self转换成连续的tensor
  auto selfContiguous = l0op::Contiguous(self, uniqueExecutor.get());
  CHECK_RET(selfContiguous != nullptr, ACLNN_ERR_INNER_NULLPTR);

  // 需要支持int64和bool类型
  auto selfCasted = selfContiguous;
  if (self->GetDataType() == op::DataType::DT_BOOL) {
    selfCasted = l0op::Cast(selfContiguous, op::DataType::DT_UINT8, uniqueExecutor.get());
    CHECK_RET(selfCasted != nullptr, ACLNN_ERR_PARAM_NULLPTR);
  }
  int64_t appendDim[1] = {dim};
  auto dimArray = uniqueExecutor.get()->AllocIntArray(appendDim, 1);

  // 进行ReduceMin计算
  auto minResult = l0op::ReduceMin(selfCasted, dimArray, keepDim, uniqueExecutor.get());
  CHECK_RET(minResult != nullptr, ACLNN_ERR_INNER_NULLPTR);
  CHECK_RET(CheckShapeAndScalarSame(minResult, minOut), ACLNN_ERR_PARAM_INVALID);

  // 进行ReduceMax计算
  auto maxResult = l0op::ReduceMax(selfCasted, dimArray, keepDim, true, uniqueExecutor.get());
  CHECK_RET(maxResult != nullptr, ACLNN_ERR_INNER_NULLPTR);
  CHECK_RET(CheckShapeAndScalarSame(maxResult, maxOut), ACLNN_ERR_PARAM_INVALID);

  // 固定写法，将计算结果转换成输出的数据类型
  auto castMinOut = l0op::Cast(minResult, minOut->GetDataType(), uniqueExecutor.get());
  CHECK_RET(castMinOut != nullptr, ACLNN_ERR_INNER_NULLPTR);

  // 固定写法，将计算结果转换成输出的数据类型
  auto castMaxOut = l0op::Cast(maxResult, maxOut->GetDataType(), uniqueExecutor.get());
  CHECK_RET(castMaxOut != nullptr, ACLNN_ERR_INNER_NULLPTR);

  // 固定写法，将计算结果拷贝到输出out上，out可能是非连续的tensor
  auto viewCopyMinOut = l0op::ViewCopy(castMinOut, minOut, uniqueExecutor.get());
  CHECK_RET(viewCopyMinOut != nullptr, ACLNN_ERR_INNER_NULLPTR);

  // 固定写法，将计算结果拷贝到输出out上，out可能是非连续的tensor
  auto viewCopyMaxOut = l0op::ViewCopy(castMaxOut, maxOut, uniqueExecutor.get());
  CHECK_RET(viewCopyMaxOut != nullptr, ACLNN_ERR_INNER_NULLPTR);

  // 固定写法，获取计算过程中需要使用的workspace大小
  *workspaceSize = uniqueExecutor->GetWorkspaceSize();
  uniqueExecutor.ReleaseTo(executor);
  return ACLNN_SUCCESS;
}

aclnnStatus aclnnAminmaxDim(void *workspace, uint64_t workspaceSize, aclOpExecutor *executor, aclrtStream stream) {
  L2_DFX_PHASE_2(aclnnAminmaxDim);
  // 固定写法，调用框架能力，完成计算
  return CommonOpExecutorRun(workspace, workspaceSize, executor, stream);
}

#ifdef __cplusplus
}
#endif
