/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include "aclnn_aminmax.h"
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
  op::DataType::DT_FLOAT, op::DataType::DT_FLOAT16, op::DataType::DT_INT32, op::DataType::DT_UINT8,
  op::DataType::DT_BOOL,
  // 如下数据类型会走AICPU
  op::DataType::DT_INT8, op::DataType::DT_INT16, op::DataType::DT_INT64, op::DataType::DT_DOUBLE};

static const std::initializer_list<op::DataType> ASCEND910B_DTYPE_DTYPE_SUPPORT_LIST = {
  op::DataType::DT_FLOAT, op::DataType::DT_FLOAT16, op::DataType::DT_INT32, op::DataType::DT_UINT8,
  op::DataType::DT_BOOL, op::DataType::DT_BF16,
  // 如下数据类型会走AICPU
  op::DataType::DT_INT8, op::DataType::DT_INT16, op::DataType::DT_INT64, op::DataType::DT_DOUBLE};

static inline bool CheckNotNull(const aclTensor *self, const aclIntArray *dim,
                                const aclTensor *minOut, const aclTensor *maxOut) {
  OP_CHECK_NULL(self, return false);
  OP_CHECK_NULL(dim, return false);
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

static inline bool CheckDtypeValid(const aclTensor *self, const aclTensor *minOut, const aclTensor *maxOut) {
  const auto& supportList = GetDtypeSupportList();
  // 检查self的数据类型是否在算子的支持列表内
  OP_CHECK_DTYPE_NOT_SUPPORT(self, supportList, return false);
  // 检查minOut,maxOut的数据类型是否与self一致
  OP_CHECK_DTYPE_NOT_MATCH(minOut, self->GetDataType(), return false);
  OP_CHECK_DTYPE_NOT_MATCH(maxOut, self->GetDataType(), return false);
  return true;
}

static inline bool CheckShape(const aclTensor *self, const aclTensor *minOut, const aclTensor *maxOut) {
  // 检查self,minOut,maxOut的维度是否大于8维
  OP_CHECK_MAX_DIM(self, MAX_DIM_LEN, return false);
  OP_CHECK_MAX_DIM(minOut, MAX_DIM_LEN, return false);
  OP_CHECK_MAX_DIM(maxOut, MAX_DIM_LEN, return false);
  return true;
}

static bool CheckDimValid(const aclTensor *self, const aclIntArray *dim) {
  auto selfViewShape = self->GetViewShape();
  auto selfDimNum = static_cast<int64_t>(selfViewShape.GetDimNum());
  // 对标竞品，当未指定dim时，self不能是空tensor
  if (dim->Size() != 1 && self->IsEmpty()) {
    OP_LOGE(ACLNN_ERR_PARAM_INVALID, "Expected self to be not empty tensor.");
    return false;
  }
  if (dim->Size() != 1) {
    return true;
  }
  auto axes = (*dim)[0];
  // 检查指定dim是否在self的维度范围内
  if (selfDimNum == 0 && axes != 0 && axes != -1) {
    OP_LOGE(ACLNN_ERR_PARAM_INVALID, "Expected dim to be in range of [-1, 0], but got %ld.", axes);
    return false;
  }
  if (selfDimNum == 0) {
    return true;
  }
  // dim不能超过范围
  if (axes >= selfDimNum || axes < (-selfDimNum)) {
    OP_LOGE(ACLNN_ERR_PARAM_INVALID, "Expected dim to be in range of [%ld, %ld], but got %ld.",
            -selfDimNum, selfDimNum - 1, axes);
    return false;
  }
  // dim指定的轴不能是空轴
  if (selfViewShape.GetDim(axes) == 0) {
    OP_LOGE(ACLNN_ERR_PARAM_INVALID, "Expected reduction dim %ld to have non-zero size, but check failed.", axes);
    return false;
  }
  return true;
}

static inline aclnnStatus CheckParams(const aclTensor *self, const aclIntArray *dim,
                                      const aclTensor *minOut, const aclTensor *maxOut) {
  // 1. 检查参数是否为空指针
  CHECK_RET(CheckNotNull(self, dim, minOut, maxOut), ACLNN_ERR_PARAM_NULLPTR);

  // 2. 检查输入的数据类型是否在API支持的数据类型范围之内，需要根据api定义校验
  CHECK_RET(CheckDtypeValid(self, minOut, maxOut), ACLNN_ERR_PARAM_INVALID);

  // 3. 检查tensor的维度
  CHECK_RET(CheckShape(self, minOut, maxOut), ACLNN_ERR_PARAM_INVALID);

  // 4. 检查dim是否在有效范围内
  CHECK_RET(CheckDimValid(self, dim), ACLNN_ERR_PARAM_INVALID);

  return ACLNN_SUCCESS;
}

aclnnStatus aclnnAminmaxGetWorkspaceSize(const aclTensor *self, const aclIntArray *dim, bool keepDim,
                                         aclTensor *minOut, aclTensor *maxOut,
                                         uint64_t *workspaceSize, aclOpExecutor **executor) {
  L2_DFX_PHASE_1(aclnnAminmax, DFX_IN(self, dim, keepDim), DFX_OUT(minOut, maxOut));
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

  // 如果self是bool类型，转为uint8类型
  auto selfCasted = selfContiguous->GetDataType() == op::DataType::DT_BOOL ?
                    l0op::Cast(selfContiguous, op::DataType::DT_UINT8, uniqueExecutor.get()) : selfContiguous;
  CHECK_RET(selfCasted != nullptr, ACLNN_ERR_INNER_NULLPTR);

  // 进行reducemin计算
  const aclTensor *min = l0op::ReduceMin(selfCasted, dim, keepDim, uniqueExecutor.get());
  CHECK_RET(min != nullptr, ACLNN_ERR_PARAM_NULLPTR);
  CHECK_RET(CheckShapeAndScalarSame(min, minOut), ACLNN_ERR_PARAM_INVALID);

  // 进行reducemax计算
  const aclTensor *max = l0op::ReduceMax(selfCasted, dim, keepDim, true, uniqueExecutor.get());
  CHECK_RET(max != nullptr, ACLNN_ERR_PARAM_NULLPTR);
  CHECK_RET(CheckShapeAndScalarSame(max, maxOut), ACLNN_ERR_PARAM_INVALID);

  // 固定写法，将计算结果转换成输出的数据类型
  auto castMinOut = l0op::Cast(min, minOut->GetDataType(), uniqueExecutor.get());
  CHECK_RET(castMinOut != nullptr, ACLNN_ERR_INNER_NULLPTR);

  // 固定写法，将计算结果转换成输出的数据类型
  auto castMaxOut = l0op::Cast(max, maxOut->GetDataType(), uniqueExecutor.get());
  CHECK_RET(castMaxOut != nullptr, ACLNN_ERR_INNER_NULLPTR);

  // 固定写法，将计算结果拷贝到输出out上，out可能是非连续的tensor
  auto viewCopyMinResult = l0op::ViewCopy(castMinOut, minOut, uniqueExecutor.get());
  CHECK_RET(viewCopyMinResult != nullptr, ACLNN_ERR_INNER_NULLPTR);

  // 固定写法，将计算结果拷贝到输出out上，out可能是非连续的tensor
  auto viewCopyMaxResult = l0op::ViewCopy(castMaxOut, maxOut, uniqueExecutor.get());
  CHECK_RET(viewCopyMaxResult != nullptr, ACLNN_ERR_INNER_NULLPTR);

  // 固定写法，获取计算过程中需要使用的workspace大小
  *workspaceSize = uniqueExecutor->GetWorkspaceSize();
  uniqueExecutor.ReleaseTo(executor);
  return ACLNN_SUCCESS;
}

aclnnStatus aclnnAminmax(void *workspace, uint64_t workspaceSize, aclOpExecutor *executor, aclrtStream stream) {
  L2_DFX_PHASE_2(aclnnAminmax);
  // 固定写法，调用框架能力，完成计算
  return CommonOpExecutorRun(workspace, workspaceSize, executor, stream);
}

#ifdef __cplusplus
}
#endif