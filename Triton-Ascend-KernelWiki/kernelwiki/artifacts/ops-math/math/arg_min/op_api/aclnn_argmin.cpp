/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include <climits>
#include "aclnn_argmin.h"
#include "argmin.h"
#include "aclnn_kernels/cast.h"
#include "aclnn_kernels/contiguous.h"
#include "aclnn_kernels/reshape.h"
#include "aclnn_kernels/transdata.h"
#include "aclnn_kernels/common/op_error_check.h"
#include "opdev/common_types.h"
#include "opdev/shape_utils.h"
#include "opdev/data_type_utils.h"
#include "opdev/format_utils.h"
#include "opdev/op_dfx.h"
#include "opdev/op_executor.h"
#include "opdev/op_log.h"
#include "opdev/tensor_view_utils.h"
#include "opdev/platform.h"
#include "conversion/fill/op_api/fill.h"
#include "op_api/op_api_def.h"
#include "op_api/aclnn_check.h"

using namespace op;
#ifdef __cplusplus
extern "C" {
#endif

static const std::initializer_list<op::DataType> OUT_DTYPE_SUPPORT_LIST = {op::DataType::DT_INT32,
                                                                           op::DataType::DT_INT64};

static const int64_t FIVE_DIM = 5;

static const std::initializer_list<DataType> ASCEND910_DTYPE_DTYPE_SUPPORT_LIST = {
      DataType::DT_FLOAT, DataType::DT_FLOAT16, DataType::DT_DOUBLE, DataType::DT_INT8, DataType::DT_INT16,
      DataType::DT_INT32, DataType::DT_INT64, DataType::DT_UINT8};

static const std::initializer_list<DataType> ASCEND910B_DTYPE_DTYPE_SUPPORT_LIST = {
      DataType::DT_FLOAT, DataType::DT_FLOAT16, DataType::DT_DOUBLE, DataType::DT_INT8, DataType::DT_INT16,
      DataType::DT_INT32, DataType::DT_INT64, DataType::DT_UINT8, DataType::DT_BF16};

static const std::initializer_list<DataType> EMPTY_DTYPE_SUPPORT_LIST = {};

static const inline std::initializer_list<DataType>& GetSupportDtypeList(NpuArch arch) {
  if (arch == NpuArch::DAV_2002 || arch == NpuArch::DAV_1001) {
    return ASCEND910_DTYPE_DTYPE_SUPPORT_LIST;
  } else if (arch == NpuArch::DAV_2201 || arch == NpuArch::DAV_3002) {
    return ASCEND910B_DTYPE_DTYPE_SUPPORT_LIST;
  } else if (IsRegBase(arch)) {
    return ASCEND910B_DTYPE_DTYPE_SUPPORT_LIST;
  } else {
    return EMPTY_DTYPE_SUPPORT_LIST;
  }
}

static bool CheckDtypeValid(const aclTensor* self, const aclTensor* out) {
  auto curArch = GetCurrentPlatformInfo().GetCurNpuArch();
  const auto& dTypeSupportList = GetSupportDtypeList(curArch);
  OP_CHECK_DTYPE_NOT_SUPPORT(self, dTypeSupportList, return false);
  OP_CHECK_DTYPE_NOT_SUPPORT(out, OUT_DTYPE_SUPPORT_LIST, return false);

  return true;
}

static bool CheckNotNull(const aclTensor* self, const aclTensor* out) {
  OP_CHECK_NULL(self, return false);
  OP_CHECK_NULL(out, return false);

  return true;
}

static bool CheckFormat(const aclTensor* self, const aclTensor* out) {
  if (op::IsPrivateFormat(self->GetViewFormat()) || op::IsPrivateFormat(out->GetViewFormat())) {
    OP_LOGE(ACLNN_ERR_PARAM_INVALID,
            "Format only support ND,NCL,NCHW,NHWC,HWCN,NDHWC,NCDHW,"
            "input format is [%s] and out format is [%s].",
            ToString(self->GetViewFormat()).GetString(), ToString(out->GetViewFormat()).GetString());
    return false;
  }
  return true;
}

static bool CheckDimValid(const aclTensor* self, const int64_t dim) {
  const int64_t dimNum = self->GetViewShape().GetDimNum();
  int64_t minimum = dimNum * (-1);
  int64_t maximum = dimNum - 1;
  if (dimNum == 0) {
    minimum = -1;
    maximum = 0;
  }
  if (dim < minimum || dim > maximum) {
    OP_LOGE(ACLNN_ERR_PARAM_INVALID, "dim must be within the range of [%ld, %ld], but it is %ld.", minimum, maximum,
            dim);
    return false;
  }
  return true;
}

static bool CheckSelfDimSizeNonZreo(const aclTensor* self, const int64_t dim) {
  int64_t dimNum = self->GetViewShape().GetDimNum();
  if (dimNum == 0) {
    return true;
  }

  int64_t realDim = dim;
  if (realDim < 0) {
    realDim = realDim + dimNum;
  }

  if (self->GetViewShape().GetDim(realDim) == 0) {
    OP_LOGE(ACLNN_ERR_PARAM_INVALID, "Excepted reduction dim %ld to have non-zero size.", dim);
    return false;
  }
  return true;
}

static bool CheckShape(const aclTensor* self, const aclTensor* out) {
  // 所有输入的维度都不能超过8
  OP_CHECK_MAX_DIM(self, MAX_SUPPORT_DIMS_NUMS, return false);
  OP_CHECK_MAX_DIM(out, MAX_SUPPORT_DIMS_NUMS, return false);
  return true;
}

static aclnnStatus CheckParams(const aclTensor* self, const int64_t dim, const aclTensor* out) {
  // 1. 检查参数是否为空指针
  CHECK_RET(CheckNotNull(self, out), ACLNN_ERR_PARAM_NULLPTR);

  // 2. 检查dim 是否合法
  CHECK_RET(CheckDimValid(self, dim), ACLNN_ERR_PARAM_INVALID);

  // 3. 检查输入的数据类型是否在API支持的数据类型范围之内，需要根据api定义校验
  CHECK_RET(CheckDtypeValid(self, out), ACLNN_ERR_PARAM_INVALID);

  // 4. 检查数据格式是否支持
  CHECK_RET(CheckFormat(self, out), ACLNN_ERR_PARAM_INVALID);

  // 5. 检查shape是否支持
  CHECK_RET(CheckShape(self, out), ACLNN_ERR_PARAM_INVALID);

  // 检查self dim轴的size不为0
  CHECK_RET(CheckSelfDimSizeNonZreo(self, dim), ACLNN_ERR_PARAM_INVALID);

  return ACLNN_SUCCESS;
}

static bool CheckNeedReshape(const aclTensor* self) {
  auto selfViewShape = self->GetViewShape();
  // when self.dim > 8 and self.dtype is double, it will run tf kernel
  return selfViewShape.GetDimNum() == MAX_SUPPORT_DIMS_NUMS && self->GetDataType() == op::DataType::DT_DOUBLE;
}

static int64_t CalculateShape(const aclTensor* self, FVector<int64_t>& newShapeVector, FVector<int64_t>& outShapeVector,
                              const int64_t dim, const bool keepdim) {
  auto selfShape = self->GetViewShape();
  int64_t realDim = dim;
  if (realDim < 0) {
    realDim = selfShape.GetDimNum() + dim;
  }

  //  Calculate out shape
  for (size_t i = 0; i < MAX_SUPPORT_DIMS_NUMS; i++) {
    if (i != (size_t)realDim) {
      outShapeVector.push_back(selfShape[i]);
    } else if (keepdim) {
      outShapeVector.push_back(1);
    }
  }

  // if realDim is less than 5, merge the last two axes, otherwise merge axes 1 and 2
  if (realDim <= FIVE_DIM) {
    for (int64_t i = 0; i <= FIVE_DIM; i++) {
      newShapeVector.push_back(selfShape[i]);
    }
    newShapeVector.push_back(-1);
  } else {
    newShapeVector.push_back(selfShape[0] * selfShape[1]);
    realDim = realDim - 1;
    for (size_t i = 1; i < MAX_SUPPORT_DIMS_NUMS - 1; i++) {
      newShapeVector.push_back(selfShape[i + 1]);
    }
  }
  return realDim;
}

static aclnnStatus HandleScalar(aclTensor* out, aclOpExecutor* executor) {
  // fill value tensor
  float fillVaule = 0;
  aclScalar* value = executor->AllocScalar(fillVaule);
  const aclTensor* valueTensor = executor->ConvertToTensor(value, out->GetDataType());
  // fill dims tensor
  FVector<int64_t> dimTmp;
  dimTmp.push_back(1);
  aclIntArray* shapeArray = executor->AllocIntArray(dimTmp.data(), dimTmp.size());
  const aclTensor* dims = executor->ConvertToTensor(dimTmp.data(), dimTmp.size(), out->GetDataType());

  auto fillOut = l0op::Fill(dims, valueTensor, shapeArray, executor);
  CHECK_RET(fillOut != nullptr, ACLNN_ERR_INNER_NULLPTR);

  auto viewCopyResult = l0op::ViewCopy(fillOut, out, executor);
  CHECK_RET(viewCopyResult != nullptr, ACLNN_ERR_INNER_NULLPTR);

  return ACLNN_SUCCESS;
}

aclnnStatus aclnnArgMinGetWorkspaceSize(const aclTensor* self, int64_t dim, bool keepdim, aclTensor* out,
                                        uint64_t* workspaceSize, aclOpExecutor** executor) {
  L2_DFX_PHASE_1(aclnnArgMin, DFX_IN(self, dim, keepdim), DFX_OUT(out));

  auto ret = CheckParams(self, dim, out);
  CHECK_RET(ret == ACLNN_SUCCESS, ret);

  auto uniqueExecutor = CREATE_EXECUTOR();
  CHECK_RET(uniqueExecutor.get() != nullptr, ACLNN_ERR_INNER_CREATE_EXECUTOR);

  int64_t dimNum = self->GetViewShape().GetDimNum();
  if (self->IsEmpty() || dimNum == 0) {
    if (dimNum == 0) {
      ret = HandleScalar(out, uniqueExecutor.get());
      CHECK_RET(ret == ACLNN_SUCCESS, ret);
    }
    *workspaceSize = 0;
    uniqueExecutor.ReleaseTo(executor);
    return ACLNN_SUCCESS;
  }

  auto selfContiguous = l0op::Contiguous(self, uniqueExecutor.get());
  CHECK_RET(selfContiguous != nullptr, ACLNN_ERR_INNER_NULLPTR);

  const aclTensor* argminOut = nullptr;
  // if self is 8 dims and will run tf kernel, it should merge axes
  if (CheckNeedReshape(self)) {
    FVector<int64_t> newShapeVector;
    FVector<int64_t> outShapeVector;
    auto realDim = CalculateShape(selfContiguous, newShapeVector, outShapeVector, dim, keepdim);

    aclIntArray* newShapeArray = uniqueExecutor.get()->AllocIntArray(newShapeVector.data(), MAX_SUPPORT_DIMS_NUMS - 1);
    CHECK_RET(newShapeArray != nullptr, ACLNN_ERR_INNER_NULLPTR);
    selfContiguous = l0op::Reshape(selfContiguous, newShapeArray, uniqueExecutor.get());
    CHECK_RET(selfContiguous != nullptr, ACLNN_ERR_INNER_NULLPTR);

    argminOut = l0op::ArgMin(selfContiguous, realDim, keepdim, out->GetDataType(), uniqueExecutor.get());
    CHECK_RET(argminOut != nullptr, ACLNN_ERR_INNER_NULLPTR);

    aclIntArray* outShapeArray = uniqueExecutor.get()->AllocIntArray(outShapeVector.data(), outShapeVector.size());
    CHECK_RET(outShapeArray != nullptr, ACLNN_ERR_INNER_NULLPTR);
    argminOut = l0op::Reshape(argminOut, outShapeArray, uniqueExecutor.get());
  } else {
    // if self.dtype is IntegralType, it cast self.dtype to int64 and it run aicore
    if (IsIntegralType(self->GetDataType(), true)) {
      selfContiguous = l0op::Cast(selfContiguous, op::DataType::DT_INT64, uniqueExecutor.get());
      CHECK_RET(selfContiguous != nullptr, ACLNN_ERR_INNER_NULLPTR);
    }
    argminOut = l0op::ArgMin(selfContiguous, dim, keepdim, out->GetDataType(), uniqueExecutor.get());
  }
  CHECK_RET(argminOut != nullptr, ACLNN_ERR_INNER_NULLPTR);
  CHECK_RET(CheckShapeAndScalarSame(argminOut, out), ACLNN_ERR_PARAM_INVALID);

  auto castResult = l0op::Cast(argminOut, out->GetDataType(), uniqueExecutor.get());
  CHECK_RET(castResult != nullptr, ACLNN_ERR_INNER_NULLPTR);

  auto reformatResult = l0op::ReFormat(castResult, out->GetViewFormat());
  CHECK_RET(reformatResult != nullptr, ACLNN_ERR_INNER_NULLPTR);

  auto viewCopyResult = l0op::ViewCopy(reformatResult, out, uniqueExecutor.get());
  CHECK_RET(viewCopyResult != nullptr, ACLNN_ERR_INNER_NULLPTR);

  *workspaceSize = uniqueExecutor->GetWorkspaceSize();
  uniqueExecutor.ReleaseTo(executor);

  return ACLNN_SUCCESS;
}

aclnnStatus aclnnArgMin(void* workspace, uint64_t workspaceSize, aclOpExecutor* executor, const aclrtStream stream) {
  L2_DFX_PHASE_2(aclnnArgMin);
  return CommonOpExecutorRun(workspace, workspaceSize, executor, stream);
}

#ifdef __cplusplus
}
#endif
