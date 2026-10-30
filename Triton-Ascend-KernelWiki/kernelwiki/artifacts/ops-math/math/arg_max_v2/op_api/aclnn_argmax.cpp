/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License")
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include "aclnn_argmax.h"
#include "argmax_v2.h"
#include "aclnn_kernels/cast.h"
#include "aclnn_kernels/reshape.h"
#include "aclnn_kernels/contiguous.h"
#include "aclnn_kernels/common/op_error_check.h"
#include "opdev/common_types.h"
#include "opdev/shape_utils.h"
#include "opdev/data_type_utils.h"
#include "opdev/format_utils.h"
#include "opdev/op_dfx.h"
#include "opdev/op_executor.h"
#include "opdev/op_log.h"
#include "opdev/tensor_view_utils.h"
#include "op_api/op_api_def.h"
#include "op_api/aclnn_check.h"


using namespace op;
#ifdef __cplusplus
extern "C" {
#endif

static const int64_t FIVE_DIM = 5;
static const int64_t SIX_DIM = 6;
static const int64_t SEVEN_DIM = 7;

static const std::initializer_list<DataType> ASCEND910_DTYPE_DTYPE_SUPPORT_LIST = {
      DataType::DT_FLOAT, DataType::DT_FLOAT16, DataType::DT_DOUBLE, DataType::DT_INT8, DataType::DT_INT16,
      DataType::DT_INT32, DataType::DT_INT64, DataType::DT_UINT8};

static const std::initializer_list<DataType> ASCEND910B_DTYPE_DTYPE_SUPPORT_LIST = {
      DataType::DT_FLOAT, DataType::DT_FLOAT16, DataType::DT_DOUBLE, DataType::DT_INT8, DataType::DT_INT16,
      DataType::DT_INT32, DataType::DT_INT64, DataType::DT_UINT8, DataType::DT_BF16};

static const std::initializer_list<DataType> ARCH_REGBASE_DTYPE_DTYPE_SUPPORT_LIST = {
      DataType::DT_FLOAT, DataType::DT_FLOAT16, DataType::DT_DOUBLE, DataType::DT_INT8, DataType::DT_INT16,
      DataType::DT_INT32, DataType::DT_INT64, DataType::DT_UINT8, op::DataType::DT_UINT16, DataType::DT_BF16};

static const std::initializer_list<DataType> EMPTY_DTYPE_SUPPORT_LIST = {};


static const inline std::initializer_list<DataType>& GetSupportDtypeList(NpuArch npuArch) {
  if (npuArch == NpuArch::DAV_2002 || npuArch == NpuArch::DAV_1001) {
    return ASCEND910_DTYPE_DTYPE_SUPPORT_LIST;
  } else if (npuArch == NpuArch::DAV_2201 || npuArch == NpuArch::DAV_3002) {
    return ASCEND910B_DTYPE_DTYPE_SUPPORT_LIST;
  } else if (IsRegBase(npuArch)) {
    return ARCH_REGBASE_DTYPE_DTYPE_SUPPORT_LIST;
  } else {
    return EMPTY_DTYPE_SUPPORT_LIST;
  }
}

static bool CheckDtypeValid(const aclTensor *self) {
  auto curArch = GetCurrentPlatformInfo().GetCurNpuArch();
  const auto& dTypeSupportList = GetSupportDtypeList(curArch);
  OP_CHECK_DTYPE_NOT_SUPPORT(self, dTypeSupportList, return false);
  return true;
}

static bool CheckNotNull(const aclTensor *self, const aclTensor *out) {
  OP_CHECK_NULL(self, return false);
  OP_CHECK_NULL(out, return false);
  return true;
}

static bool CheckShape(const aclTensor *self, const aclTensor *out) {
  OP_CHECK_MAX_DIM(self, MAX_SUPPORT_DIMS_NUMS, return false);
  OP_CHECK_MAX_DIM(out, MAX_SUPPORT_DIMS_NUMS, return false);
  return true;
}

static bool CheckDim(const aclTensor *self, int64_t dim) {
  int64_t shapeSize = self->GetViewShape().GetDimNum();
  int64_t minimum = shapeSize * (-1);
  int64_t maximum = shapeSize - 1;
  if (shapeSize == 0) {
    minimum = -1;
    maximum = 0;
  }
  if (dim < minimum || dim > maximum) {
    OP_LOGE(ACLNN_ERR_PARAM_INVALID, "dim must be within the range of [%ld, %ld], but it is %ld.",
            minimum, maximum, dim);
    return false;
  }
  return true;
}

static void CheckFormat(const aclTensor* self) {
      op::Format format = self->GetStorageFormat();
      if (format == Format::FORMAT_FRACTAL_NZ){
          OP_LOGW("Format of inputs gets [%s],this format mat lead to precision failure",op::ToString(format).GetString());
      }
  }
  
static bool CheckIfNeedReshape(const aclTensor *self) {
  return self->GetViewShape().GetDimNum() == MAX_SUPPORT_DIMS_NUMS &&
         self->GetDataType() != op::DataType::DT_FLOAT &&
         self->GetDataType() != op::DataType::DT_FLOAT16;
}

static bool ComputeShape(const aclTensor *self,
                         FVector<int64_t> &newShapeVector,
                         FVector<int64_t> &outShapeVector,
                         int64_t &dim,
                         const bool keepdim) {
  auto selfShape = self->GetViewShape();
  for (int64_t i = 0; i <= SEVEN_DIM; i++) {
    if (i == dim && keepdim) {
      outShapeVector.push_back(1);
    } else if (i != dim) {
      outShapeVector.push_back(selfShape[i]);
    }
  }

  if (dim <= FIVE_DIM) {
    for (int64_t i = 0; i <= FIVE_DIM; i++) {
      newShapeVector.push_back(selfShape[i]);
    }
    newShapeVector.push_back(selfShape[SIX_DIM] * selfShape[SEVEN_DIM]);
  } else {
    newShapeVector.push_back(selfShape[0] * selfShape[1]);
    for (int64_t i = 1; i <= SIX_DIM; i++) {
      newShapeVector.push_back(selfShape[i + 1]);
    }
    dim = (dim >= 1 ? dim - 1 : dim);
  }

  return true;
}

int64_t DimWrap(int64_t dim, int64_t dimNum) {
  if (dimNum <= 0) {
    dimNum = 1;
  }
  if (dim < 0) {
    dim += dimNum;
  }
  return dim;
}

static aclnnStatus CheckParams(const aclTensor *self, const aclTensor *out, int64_t dim) {
  // 1. 检查参数是否为空指针
  CHECK_RET(CheckNotNull(self, out), ACLNN_ERR_PARAM_NULLPTR);

  // 2. 检查输入的数据类型是否在API支持的数据类型范围之内，需要根据api定义校验
  CHECK_RET(CheckDtypeValid(self), ACLNN_ERR_PARAM_INVALID);

  // 3. 检查shape是否支持
  CHECK_RET(CheckShape(self, out), ACLNN_ERR_PARAM_INVALID);

  // 4. 检查dim是否在支持范围内
  CHECK_RET(CheckDim(self, dim), ACLNN_ERR_PARAM_INVALID);
  // 5. 检查是否是ND格式
  CheckFormat(self);
  return ACLNN_SUCCESS;
}

aclnnStatus aclnnArgMaxGetWorkspaceSize(const aclTensor *self, int64_t dim, bool keepdim,
                                        aclTensor *out, uint64_t *workspaceSize, aclOpExecutor **executor) {
  L2_DFX_PHASE_1(aclnnArgMax, DFX_IN(self, dim, keepdim), DFX_OUT(out));

  auto uniqueExecutor = CREATE_EXECUTOR();
  CHECK_RET(uniqueExecutor.get() != nullptr, ACLNN_ERR_INNER_CREATE_EXECUTOR);

  auto ret = CheckParams(self, out, dim);
  CHECK_RET(ret == ACLNN_SUCCESS, ret);

  if (self->IsEmpty() || out->IsEmpty()) {
    *workspaceSize = 0;
    uniqueExecutor.ReleaseTo(executor);
    return ACLNN_SUCCESS;
  }
  int64_t realDim = DimWrap(dim, self->GetViewShape().GetDimNum());

  if (op::ToShapeVector(self->GetViewShape()).size() == 0) {
    int64_t shape = 1;
    self = l0op::Reshape(self, uniqueExecutor.get()->AllocIntArray(&shape, 1), uniqueExecutor.get());
  }

  auto selfContiguous = l0op::Contiguous(self, uniqueExecutor.get());
  CHECK_RET(selfContiguous != nullptr, ACLNN_ERR_INNER_NULLPTR);

  const aclTensor *argmaxOut = nullptr;
  if (CheckIfNeedReshape(self)) {
    FVector<int64_t> newShapeVector;
    FVector<int64_t> outShapeVector;
    ComputeShape(self, newShapeVector, outShapeVector, realDim, keepdim);

    aclIntArray* newShapeArray = uniqueExecutor.get()->AllocIntArray(newShapeVector.data(), SEVEN_DIM);
    CHECK_RET(newShapeArray != nullptr, ACLNN_ERR_INNER_NULLPTR);
    selfContiguous = l0op::Reshape(selfContiguous, newShapeArray, uniqueExecutor.get());
    CHECK_RET(selfContiguous != nullptr, ACLNN_ERR_INNER_NULLPTR);

    argmaxOut = l0op::ArgMaxV2(selfContiguous, realDim, keepdim, uniqueExecutor.get());
    CHECK_RET(argmaxOut != nullptr, ACLNN_ERR_INNER_NULLPTR);

    aclIntArray* outShapeArray = uniqueExecutor.get()->AllocIntArray(outShapeVector.data(), outShapeVector.size());
    CHECK_RET(outShapeArray != nullptr, ACLNN_ERR_INNER_NULLPTR);
    argmaxOut = l0op::Reshape(argmaxOut, outShapeArray, uniqueExecutor.get());
  } else {
    argmaxOut = l0op::ArgMaxV2(selfContiguous, realDim, keepdim, uniqueExecutor.get());
  }
  CHECK_RET(argmaxOut != nullptr, ACLNN_ERR_INNER_NULLPTR);
  CHECK_RET(CheckShapeAndScalarSame(argmaxOut, out), ACLNN_ERR_PARAM_INVALID);

  auto castResult = l0op::Cast(argmaxOut, out->GetDataType(), uniqueExecutor.get());
  CHECK_RET(castResult != nullptr, ACLNN_ERR_INNER_NULLPTR);

  auto viewCopyResult = l0op::ViewCopy(castResult, out, uniqueExecutor.get());
  CHECK_RET(viewCopyResult != nullptr, ACLNN_ERR_INNER_NULLPTR);

  *workspaceSize = uniqueExecutor->GetWorkspaceSize();
  uniqueExecutor.ReleaseTo(executor);

  return ACLNN_SUCCESS;
}

aclnnStatus aclnnArgMax(void *workspace, uint64_t workspaceSize, aclOpExecutor *executor, aclrtStream stream) {
  L2_DFX_PHASE_2(aclnnArgMax);
  return CommonOpExecutorRun(workspace, workspaceSize, executor, stream);
}

#ifdef __cplusplus
}
#endif

