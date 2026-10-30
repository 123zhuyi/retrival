/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

/*!
 * \file aclnn_diag.cpp
 * \brief
 */

#include "aclnn_diag.h"
#include "aclnn_kernels/cast.h"
#include "aclnn_kernels/contiguous.h"
#include "diagv2.h"
#include "conversion/fill/op_api/fill.h"
#include "conversion/diag_flat/op_api/diag_flat.h"
#include "aclnn_kernels/common/op_error_check.h"
#include "opdev/common_types.h"
#include "opdev/data_type_utils.h"
#include "opdev/format_utils.h"
#include "opdev/op_executor.h"
#include "opdev/op_log.h"
#include "opdev/shape_utils.h"
#include "opdev/tensor_view_utils.h"
#include "opdev/platform.h"
#include "op_api/aclnn_check.h"

using namespace op;
#ifdef __cplusplus
extern "C" {
#endif

// 根据API定义，需要列出所能支持的所有dtype
static const std::initializer_list<op::DataType> ASCEND910_DTYPE_SUPPORT_LIST = {
    op::DataType::DT_FLOAT16, op::DataType::DT_FLOAT, op::DataType::DT_INT64, op::DataType::DT_INT32,
    op::DataType::DT_INT16, op::DataType::DT_INT8, op::DataType::DT_UINT8, op::DataType::DT_DOUBLE,
    op::DataType::DT_BOOL, op::DataType::DT_COMPLEX64};

static const std::initializer_list<op::DataType> ASCEND910B_DTYPE_SUPPORT_LIST = {
    op::DataType::DT_FLOAT16, op::DataType::DT_FLOAT, op::DataType::DT_INT64, op::DataType::DT_INT32,
    op::DataType::DT_INT16, op::DataType::DT_INT8, op::DataType::DT_UINT8, op::DataType::DT_DOUBLE,
    op::DataType::DT_BOOL, op::DataType::DT_COMPLEX64, op::DataType::DT_BF16};

static const std::initializer_list<op::DataType> EMPTY_INPUT_DTYPE_SUPPORT_LIST = {
    op::DataType::DT_FLOAT16, op::DataType::DT_FLOAT, op::DataType::DT_INT64, op::DataType::DT_INT32,
    op::DataType::DT_INT16, op::DataType::DT_INT8, op::DataType::DT_UINT8, op::DataType::DT_DOUBLE,
    op::DataType::DT_BOOL, op::DataType::DT_BF16};

static constexpr size_t DIM_TWO = 2;

static const std::initializer_list<DataType>& GetDtypeSupportList() {
  auto socVersion = GetCurrentPlatformInfo().GetSocVersion();
  if (socVersion == SocVersion::ASCEND910B || socVersion == SocVersion::ASCEND910_93 || IsRegBase()) {
    return ASCEND910B_DTYPE_SUPPORT_LIST;
  } else {
    return ASCEND910_DTYPE_SUPPORT_LIST;
  }
}

static bool CheckNotNull(const aclTensor* self, const aclTensor* out) {
  OP_CHECK_NULL(self, return false);
  OP_CHECK_NULL(out, return false);
  return true;
}

static bool CheckDtypeValid(const aclTensor* self, const aclTensor* out) {
  OP_CHECK_DTYPE_NOT_SUPPORT(self, GetDtypeSupportList(), return false);
  // Check real dtype equal to imag dtype
  OP_CHECK_DTYPE_NOT_SAME(self, out, return false);
  return true;
}

static bool CheckShape(const aclTensor* self) {
  OP_CHECK_MAX_DIM(self, DIM_TWO, return false);
  return true;
}

static bool CheckDiagonal(const aclTensor* self, int64_t diagonal) {
  auto input_shape = self->GetViewShape();
  int64_t input_dim_num = input_shape.GetDimNum();
  if (input_dim_num == DIM_TWO) {
    if (diagonal > 0 && input_shape.GetDim(1) == diagonal) {
      return true;
    }
    if (diagonal < 0 && input_shape.GetDim(0) == -diagonal) {
      return true;
    }
  }
  return false;
}

static aclnnStatus FillScalar(aclTensor *out, float val, aclOpExecutor *executor)
{
    OP_CHECK_DTYPE_NOT_SUPPORT(out, EMPTY_INPUT_DTYPE_SUPPORT_LIST, return ACLNN_ERR_PARAM_INVALID);
    FVector<int64_t> shape;
    size_t dimNum = out->GetViewShape().GetDimNum();
    for (size_t idx = 0; idx < dimNum; idx++) {
        int64_t tmpVal = out->GetViewShape().GetDim(idx);
        shape.push_back(tmpVal);
    }
    auto dims = executor->ConvertToTensor(shape.data(), shape.size(), DataType::DT_INT64);
    auto shapeArray = executor->AllocIntArray(shape.data(), shape.size());

    FVector<float> valVector = {val};
    auto valTensor = executor->ConvertToTensor(valVector.data(), valVector.size(), out->GetDataType());
    auto fillOut = l0op::Fill(dims, valTensor, shapeArray, executor);
    CHECK_RET(fillOut != nullptr, ACLNN_ERR_INNER_NULLPTR);
    auto viewCopyResult = l0op::ViewCopy(fillOut, out, executor);
    CHECK_RET(viewCopyResult != nullptr, ACLNN_ERR_INNER_NULLPTR);
    return ACLNN_SUCCESS;
}

static aclnnStatus CheckParams(const aclTensor* self, const aclTensor* y) {
  CHECK_RET(CheckNotNull(self, y), ACLNN_ERR_PARAM_NULLPTR);

  CHECK_RET(CheckDtypeValid(self, y), ACLNN_ERR_PARAM_INVALID);

  CHECK_RET(CheckShape(self), ACLNN_ERR_PARAM_INVALID);

  return ACLNN_SUCCESS;
}

static void CheckFormat(const aclTensor* self){
  ge::Format selfStorageFormat = self->GetStorageFormat();
  if (selfStorageFormat == ge::Format::FORMAT_FRACTAL_NZ){
    OP_LOGW("aclnnDiag doesn't support format NZ.");
  }
}

aclnnStatus aclnnDiagGetWorkspaceSize(const aclTensor* self, int64_t diagonal, aclTensor* out,
                                      uint64_t* workspaceSize, aclOpExecutor** executor) {
  L2_DFX_PHASE_1(aclnnDiag, DFX_IN(self, diagonal), DFX_OUT(out));
  auto uniqueExecutor = CREATE_EXECUTOR();
  CHECK_RET(uniqueExecutor.get() != nullptr, ACLNN_ERR_INNER_CREATE_EXECUTOR);

  auto ret = CheckParams(self, out);
  CHECK_RET(ret == ACLNN_SUCCESS, ret);

  CheckFormat(self);

  if ((self->IsEmpty() && diagonal == 0) || CheckDiagonal(self, diagonal)) {
    *workspaceSize = 0;
    uniqueExecutor.ReleaseTo(executor);
    return ACLNN_SUCCESS;
  }

  if ((self->IsEmpty() && std::abs(diagonal) > 0)) {
    ret = FillScalar(out, 0, uniqueExecutor.get());
    if (ret == ACLNN_SUCCESS) {
      *workspaceSize = uniqueExecutor->GetWorkspaceSize();
      uniqueExecutor.ReleaseTo(executor);
    }
    return ret;
  }

  auto selfContiguous = l0op::Contiguous(self, uniqueExecutor.get());
  CHECK_RET(selfContiguous != nullptr, ACLNN_ERR_INNER_NULLPTR);

  const aclTensor* diagOpOut = nullptr;
  if (self->GetViewShape().GetDimNum() < DIM_TWO) {
    diagOpOut = l0op::DiagFlat(selfContiguous, diagonal, uniqueExecutor.get());
  } else {
    diagOpOut = l0op::DiagV2(selfContiguous, diagonal, uniqueExecutor.get());
  }
  CHECK_RET(diagOpOut != nullptr, ACLNN_ERR_INNER_NULLPTR);
  CHECK_RET(CheckReduceOutShape(diagOpOut, out), ACLNN_ERR_PARAM_INVALID);
  auto viewCopyResult = l0op::ViewCopy(diagOpOut, out, uniqueExecutor.get());
  CHECK_RET(viewCopyResult != nullptr, ACLNN_ERR_INNER_NULLPTR);

  *workspaceSize = uniqueExecutor->GetWorkspaceSize();
  uniqueExecutor.ReleaseTo(executor);
  return ACLNN_SUCCESS;
}

aclnnStatus aclnnDiag(void* workspace, uint64_t workspaceSize, aclOpExecutor* executor, aclrtStream stream) {
  L2_DFX_PHASE_2(aclnnDiag);
  return CommonOpExecutorRun(workspace, workspaceSize, executor, stream);
}

#ifdef __cplusplus
}
#endif