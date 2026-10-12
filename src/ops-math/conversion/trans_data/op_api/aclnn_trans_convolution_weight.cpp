/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */
#include "aclnn_trans_convolution_weight.h"

#include "aclnn_kernels/cast.h"
#include "aclnn_kernels/contiguous.h"
#include "aclnn_kernels/transdata.h"
#include "conversion/tensor_move/op_api/tensor_move.h"
#include "aclnn_kernels/common/op_error_check.h"
#include "util/math_util.h"
#include "opdev/make_op_executor.h"
#include "opdev/common_types.h"
#include "opdev/data_type_utils.h"
#include "opdev/format_utils.h"
#include "opdev/op_dfx.h"
#include "opdev/op_executor.h"
#include "opdev/op_log.h"
#include "opdev/platform.h"
#include "opdev/tensor_view_utils.h"
#include "opdev/shape_utils.h"

using namespace op;
static uint64_t g_dimNum = 4;

#ifdef __cplusplus
extern "C" {
#endif

static uint64_t CalculateConvWeightSize(const aclTensor *weightFz) {
  CHECK_RET(weightFz->GetViewShape().GetDimNum() == g_dimNum, ACLNN_ERR_PARAM_INVALID);
  uint64_t shapeSize = 1;
  for (size_t i = 0; i < weightFz->GetViewShape().GetDimNum(); i++) {
    shapeSize *= static_cast<int64_t>(weightFz->GetStorageShape().GetDim(i));
  }
  return shapeSize;
}

static aclnnStatus CheckParams(uint64_t tensorDim, bool transposed, int64_t groups) {
  if (tensorDim != g_dimNum) {
    OP_LOGE(ACLNN_ERR_PARAM_INVALID, "only support 4 dim tensorShape, but got dim %lu.", tensorDim);
    return ACLNN_ERR_PARAM_INVALID;
  }

  if (transposed == true) {
    OP_LOGE(ACLNN_ERR_PARAM_INVALID,
            "only support nontranspose convolution, now transpose is %d.", transposed);
    return ACLNN_ERR_PARAM_INVALID;
  }

  if (groups == 0) {
    OP_LOGE(ACLNN_ERR_PARAM_INVALID,
            "only support groups larger than 0, now groups is %lu.", static_cast<uint64_t>(groups));
    return ACLNN_ERR_PARAM_INVALID;
  }

  auto soc = GetCurrentPlatformInfo().GetSocVersion();
  if (soc != SocVersion::ASCEND310P) {
    OP_LOGE(ACLNN_ERR_PARAM_INVALID,
            "only support ascend310P, now soc is %s.", op::ToString(soc).GetString());
    return ACLNN_ERR_PARAM_INVALID;
  }

  return ACLNN_SUCCESS;
}

static aclnnStatus CheckTensor(const aclTensor *weightIn, const aclTensor *weightOut) {
  if (weightIn->GetDataType() != op::DataType::DT_FLOAT16 && weightIn->GetDataType() != op::DataType::DT_FLOAT) {
    OP_LOGE(ACLNN_ERR_PARAM_INVALID,
            "only support weightIn datatype is float16/float32, now datatype is %s.",
            op::ToString(weightIn->GetDataType()).GetString());
    return ACLNN_ERR_PARAM_INVALID;
  }

  if (weightOut->GetDataType() != op::DataType::DT_FLOAT16) {
    OP_LOGE(ACLNN_ERR_PARAM_INVALID,
            "only support weightOut datatype is float16, now datatype is %s.",
            op::ToString(weightOut->GetDataType()).GetString());
    return ACLNN_ERR_PARAM_INVALID;
  }

  if (weightIn->GetViewFormat() != op::Format::FORMAT_NCHW) {
    OP_LOGE(ACLNN_ERR_PARAM_INVALID,
            "only support weightIn format is NCHW, now format is %s.",
            op::ToString(weightIn->GetViewFormat()).GetString());
    return ACLNN_ERR_PARAM_INVALID;
  }

  if (weightOut->GetViewFormat() != op::Format::FORMAT_NCHW) {
    OP_LOGE(ACLNN_ERR_PARAM_INVALID,
            "only support weightOut format is NCHW, now format is %s.",
            op::ToString(weightOut->GetViewFormat()).GetString());
    return ACLNN_ERR_PARAM_INVALID;
  }

  if (!op::IsContiguous(weightOut)) {
    OP_LOGE(ACLNN_ERR_PARAM_INVALID, "only support weightOut tensor is contiguous.");
    return ACLNN_ERR_PARAM_INVALID;
  }

  return ACLNN_SUCCESS;
}

aclnnStatus aclnnCalculateConvolutionWeightSize(const aclIntArray *tensorShape, bool transposed, int64_t groups,
                                                aclDataType dataType, uint64_t *weightTensorSize) {
  // 空指针校验
  OP_CHECK_NULL(tensorShape, return ACLNN_ERR_PARAM_NULLPTR);
  if (weightTensorSize == nullptr) {
    OP_LOGE(ACLNN_ERR_PARAM_NULLPTR,
            "expected a value of type number for argument weightTensorSize but instead found type null.");
    return ACLNN_ERR_PARAM_NULLPTR;
  }

  // 入参校验
  auto ret = CheckParams(tensorShape->Size(), transposed, groups);
  CHECK_RET(ret == ACLNN_SUCCESS, ACLNN_ERR_PARAM_INVALID);

  if (dataType != ACL_FLOAT16) {
    OP_LOGE(ACLNN_ERR_PARAM_INVALID,
            "only support datatype is float16, now datatype is %d.", dataType);
    return ACLNN_ERR_PARAM_INVALID;
  }

  // 空tensor处理：shape中任意维度为0时，tensor为空，直接返回0
  for (size_t i = 0; i < g_dimNum; i++) {
    int64_t dim = (*tensorShape)[i];
    if (dim < 0) {
      OP_LOGE(ACLNN_ERR_PARAM_INVALID, "tensorShape dim[%zu] must be non-negative, but got %ld.", i, dim);
      return ACLNN_ERR_PARAM_INVALID;
    }
    if (dim == 0) {
      *weightTensorSize = 0;
      return ACLNN_SUCCESS;
    }
  }

  // 固定写法，创建OpExecutor
  aclOpExecutor *executor;
  auto uniqueExecutor = CREATE_EXECUTOR();
  CHECK_RET(uniqueExecutor.get() != nullptr, ACLNN_ERR_INNER_CREATE_EXECUTOR);

  // 计算weightsize
  // 创建self aclTensor
  std::vector<int64_t> weightShape = {};
  uint64_t oriTensorSize = 1;
  for (size_t i = 0; i < g_dimNum; i++) {
    weightShape.push_back((*tensorShape)[i]);
    oriTensorSize *= (*tensorShape)[i];
  }

  void* dataAddr = nullptr;
  aclTensor* weight = nullptr;
  weight = aclCreateTensor(weightShape.data(), weightShape.size(), dataType, nullptr, 0, aclFormat::ACL_FORMAT_NCHW,
    weightShape.data(), weightShape.size(), dataAddr);
  CHECK_RET(weight != nullptr, ACLNN_ERR_INNER_NULLPTR);
  // 调用transdata_l0
  auto weightFz = l0op::TransData(weight, Format::FORMAT_FRACTAL_Z, groups, uniqueExecutor.get());
  CHECK_RET(weightFz != nullptr, ACLNN_ERR_INNER_NULLPTR);
  // 计算elements
  *weightTensorSize = CalculateConvWeightSize(weightFz);
  uniqueExecutor.ReleaseTo(&executor);
  return ACLNN_SUCCESS;
}

aclnnStatus aclnnTransConvolutionWeightGetWorkspaceSize(const aclTensor *weightIn, bool transposed,
                                                        const int64_t groups, aclTensor *weightOut,
                                                        uint64_t *workspaceSize, aclOpExecutor **executor) {
  L2_DFX_PHASE_1(aclnnTransConvolutionWeight, DFX_IN(weightIn, transposed, groups), DFX_OUT(weightOut));

  // 固定写法，参数检查
  OP_CHECK_NULL(weightIn, return ACLNN_ERR_PARAM_NULLPTR);
  OP_CHECK_NULL(weightOut, return ACLNN_ERR_PARAM_NULLPTR);

  // 入参校验
  uint64_t tensorDim = static_cast<uint64_t>(weightIn->GetViewShape().GetDimNum());
  auto ret = CheckParams(tensorDim, transposed, groups);
  CHECK_RET(ret == ACLNN_SUCCESS, ACLNN_ERR_PARAM_INVALID);
  ret = CheckTensor(weightIn, weightOut);
  CHECK_RET(ret == ACLNN_SUCCESS, ACLNN_ERR_PARAM_INVALID);

  // 固定写法，创建OpExecutor
  auto uniqueExecutor = CREATE_EXECUTOR();
  CHECK_RET(uniqueExecutor.get() != nullptr, ACLNN_ERR_INNER_CREATE_EXECUTOR);
  uniqueExecutor.get()->AbandonCache();

  // 空Tensor处理
  bool weightInEmpty = weightIn->IsEmpty();
  bool weightOutEmpty = weightOut->IsEmpty();
  if (weightInEmpty || weightOutEmpty) {
    if (weightInEmpty != weightOutEmpty) {
      OP_LOGE(ACLNN_ERR_PARAM_INVALID,
              "weightIn and weightOut should both be empty or non-empty, but got weightIn empty=%d, weightOut empty=%d.",
              static_cast<int>(weightInEmpty), static_cast<int>(weightOutEmpty));
      return ACLNN_ERR_PARAM_INVALID;
    }
    *workspaceSize = 0;
    uniqueExecutor.ReleaseTo(executor);
    return ACLNN_SUCCESS;
  }

  // 非连续转连续
  auto weightInContiguous = l0op::Contiguous(weightIn, uniqueExecutor.get());
  CHECK_RET(weightInContiguous != nullptr, ACLNN_ERR_INNER_NULLPTR);
  // cast
  auto weightInCast = l0op::Cast(weightInContiguous, weightOut->GetDataType(), uniqueExecutor.get());
  CHECK_RET(weightInCast != nullptr, ACLNN_ERR_INNER_NULLPTR);
  // transdata
  auto weightFz = l0op::TransData(weightInCast, Format::FORMAT_FRACTAL_Z, groups, uniqueExecutor.get());
  CHECK_RET(weightFz != nullptr, ACLNN_ERR_INNER_NULLPTR);

  aclTensor *weighttemp = const_cast<aclTensor *>(weightFz);

  // set tensor
  weighttemp->SetStorageFormat(weightFz->GetStorageFormat());
  weightOut->SetStorageFormat(weightFz->GetStorageFormat());
  weightOut->SetStorageShape(weighttemp->GetStorageShape());

  // set for view copy execute
  weighttemp->SetViewShape(weightOut->GetStorageShape());
  weightOut->SetViewShape(weightOut->GetStorageShape());
  weightOut->SetViewFormat(op::Format::FORMAT_FRACTAL_Z);

  // viewcopy
  auto viewCopyResult = l0op::ViewCopy(weighttemp, weightOut, uniqueExecutor.get());
  CHECK_RET(viewCopyResult != nullptr, ACLNN_ERR_INNER_NULLPTR);

  // set for convolution execute
  weightOut->SetViewShape(weightFz->GetOriginalShape());
  weightOut->SetViewFormat(op::Format::FORMAT_NCHW);

  // 固定写法，获取计算过程中需要使用的workspace大小
  *workspaceSize = uniqueExecutor->GetWorkspaceSize();
  uniqueExecutor.ReleaseTo(executor);
  return ACLNN_SUCCESS;
}

aclnnStatus aclnnTransConvolutionWeight(void *workspace, uint64_t workspaceSize,
                             aclOpExecutor *executor, const aclrtStream stream) {
  // 固定写法，调用框架能力，完成计算
  L2_DFX_PHASE_2(aclnnTransConvolutionWeight);
  return CommonOpExecutorRun(workspace, workspaceSize, executor, stream);
}

#ifdef __cplusplus
}
#endif
