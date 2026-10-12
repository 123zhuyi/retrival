/**
 * This program is free software, you can redistribute it and/or modify.
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This file is a part of the CANN Open Software.
 * Licensed under CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */
#include "aclnn_silent_check_v2.h"
#include "silent_check_v3.h"
#include "aclnn_kernels/common/op_error_check.h"
#include "aclnn_kernels/contiguous.h"
#include "opdev/make_op_executor.h"
#include "opdev/platform.h"
#include "opdev/op_dfx.h"
#include "opdev/op_executor.h"
#include "opdev/op_log.h"

using namespace op;
#ifdef __cplusplus
extern "C" {
#endif

static const std::initializer_list<DataType> DTYPE_SUPPORT_LIST_FP16_FP32_BF16 = {DataType::DT_FLOAT16, DataType::DT_BF16, DataType::DT_FLOAT};

static const std::initializer_list<DataType> DTYPE_SUPPORT_LIST_FP32 = {DataType::DT_FLOAT};

static const std::initializer_list<DataType> DTYPE_SUPPORT_LIST_INT64 = {DataType::DT_INT64};

static constexpr int DIM_NUM_1 = 1;

static inline bool CheckNotNull(const aclTensor *val, const aclTensor *max, aclTensor *avgRef, aclTensor *inputGradRef, aclTensor *stepRef, 
                                  aclTensor *dstSizeTensor, aclTensor *dstStrideTensor, aclTensor *dstOffsetTensor, uint64_t *workspaceSize) {
  OP_CHECK_NULL(val, return false);
  OP_CHECK_NULL(max, return false);
  OP_CHECK_NULL(avgRef, return false);
  OP_CHECK_NULL(inputGradRef, return false);
  OP_CHECK_NULL(stepRef, return false);
  OP_CHECK_NULL(dstSizeTensor, return false);
  OP_CHECK_NULL(dstStrideTensor, return false);
  OP_CHECK_NULL(dstOffsetTensor, return false);
  if (workspaceSize == nullptr) {
    return false;
  }
  return true;
}

static inline bool CheckIntArrNotNull(aclIntArray *dstSize, aclIntArray *dstStride, aclIntArray *dstOffset) {
  OP_CHECK_NULL(dstSize, return false);
  OP_CHECK_NULL(dstStride, return false);
  OP_CHECK_NULL(dstOffset, return false);
  return true;
}

static inline bool CheckDtypeValid(const aclTensor *val, const aclTensor *max, aclTensor *avgRef, aclTensor *inputGradRef, aclTensor *stepRef, 
                                  aclTensor *dstSizeTensor, aclTensor *dstStrideTensor, aclTensor *dstOffsetTensor) {
  OP_CHECK_DTYPE_NOT_SUPPORT(val, DTYPE_SUPPORT_LIST_FP16_FP32_BF16, return false);
  OP_CHECK_DTYPE_NOT_SUPPORT(max, DTYPE_SUPPORT_LIST_FP16_FP32_BF16, return false);
  OP_CHECK_DTYPE_NOT_SUPPORT(avgRef, DTYPE_SUPPORT_LIST_FP16_FP32_BF16, return false);
  OP_CHECK_DTYPE_NOT_SUPPORT(inputGradRef, DTYPE_SUPPORT_LIST_FP16_FP32_BF16, return false);
  OP_CHECK_DTYPE_NOT_SUPPORT(stepRef, DTYPE_SUPPORT_LIST_INT64, return false);
  OP_CHECK_DTYPE_NOT_SUPPORT(dstSizeTensor, DTYPE_SUPPORT_LIST_INT64, return false);
  OP_CHECK_DTYPE_NOT_SUPPORT(dstStrideTensor, DTYPE_SUPPORT_LIST_INT64, return false);
  OP_CHECK_DTYPE_NOT_SUPPORT(dstOffsetTensor, DTYPE_SUPPORT_LIST_INT64, return false);
  return true;
}

static inline bool CheckShape(const aclTensor *val, const aclTensor *max, aclTensor *avgRef, aclTensor *stepRef, aclTensor *dstSizeTensor, aclTensor *dstStrideTensor, aclTensor *dstOffsetTensor) {
  OP_CHECK_WRONG_DIMENSION(val, DIM_NUM_1, return false);
  OP_CHECK_WRONG_DIMENSION(max, DIM_NUM_1, return false);
  OP_CHECK_WRONG_DIMENSION(avgRef, DIM_NUM_1, return false);
  OP_CHECK_WRONG_DIMENSION(stepRef, DIM_NUM_1, return false);
  OP_CHECK_WRONG_DIMENSION(dstSizeTensor, DIM_NUM_1, return false);
  OP_CHECK_WRONG_DIMENSION(dstStrideTensor, DIM_NUM_1, return false);
  OP_CHECK_WRONG_DIMENSION(dstOffsetTensor, DIM_NUM_1, return false);
  return true;
}

static aclnnStatus CheckParams(const aclTensor *val, const aclTensor *max, aclTensor *avgRef, aclTensor *inputGradRef, aclTensor *stepRef, 
                              aclTensor *dstSizeTensor, aclTensor *dstStrideTensor, aclTensor *dstOffsetTensor, uint64_t *workspaceSize) {
  // 1. 检查参数是否为空指针
  CHECK_RET(CheckNotNull(val, max, avgRef, inputGradRef, stepRef, dstSizeTensor, dstStrideTensor, dstOffsetTensor, workspaceSize),  ACLNN_ERR_PARAM_NULLPTR);

  // 2. 检查参数的数据类型是否符合预期
  CHECK_RET(CheckDtypeValid(val, max, avgRef, inputGradRef, stepRef, dstSizeTensor, dstStrideTensor, dstOffsetTensor), ACLNN_ERR_PARAM_INVALID);

  // 3. 检查输入tensor的shape
  CHECK_RET(CheckShape(val, max, avgRef, stepRef, dstSizeTensor, dstStrideTensor, dstOffsetTensor), ACLNN_ERR_PARAM_INVALID);

  return ACLNN_SUCCESS;
}

static aclnnStatus CheckIntArrParams(aclIntArray *dstSize, aclIntArray *dstStride, aclIntArray *dstOffset) {
  // 1. 检查参数是否为空指针
  CHECK_RET(CheckIntArrNotNull(dstSize, dstStride, dstOffset),  ACLNN_ERR_PARAM_NULLPTR);

  return ACLNN_SUCCESS;
}

static aclnnStatus CheckAttrParams(float cThreshL1, float cThreshL2, float beta1) {
  // 1. 检查参数cThreshL1、cThreshL2大小
  if (cThreshL1 <= cThreshL2) {
    OP_LOGE(ACLNN_ERR_PARAM_INVALID, "the cThreshL1 value is %f, cannot be less than or equal to cThreshL2 value %f.", cThreshL1, cThreshL2);
    CHECK_RET(false,  ACLNN_ERR_PARAM_INVALID);
  }

  // 2. 检查参数beta1是否大于0且小于1
  if (beta1 < 0 || beta1 > 1) {
    OP_LOGE(ACLNN_ERR_PARAM_INVALID, "the beta1 value is %f, cannot be less than 0 or greater than 1.", beta1);
    CHECK_RET(false,  ACLNN_ERR_PARAM_INVALID);
  }
  return ACLNN_SUCCESS;
}

aclnnStatus aclnnSilentCheckV2GetWorkspaceSize(const aclTensor *val, const aclTensor *max, aclTensor *avgRef, aclTensor *inputGradRef, aclTensor *stepRef, 
                                             aclIntArray *dstSize, aclIntArray *dstStride, aclIntArray *dstOffset, float cThreshL1,  
                                             float cThreshL2, float beta1, int32_t npuAsdDetect, aclTensor* result,
                                             uint64_t *workspaceSize, aclOpExecutor **executor) {
  L2_DFX_PHASE_1(aclnnSilentCheckV2,
                 DFX_IN(val, max, avgRef, inputGradRef, stepRef, dstSize, dstStride, dstOffset, cThreshL1, cThreshL2, beta1, npuAsdDetect),
                 DFX_OUT(avgRef, inputGradRef, stepRef, result));
  // 固定写法，创建OpExecutor
  auto uniqueExecutor = CREATE_EXECUTOR();
  CHECK_RET(uniqueExecutor.get() != nullptr, ACLNN_ERR_INNER_CREATE_EXECUTOR);

  // aclIntArray参数检查
  auto intArrRet = CheckIntArrParams(dstSize, dstStride, dstOffset);
  CHECK_RET(intArrRet == ACLNN_SUCCESS, intArrRet);

  // aclIntArray参数转换成aclTensor
  auto dstSizeTensor = const_cast<aclTensor*>(uniqueExecutor->ConvertToTensor(dstSize, DataType::DT_INT64));
  auto dstStrideTensor = const_cast<aclTensor*>(uniqueExecutor->ConvertToTensor(dstStride, DataType::DT_INT64));
  auto dstOffsetTensor = const_cast<aclTensor*>(uniqueExecutor->ConvertToTensor(dstOffset, DataType::DT_INT64));

  // 参数检查
  auto ret = CheckParams(val, max, avgRef, inputGradRef, stepRef, dstSizeTensor, dstStrideTensor, dstOffsetTensor, workspaceSize);
  CHECK_RET(ret == ACLNN_SUCCESS, ret);
  auto attrRet = CheckAttrParams(cThreshL1, cThreshL2, beta1);
  CHECK_RET(attrRet == ACLNN_SUCCESS, attrRet);

  // 空tensor处理
  if (val->IsEmpty() || max->IsEmpty() || avgRef->IsEmpty() || inputGradRef->IsEmpty() || stepRef->IsEmpty() || dstSizeTensor->IsEmpty() 
    || dstStrideTensor->IsEmpty() || dstOffsetTensor->IsEmpty()) {
    *workspaceSize = 0;
    uniqueExecutor.ReleaseTo(executor);
    return ACLNN_SUCCESS;
  }

  // 执行L0算子SilentCheckV3
  auto silentCheckV3Result = l0op::SilentCheckV3(val, max, avgRef, inputGradRef, stepRef, dstSizeTensor, dstStrideTensor, dstOffsetTensor, cThreshL1, cThreshL2, beta1, npuAsdDetect, uniqueExecutor.get());
  CHECK_RET(silentCheckV3Result != nullptr, ACLNN_ERR_INNER_NULLPTR);

  // 固定写法，将计算结果拷贝到输出result上，result可能是非连续的tensor
  auto viewCopyResult = l0op::ViewCopy(silentCheckV3Result, result, uniqueExecutor.get());
  CHECK_RET(viewCopyResult != nullptr, ACLNN_ERR_INNER_NULLPTR);

  // 获取计算过程中需要使用的workspace大小
  *workspaceSize = uniqueExecutor->GetWorkspaceSize();
  uniqueExecutor.ReleaseTo(executor);
  return ACLNN_SUCCESS;
}

aclnnStatus aclnnSilentCheckV2(void *workspace, uint64_t workspaceSize, aclOpExecutor *executor, aclrtStream stream) {
    L2_DFX_PHASE_2(aclnnSilentCheckV2);
    return CommonOpExecutorRun(workspace, workspaceSize, executor, stream);
}

#ifdef __cplusplus
}
#endif
