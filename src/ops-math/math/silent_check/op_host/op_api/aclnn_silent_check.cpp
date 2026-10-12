/**
 * This program is free software, you can redistribute it and/or modify.
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This file is a part of the CANN Open Software.
 * Licensed under CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */
#include "aclnn_silent_check.h"
#include "silent_check_v2.h"
#include "aclnn_kernels/common/op_error_check.h"
#include "aclnn_kernels/contiguous.h"
#include "opdev/make_op_executor.h"
#include "opdev/platform.h"
#include "opdev/op_dfx.h"
#include "opdev/op_executor.h"

using namespace op;
#ifdef __cplusplus
extern "C" {
#endif

static const std::initializer_list<DataType> DTYPE_SUPPORT_LIST_FP16_FP32_BF16 = {DataType::DT_FLOAT16, DataType::DT_BF16, DataType::DT_FLOAT};

static const std::initializer_list<DataType> DTYPE_SUPPORT_LIST_FP32 = {DataType::DT_FLOAT};

static const std::initializer_list<DataType> DTYPE_SUPPORT_LIST_INT64 = {DataType::DT_INT64};

static constexpr int INDEX_0 = 0;
static constexpr int DIM_NUM_0 = 0;
static constexpr int DIM_NUM_1 = 1;
static constexpr size_t SFDA_DIM0_SIZE = 3;

static inline bool CheckNotNull(const aclTensor *val, aclTensor *inputGradRef, aclTensor *sfdaRef, aclTensor *stepRef, uint64_t *workspaceSize) {
  OP_CHECK_NULL(val, return false);
  OP_CHECK_NULL(inputGradRef, return false);
  OP_CHECK_NULL(sfdaRef, return false);
  OP_CHECK_NULL(stepRef, return false);
  if (workspaceSize == nullptr) {
    return false;
  }
  return true;
}

static inline bool CheckDtypeValid(const aclTensor *val, aclTensor *inputGradRef, aclTensor *sfdaRef, aclTensor *stepRef) {
  OP_CHECK_DTYPE_NOT_SUPPORT(val, DTYPE_SUPPORT_LIST_FP16_FP32_BF16, return false);
  OP_CHECK_DTYPE_NOT_SUPPORT(inputGradRef, DTYPE_SUPPORT_LIST_FP16_FP32_BF16, return false);
  OP_CHECK_DTYPE_NOT_SUPPORT(sfdaRef, DTYPE_SUPPORT_LIST_FP32, return false);
  OP_CHECK_DTYPE_NOT_SUPPORT(stepRef, DTYPE_SUPPORT_LIST_INT64, return false);
  return true;
}

static inline bool CheckShape(const aclTensor *val, aclTensor *sfdaRef, aclTensor *stepRef) {
  OP_CHECK_WRONG_DIMENSION(val, DIM_NUM_0, return false);
  OP_CHECK_WRONG_DIMENSION(sfdaRef, DIM_NUM_1, return false);
  OP_CHECK_WRONG_DIMENSION(stepRef, DIM_NUM_1, return false);

  if (sfdaRef->GetViewShape().GetDim(INDEX_0) != SFDA_DIM0_SIZE || stepRef->GetViewShape().GetDim(INDEX_0) != 1) {
    OP_LOGE(ACLNN_ERR_PARAM_INVALID, "dimension of input tensor error");
    return false;
  }
  return true;
}

static aclnnStatus CheckParams(const aclTensor *val, aclTensor *inputGradRef, aclTensor *sfdaRef, aclTensor *stepRef, uint64_t *workspaceSize) {
  // 1. 检查参数是否为空指针
  CHECK_RET(CheckNotNull(val, inputGradRef, sfdaRef, stepRef, workspaceSize),  ACLNN_ERR_PARAM_NULLPTR);

  // 2. 检查参数的数据类型是否符合预期
  CHECK_RET(CheckDtypeValid(val, inputGradRef, sfdaRef, stepRef), ACLNN_ERR_PARAM_INVALID);

  // 3. 检查输入tensor的shape
  CHECK_RET(CheckShape(val, sfdaRef, stepRef), ACLNN_ERR_PARAM_INVALID);

  return ACLNN_SUCCESS;
}

aclnnStatus aclnnSilentCheckGetWorkspaceSize(const aclTensor *val, aclTensor *inputGradRef, aclTensor *sfdaRef, aclTensor *stepRef, 
                                             const int32_t cMinSteps, const float cThreshL1, const float cCoeffL1, const float cThreshL2, const float cCoeffL2, 
                                             const int32_t npuAsdDetect, aclTensor* result,
                                             uint64_t *workspaceSize, aclOpExecutor **executor) {
  L2_DFX_PHASE_1(aclnnSilentCheck,
                 DFX_IN(val, inputGradRef, sfdaRef, stepRef, cMinSteps, cThreshL1, cCoeffL1, cThreshL2, cCoeffL2, npuAsdDetect),
                 DFX_OUT(inputGradRef, sfdaRef, stepRef, result));
                
  // 固定写法，创建OpExecutor
  auto uniqueExecutor = CREATE_EXECUTOR();
  CHECK_RET(uniqueExecutor.get() != nullptr, ACLNN_ERR_INNER_CREATE_EXECUTOR);

  // 参数检查
  auto ret = CheckParams(val, inputGradRef, sfdaRef, stepRef, workspaceSize);
  CHECK_RET(ret == ACLNN_SUCCESS, ret);

  // 空tensor处理
  if (val->IsEmpty() || inputGradRef->IsEmpty() || sfdaRef->IsEmpty() || stepRef->IsEmpty()) {
    *workspaceSize = 0;
    uniqueExecutor.ReleaseTo(executor);
    return ACLNN_SUCCESS;
  }

  // 将输入val转换成连续的tensor
  auto valContiguous = l0op::Contiguous(val, uniqueExecutor.get());
  CHECK_RET(valContiguous != nullptr, ACLNN_ERR_INNER_NULLPTR);

  // 将输入inputGradRef转换成连续的tensor
  auto inputGradRefContiguous = l0op::Contiguous(inputGradRef, uniqueExecutor.get());
  CHECK_RET(inputGradRefContiguous != nullptr, ACLNN_ERR_INNER_NULLPTR);

  // 将输入sfdaRef转换成连续的tensor
  auto sfdaRefContiguous = l0op::Contiguous(sfdaRef, uniqueExecutor.get());
  CHECK_RET(sfdaRefContiguous != nullptr, ACLNN_ERR_INNER_NULLPTR);

  // 将输入stepRef转换成连续的tensor
  auto stepRefContiguous = l0op::Contiguous(stepRef, uniqueExecutor.get());
  CHECK_RET(stepRefContiguous != nullptr, ACLNN_ERR_INNER_NULLPTR);

  // 执行L0算子
  auto silentCheckV2Result = l0op::SilentCheckV2(valContiguous, inputGradRefContiguous, sfdaRefContiguous, stepRefContiguous, cMinSteps, cThreshL1, cCoeffL1, cThreshL2, cCoeffL2, npuAsdDetect, uniqueExecutor.get());
  CHECK_RET(silentCheckV2Result != nullptr, ACLNN_ERR_INNER_NULLPTR);

  // 固定写法，将计算结果拷贝到输出result上，result可能是非连续的tensor
  auto viewCopyResult = l0op::ViewCopy(silentCheckV2Result, result, uniqueExecutor.get());
  CHECK_RET(viewCopyResult != nullptr, ACLNN_ERR_INNER_NULLPTR);

  // 固定写法，将计算结果拷贝到输出inputGradRef上
  auto viewCopyInputGradRef = l0op::ViewCopy(inputGradRefContiguous, inputGradRef, uniqueExecutor.get());
  CHECK_RET(viewCopyInputGradRef != nullptr, ACLNN_ERR_INNER_NULLPTR);

  // 固定写法，将计算结果拷贝到输出preValRef上
  auto viewCopySfdaRef = l0op::ViewCopy(sfdaRefContiguous, sfdaRef, uniqueExecutor.get());
  CHECK_RET(viewCopySfdaRef != nullptr, ACLNN_ERR_INNER_NULLPTR);

  // 固定写法，将计算结果拷贝到输出stepRef上
  auto viewCopyStepRef = l0op::ViewCopy(stepRefContiguous, stepRef, uniqueExecutor.get());
  CHECK_RET(viewCopyStepRef != nullptr, ACLNN_ERR_INNER_NULLPTR);

  // 获取计算过程中需要使用的workspace大小
  *workspaceSize = uniqueExecutor->GetWorkspaceSize();
  uniqueExecutor.ReleaseTo(executor);
  return ACLNN_SUCCESS;
}

aclnnStatus aclnnSilentCheck(void *workspace, uint64_t workspaceSize, aclOpExecutor *executor, aclrtStream stream) {
    L2_DFX_PHASE_2(aclnnSilentCheck);
    return CommonOpExecutorRun(workspace, workspaceSize, executor, stream);
}

#ifdef __cplusplus
}
#endif
