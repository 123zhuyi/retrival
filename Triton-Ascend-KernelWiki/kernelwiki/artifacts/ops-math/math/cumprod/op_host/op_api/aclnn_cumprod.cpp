/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include "aclnn_cumprod.h"
#include "cumprod.h"
#include "aclnn_kernels/common/op_error_check.h"
#include "aclnn_kernels/contiguous.h"
#include "op_api/op_api_def.h"
#include "aclnn_kernels/cast.h"
#include "opdev/make_op_executor.h"
#include "opdev/platform.h"
#include "opdev/op_dfx.h"
#include "opdev/op_executor.h"
#include "aclnn_kernels/reshape.h"

using namespace op;
namespace {
  static const std::initializer_list<DataType> DTYPE_SUPPORT_LIST = {DataType::DT_FLOAT, DataType::DT_FLOAT16,DataType::DT_BF16,DataType::DT_DOUBLE,DataType::DT_INT8,DataType::DT_INT16,DataType::DT_INT32,DataType::DT_INT64,DataType::DT_UINT8,DataType::DT_UINT16,DataType::DT_UINT32,DataType::DT_UINT64};

  static const std::initializer_list<DataType> DTYPE_SUPPORT_LIST_INT = {DataType::DT_INT32,DataType::DT_INT64};
}
#ifdef __cplusplus
extern "C" {
#endif

static constexpr int INDEX_0 = 0;
static constexpr int DIM_NUM_0 = 0;
static constexpr int DIM_NUM_1 = 1;
static constexpr size_t SFDA_DIM0_SIZE = 3;

static inline bool CheckNotNull(const aclTensor *input, const aclScalar *inputDim, const uint64_t *workspaceSize) {
  OP_CHECK_NULL(input, return false);
  OP_CHECK_NULL(inputDim, return false);
  if (workspaceSize == nullptr) {
    return false;
  }
  return true;
}

static inline bool CheckDtypeValid(const aclTensor *input, const aclTensor *out, const aclScalar *inputDim) {
  OP_CHECK_DTYPE_NOT_SUPPORT(input, DTYPE_SUPPORT_LIST, return false);
  OP_CHECK_DTYPE_NOT_SUPPORT(out, DTYPE_SUPPORT_LIST, return false);
  OP_CHECK_DTYPE_NOT_SUPPORT(inputDim, DTYPE_SUPPORT_LIST_INT, return false);
  return true;
}

static inline bool CheckShape(const aclTensor *input, const aclTensor *out, const aclScalar *axis) {
  int32_t inputDim = input->GetViewShape().GetDimNum();
  int32_t inputAxis = axis->ToInt32();
  if (inputDim == 0 && inputAxis != 0 && inputAxis != -1) {
    OP_LOGE(ACLNN_ERR_PARAM_INVALID, "Dimension out of range (expected to be in range of [-1, 0], but got %d)", inputAxis);
    return false;
  } else if(inputDim > 0 && (inputAxis > inputDim - 1 || inputAxis < -inputDim)) {
    OP_LOGE(ACLNN_ERR_PARAM_INVALID, "input dim(%d) out of range(%d,%d)", inputAxis, -inputDim, inputDim - 1);
    return false;
  }
  OP_CHECK_SHAPE_NOT_EQUAL(out,input,return false);
  return true;
}

static aclnnStatus CheckParams(const aclTensor *input, const aclTensor *out, const aclScalar *inputDim, uint64_t *workspaceSize) {
  // 1. 检查参数是否为空指针
  CHECK_RET(CheckNotNull(input, inputDim, workspaceSize),  ACLNN_ERR_PARAM_NULLPTR);
  // 2. 检查参数的数据类型是否符合预期
  CHECK_RET(CheckDtypeValid(input, out, inputDim), ACLNN_ERR_PARAM_INVALID);
  // 3. 检查输入tensor的shape
  CHECK_RET(CheckShape(input, out, inputDim), ACLNN_ERR_PARAM_INVALID);
  return ACLNN_SUCCESS;
}

namespace {
  static const aclTensor *AdaptInputZeroDimTensor(const aclTensor *self, int64_t dimNum, aclOpExecutor *executor) {
    if (dimNum != 0) {
      return self;
    }
    int64_t selfShapeValue[1] = {1};
    aclIntArray *selfShape = executor->AllocIntArray(selfShapeValue, 1);
    auto selfReshape = l0op::Reshape(self, selfShape, executor);
    return selfReshape;
  }
}

static aclnnStatus doGetWorkspaceSize(aclTensor *input, const aclScalar *inputDim, const aclDataType dtype, aclTensor *out, 
                                          uint64_t *workspaceSize, aclOpExecutor **executor) {
  // 固定写法，创建OpExecutor
  auto uniqueExecutor = CREATE_EXECUTOR();
  CHECK_RET(uniqueExecutor.get() != nullptr, ACLNN_ERR_INNER_CREATE_EXECUTOR);
  // 检查Format
  if(input->GetStorageFormat() != Format::FORMAT_ND){
    OP_LOGW("Format only support ND");
  }
  // 参数检查
  auto ret = CheckParams(input, out, inputDim, workspaceSize);
  CHECK_RET(ret == ACLNN_SUCCESS, ret);
 
  auto tensorType = op::ToOpDataType(dtype);
  if(tensorType == DataType::DT_UNDEFINED){
    tensorType = out->GetDataType();
  }
  OP_CHECK_DTYPE_NOT_MATCH(out, tensorType, return false);

  // 空tensor处理
  if (input->IsEmpty() || out->IsEmpty()) {
      *workspaceSize = 0;
      uniqueExecutor.ReleaseTo(executor);
      return ACLNN_SUCCESS;
  }

  auto inputContinuous = l0op::Contiguous(input, uniqueExecutor.get());
  CHECK_RET(inputContinuous != nullptr, ACLNN_ERR_INNER_NULLPTR);

  int64_t dimNum = static_cast<int64_t>(inputContinuous->GetViewShape().GetDimNum());
  auto inputReshape = AdaptInputZeroDimTensor(inputContinuous, dimNum, uniqueExecutor.get());
  CHECK_RET(inputReshape != nullptr, ACLNN_ERR_INNER_NULLPTR);

  if(input->GetDataType() != tensorType){
    inputReshape = l0op::Cast(inputReshape, tensorType, uniqueExecutor.get());
    CHECK_RET(inputReshape != nullptr, ACLNN_ERR_INNER_NULLPTR);
  }
  // 执行L0算子
  auto cumprodResult = l0op::Cumprod(inputReshape, inputDim, false, false, uniqueExecutor.get());
  CHECK_RET(cumprodResult != nullptr, ACLNN_ERR_INNER_NULLPTR);

  // 固定写法，将计算结果拷贝到输出out上，out可能是非连续的tensor
  auto result = l0op::ViewCopy(cumprodResult, out, uniqueExecutor.get());
  CHECK_RET(result != nullptr, ACLNN_ERR_INNER_NULLPTR);
  // 获取计算过程中需要使用的workspace大小
  *workspaceSize = uniqueExecutor->GetWorkspaceSize();
  uniqueExecutor.ReleaseTo(executor);
  return ACLNN_SUCCESS;
}

aclnnStatus aclnnCumprodGetWorkspaceSize(const aclTensor *input, const aclScalar *dim, const aclDataType dtype,
                                         aclTensor *out, uint64_t *workspaceSize, aclOpExecutor **executor) {
  L2_DFX_PHASE_1(aclnnCumprod, DFX_IN(input, dim, dtype), DFX_OUT(out));
  auto inputNonConst = const_cast<aclTensor*>(input);
  return doGetWorkspaceSize(inputNonConst, dim, dtype, out, workspaceSize, executor);
}

aclnnStatus aclnnInplaceCumprodGetWorkspaceSize(aclTensor *input, const aclScalar *dim, uint64_t *workspaceSize,
                                                aclOpExecutor **executor) {
  L2_DFX_PHASE_1(aclnnInplaceCumprod, DFX_IN(input, dim), DFX_OUT(input));
  return doGetWorkspaceSize(input, dim, ToAclDataType(input->GetDataType()), input, workspaceSize, executor);
}
aclnnStatus aclnnCumprod(void *workspace, uint64_t workspaceSize, aclOpExecutor *executor, aclrtStream stream) {
  L2_DFX_PHASE_2(aclnnCumprod);
  return CommonOpExecutorRun(workspace, workspaceSize, executor, stream);
}
aclnnStatus aclnnInplaceCumprod(void *workspace, uint64_t workspaceSize, aclOpExecutor *executor, aclrtStream stream) {
  L2_DFX_PHASE_2(aclnnInplaceCumprod);
  return CommonOpExecutorRun(workspace, workspaceSize, executor, stream);
}
#ifdef __cplusplus
}
#endif
