/**
 * This program is free software, you can redistribute it and/or modify it.
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This file is a part of the CANN Open Software.
 * Licensed under CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include "aclnn_exp_segsum_backward.h"
#include "exp_segsum_grad.h"
#include "aclnn_kernels/contiguous.h"
#include "aclnn/aclnn_base.h"
#include "opdev/common_types.h"
#include "opdev/shape_utils.h"
#include "opdev/data_type_utils.h"
#include "opdev/format_utils.h"
#include "opdev/op_dfx.h"
#include "opdev/op_log.h"
#include "opdev/op_executor.h"
#include "opdev/tensor_view_utils.h"
#include "opdev/make_op_executor.h"
#include "aclnn_kernels/common/op_error_check.h"
#include "opdev/platform.h"

using namespace op;
#ifdef __cplusplus
extern "C" {
#endif

namespace
{
static const size_t TWODIMS = 2;
static const size_t THREEDIMS = 3;
static const size_t FOURDIMS = 4;
static const size_t FIVEDIMS = 5;

static const std::initializer_list<op::DataType> AICORE_DTYPE_DTYPE_SUPPORT_LIST = {
    op::DataType::DT_FLOAT16, op::DataType::DT_FLOAT,  op::DataType::DT_BF16};

static bool CheckNotNull(const aclTensor* gradOutput, const aclTensor* gradSelf, aclTensor *gradInput) {
  OP_CHECK_NULL(gradOutput, return false);
  OP_CHECK_NULL(gradSelf, return false);
  OP_CHECK_NULL(gradInput, return false);
  return true;
}

static bool CheckDtypeValid(const aclTensor* gradOutput, const aclTensor* gradSelf) {
  OP_CHECK_DTYPE_NOT_SUPPORT(gradOutput, AICORE_DTYPE_DTYPE_SUPPORT_LIST, return false);
  OP_CHECK_DTYPE_NOT_SUPPORT(gradSelf, AICORE_DTYPE_DTYPE_SUPPORT_LIST, return false);
  return true;
}

static bool CheckDtypeEqual(const aclTensor* gradOutput, aclTensor *gradInput) {
  OP_CHECK_DTYPE_NOT_MATCH(gradOutput, gradInput->GetDataType(), return false);
  return true;
}

static bool CheckShape(const aclTensor* gradOutput, const aclTensor* gradSelf, aclTensor *gradInput) {
  auto gradOutputShape = gradOutput->GetViewShape();
  auto gradSelfShape = gradSelf->GetViewShape();
  auto gradInputShape = gradInput->GetViewShape();

  auto gradOutputDimNum = gradOutputShape.GetDimNum();
  auto gradSelfDimNum = gradSelfShape.GetDimNum();
  auto gradInputDimNum = gradInputShape.GetDimNum();
  if (gradOutputDimNum != FOURDIMS && gradOutputDimNum != FIVEDIMS) {
    OP_LOGE(ACLNN_ERR_PARAM_INVALID, "It is expected dimNum of gradOutput equals to %zu or %zu, but got sizes %zu.",
            FOURDIMS, FIVEDIMS, gradSelfDimNum);
    return false;
  }
  if (gradOutputDimNum != gradSelfDimNum) {
    OP_LOGE(ACLNN_ERR_PARAM_INVALID, "It is expected dimNum of gradOutput equals to dimNum of gradSelf,\
            but got gradOutput(%zu), gradSelf(%zu).",
            gradOutputDimNum, gradSelfDimNum);
    return false;
  }
  if (gradOutputDimNum != gradInputDimNum + 1) {
    OP_LOGE(ACLNN_ERR_PARAM_INVALID,
            "It is expected dimNum of gradInput is 1 small than dimNum of gradOutput,\
            but got gradInput(%zu), gradOutput(%zu).",
            gradInputDimNum, gradOutputDimNum); 
    return false;
  }
  if (gradOutputShape.GetDim(gradOutputDimNum - 1) != gradOutputShape.GetDim(gradOutputDimNum - TWODIMS)) {
    OP_LOGE(ACLNN_ERR_PARAM_INVALID,
            "It is expected size of the last two dimensions of gradOutput are equal, but got gradOutput: %s.",
            op::ToString(gradOutputShape).GetString());
    return false;
  }
  for (size_t dim = 0; dim < gradOutputDimNum; dim++) {
    if (gradOutputShape.GetDim(dim) != gradSelfShape.GetDim(dim)) {
      OP_LOGE(ACLNN_ERR_PARAM_INVALID,
              "It is expected shape of gradSelf equals to shape of gradOutput, but got gradSelf: %s, gradOutput %s.",
              op::ToString(gradSelfShape).GetString(), op::ToString(gradOutputShape).GetString());
      return false;
    }
  }
  for (size_t dim = 0; dim < gradOutputDimNum - 1; dim++) {
    if (gradOutputShape.GetDim(dim) != gradInputShape.GetDim(dim)) {
      OP_LOGE(ACLNN_ERR_PARAM_INVALID,
              "It is expected size of gradInput equals to size of gradOutput at index %zu ,\
              but got gradInput: %s, gradOutput: %s.",
              dim, op::ToString(gradInputShape).GetString(), op::ToString(gradOutputShape).GetString());
      return false;
    }
  }
  return true;
}

static bool CheckFormat(const aclTensor* gradOutput, const aclTensor* gradSelf, aclTensor *gradInput) {
  Format gradOutputFormat = gradOutput->GetStorageFormat();
  Format gradSelfFormat = gradSelf->GetStorageFormat();
  Format gradInputFormat = gradInput->GetStorageFormat();
  if (gradOutputFormat != Format::FORMAT_ND) {
      OP_LOGE(ACLNN_ERR_PARAM_INVALID,
              "It is expected format of gradOutput is ND, but got gradOutput[%s].",
              op::ToString(gradOutputFormat).GetString());
      return false;
  }
  if (gradSelfFormat != gradOutputFormat) {
    OP_LOGE(ACLNN_ERR_PARAM_INVALID,
            "It is expected format of gradSelf equals to format of gradOutput, but gradSelf[%s], gradOutput[%s].",
            op::ToString(gradSelfFormat).GetString(), op::ToString(gradOutputFormat).GetString());
    return false;
  }
  if (gradInputFormat != gradOutputFormat) {
    OP_LOGE(ACLNN_ERR_PARAM_INVALID,
            "It is expected format of gradInput equals to format of gradOutput, but gradInput[%s], gradOutput[%s].",
            op::ToString(gradInputFormat).GetString(), op::ToString(gradOutputFormat).GetString());
    return false;
  }
  return true;
}

static aclnnStatus CheckParams(const aclTensor* gradOutput, const aclTensor* gradSelf, aclTensor *gradInput) {
  // 1. 检查参数是否为空指针
  CHECK_RET(CheckNotNull(gradOutput, gradSelf, gradInput), ACLNN_ERR_PARAM_NULLPTR);

  // 2. 检查输入的数据类型是否在API支持的数据类型范围之内
  CHECK_RET(CheckDtypeValid(gradOutput, gradSelf), ACLNN_ERR_PARAM_INVALID);

  // 3. 检查self和out数据类型是否相同
  CHECK_RET(CheckDtypeEqual(gradOutput, gradInput), ACLNN_ERR_PARAM_INVALID);

  // 4. 检查shape是否支持
  CHECK_RET(CheckShape(gradOutput, gradSelf, gradInput), ACLNN_ERR_PARAM_INVALID);

  // 5. 检查format是否支持
  CHECK_RET(CheckFormat(gradOutput, gradSelf, gradInput), ACLNN_ERR_PARAM_INVALID);

  return ACLNN_SUCCESS;
}
} // namespace
aclnnStatus aclnnExpSegsumBackwardGetWorkspaceSize(const aclTensor* gradOutput,const aclTensor* gradSelf,
                                                   aclTensor *gradInput, uint64_t *workspaceSize,
                                                   aclOpExecutor **executor) {
  OP_CHECK_COMM_INPUT(workspaceSize, executor);

  L2_DFX_PHASE_1(aclnnExpSegsumBackward, DFX_IN(gradOutput, gradSelf), DFX_OUT(gradInput));
  
  auto uniqueExecutor = CREATE_EXECUTOR();
  CHECK_RET(uniqueExecutor.get() != nullptr, ACLNN_ERR_INNER_CREATE_EXECUTOR);

  if (gradOutput->IsEmpty() || gradSelf->IsEmpty() || gradInput->IsEmpty()) {
    *workspaceSize = 0;
    uniqueExecutor.ReleaseTo(executor);
    return ACLNN_SUCCESS;
  }
  auto ret = CheckParams(gradOutput, gradSelf, gradInput);
  CHECK_RET(ret == ACLNN_SUCCESS, ret);

  auto gradOutputContiguous = l0op::Contiguous(gradOutput, uniqueExecutor.get());
  CHECK_RET(gradOutputContiguous != nullptr, ACLNN_ERR_INNER_NULLPTR);

  auto gradSelfContiguous = l0op::Contiguous(gradSelf, uniqueExecutor.get());
  CHECK_RET(gradSelfContiguous != nullptr, ACLNN_ERR_INNER_NULLPTR);
  auto result = l0op::ExpSegsumGrad(gradOutputContiguous, gradSelfContiguous, gradInput, uniqueExecutor.get());
  CHECK_RET(result != nullptr, ACLNN_ERR_INNER_NULLPTR);

  auto viewCopyResult = l0op::ViewCopy(result, gradInput, uniqueExecutor.get());
  CHECK_RET(viewCopyResult != nullptr, ACLNN_ERR_INNER_NULLPTR);

  *workspaceSize = uniqueExecutor->GetWorkspaceSize();
  uniqueExecutor.ReleaseTo(executor);
  return ACLNN_SUCCESS;
}

aclnnStatus aclnnExpSegsumBackward(void *workspace, uint64_t workspaceSize,
                                   aclOpExecutor *executor, aclrtStream stream) {
  L2_DFX_PHASE_2(aclnnExpSegsumBackward);
  return CommonOpExecutorRun(workspace, workspaceSize, executor, stream);
}

#ifdef __cplusplus
}
#endif