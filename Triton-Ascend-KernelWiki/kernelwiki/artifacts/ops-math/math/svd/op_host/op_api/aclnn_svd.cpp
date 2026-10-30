/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include "aclnn_svd.h"
#include "svd.h"
#include "conversion/view_copy/op_api/view_copy.h"
#include "aclnn_kernels/contiguous.h"
#include "aclnn/aclnn_base.h"
#include "aclnn_kernels/cast.h"
#include "opdev/shape_utils.h"
#include "opdev/common_types.h"
#include "opdev/data_type_utils.h"
#include "opdev/format_utils.h"
#include "opdev/op_dfx.h"
#include "opdev/op_executor.h"
#include "opdev/op_log.h"
#include "opdev/tensor_view_utils.h"
#include "aclnn_kernels/common/op_error_check.h"

using namespace op;
#ifdef __cplusplus
extern "C" {
#endif

static constexpr int32_t MIN_INPUT_DIM = 2;
static constexpr int32_t MAX_INPUT_DIM = 8;

static const std::initializer_list<op::DataType> DTYPE_SUPPORT_LIST = {
  op::DataType::DT_DOUBLE, op::DataType::DT_FLOAT,
  op::DataType::DT_COMPLEX64, op::DataType::DT_COMPLEX128};

static bool CheckNotNull(const aclTensor *input, const aclTensor *sigma, const aclTensor *u, const aclTensor *v)
{
  OP_CHECK_NULL(input, return false);
  OP_CHECK_NULL(sigma, return false);
  OP_CHECK_NULL(u, return false);
  OP_CHECK_NULL(v, return false);
  return true;
}

static bool CheckDtypeValid(const aclTensor *input, const aclTensor *sigma, const aclTensor *u, const aclTensor *v)
{
  OP_CHECK_DTYPE_NOT_SUPPORT(input, DTYPE_SUPPORT_LIST, return false);
  OP_CHECK_DTYPE_NOT_SUPPORT(sigma, DTYPE_SUPPORT_LIST, return false);
  OP_CHECK_DTYPE_NOT_SUPPORT(u, DTYPE_SUPPORT_LIST, return false);
  OP_CHECK_DTYPE_NOT_SUPPORT(v, DTYPE_SUPPORT_LIST, return false);
  OP_CHECK_DTYPE_NOT_SAME(input, sigma, return false);
  OP_CHECK_DTYPE_NOT_SAME(input, u, return false);
  OP_CHECK_DTYPE_NOT_SAME(input, v, return false);
  return true;
}

static bool CheckShape(const aclTensor *input, const aclTensor *sigma, const aclTensor *u,
                       const aclTensor *v, const bool fullMatrices, const bool computeUV)
{
  const int64_t inputDim = input->GetViewShape().GetDimNum();
  OP_CHECK(
    inputDim >= MIN_INPUT_DIM,
    OP_LOGE(ACLNN_ERR_PARAM_INVALID, "Input dim num should be greater than or equal to 2."), return false);
  OP_CHECK(
    inputDim <= MAX_INPUT_DIM,
    OP_LOGE(ACLNN_ERR_PARAM_INVALID, "Input dim num should be less than or equal to 8."), return false);
  op::Shape inputShape = input->GetViewShape();
  const int64_t m = inputShape[inputDim - MIN_INPUT_DIM];
  const int64_t n = inputShape[inputDim - 1];
  int64_t k = std::min(m, n);
  op::Shape sigmaShapeExpected;
  op::Shape uShapeExpected;
  op::Shape vShapeExpected;
  // sigma、u、v与input的前inputDim-2维格式相同
  for (int64_t i = 0; i < inputDim - MIN_INPUT_DIM; i++) {
    sigmaShapeExpected.AppendDim(inputShape[i]);
    uShapeExpected.AppendDim(inputShape[i]);
    vShapeExpected.AppendDim(inputShape[i]);
  }
  /*
  * input:[..., m, n]  k = min(m, n)
  * fullMatrices == true --> u:[..., m, m]，(sigma):[..., k]，v:[..., n, n]，diag(sigma):[..., m, n]
  * fullMatrices == false --> u:[..., m, k]，(sigma):[..., k]，v:[..., n, k]，diag(sigma):[..., k, k]
  */
  if (fullMatrices) {
    uShapeExpected.AppendDim(m);
    uShapeExpected.AppendDim(m);
    sigmaShapeExpected.AppendDim(k);
    vShapeExpected.AppendDim(n);
    vShapeExpected.AppendDim(n);
  } else {
    uShapeExpected.AppendDim(m);
    uShapeExpected.AppendDim(k);
    sigmaShapeExpected.AppendDim(k);
    vShapeExpected.AppendDim(n);
    vShapeExpected.AppendDim(k);
  }
  OP_CHECK(sigmaShapeExpected == sigma->GetViewShape(),
           OP_LOGE(ACLNN_ERR_PARAM_INVALID, "Shape of output tensor sigma is invalid."), return false);
  // computeUV为false时，不再校验输入U、V的shape
  if (computeUV) {
    OP_CHECK(uShapeExpected == u->GetViewShape(),
            OP_LOGE(ACLNN_ERR_PARAM_INVALID, "Shape of output tensor u is invalid."), return false);
    
    OP_CHECK(vShapeExpected == v->GetViewShape(),
            OP_LOGE(ACLNN_ERR_PARAM_INVALID, "Shape of output tensor v is invalid."), return false);
  }
  return true;
}

static aclnnStatus CheckParams(const aclTensor *input, const aclTensor *sigma, const aclTensor *u, 
                               const aclTensor *v, const bool fullMatrices, const bool computeUV)
{
  CHECK_RET(CheckNotNull(input, sigma, u, v), ACLNN_ERR_PARAM_NULLPTR);
  CHECK_RET(CheckDtypeValid(input, sigma, u, v), ACLNN_ERR_PARAM_INVALID);
  CHECK_RET(CheckShape(input, sigma, u, v, fullMatrices, computeUV), ACLNN_ERR_PARAM_INVALID);
  return ACLNN_SUCCESS;
}

aclnnStatus aclnnSvdGetWorkspaceSize(const aclTensor *input, const bool fullMatrices, const bool computeUV,
                                     aclTensor *sigma, aclTensor *u, aclTensor *v, uint64_t *workspaceSize,
                                     aclOpExecutor **executor) {
  L2_DFX_PHASE_1(aclnnSvd, DFX_IN(input, fullMatrices, computeUV), DFX_OUT(sigma, u, v));

  // 创建OpExecutor
  auto uniqueExecutor = CREATE_EXECUTOR();
  CHECK_RET(uniqueExecutor.get() != nullptr, ACLNN_ERR_INNER_CREATE_EXECUTOR);

  // 参数dtype检查
  auto ret = CheckParams(input, sigma, u, v, fullMatrices, computeUV);
  CHECK_RET(ret == ACLNN_SUCCESS, ret);

  if (input->IsEmpty()) {
    *workspaceSize = 0;
    uniqueExecutor.ReleaseTo(executor);
    OP_LOGD("The input tensor is empty.");
    return ACLNN_SUCCESS;
  }

  auto inputContiguous = l0op::Contiguous(input, uniqueExecutor.get());
  CHECK_RET(inputContiguous != nullptr, ACLNN_ERR_INNER_NULLPTR);

  auto [sigmaOut, uOut, vOut] = l0op::Svd(inputContiguous, fullMatrices, computeUV, uniqueExecutor.get());
  CHECK_RET(sigmaOut != nullptr, ACLNN_ERR_INNER_NULLPTR);
  CHECK_RET(uOut != nullptr, ACLNN_ERR_INNER_NULLPTR);
  CHECK_RET(vOut != nullptr, ACLNN_ERR_INNER_NULLPTR);

  // 将计算结果xxxOut拷贝到输出xxx上，xxx可能是非连续的tensor
  auto sigmaViewCopyResult = l0op::ViewCopy(sigmaOut, sigma, uniqueExecutor.get());
  CHECK_RET(sigmaViewCopyResult != nullptr, ACLNN_ERR_INNER_NULLPTR);
  if (computeUV) {
    auto uViewCopyResult = l0op::ViewCopy(uOut, u, uniqueExecutor.get());
    CHECK_RET(uViewCopyResult != nullptr, ACLNN_ERR_INNER_NULLPTR);
    auto vViewCopyResult = l0op::ViewCopy(vOut, v, uniqueExecutor.get());
    CHECK_RET(vViewCopyResult != nullptr, ACLNN_ERR_INNER_NULLPTR);
  }
  // 获取计算过程中需要使用的workspace大小
  *workspaceSize = uniqueExecutor->GetWorkspaceSize();
  uniqueExecutor.ReleaseTo(executor);
  return ACLNN_SUCCESS;
}

aclnnStatus aclnnSvd(void *workspace, uint64_t workspaceSize,
                     aclOpExecutor *executor, aclrtStream stream) {
  L2_DFX_PHASE_2(aclnnSvd);
  return CommonOpExecutorRun(workspace, workspaceSize, executor, stream);
}


#ifdef __cplusplus
}
#endif
