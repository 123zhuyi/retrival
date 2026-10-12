/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include "svd.h"
#include "opdev/make_op_executor.h"
#include "opdev/aicpu/aicpu_task.h"
#include "opdev/op_def.h"
#include "opdev/op_dfx.h"
#include "opdev/op_executor.h"
#include "opdev/op_log.h"
#include "opdev/shape_utils.h"
#include "opdev/platform.h"
#include "aclnn_kernels/transpose.h"

using namespace op;

namespace l0op {
OP_TYPE_REGISTER(Svd);

static constexpr int32_t MIN_X_DIM = 2;

static std::tuple<aclTensor*, aclTensor*, aclTensor*>
  allocOutTensor(const aclTensor *x, const bool fullMatrices, aclOpExecutor *executor, const bool computeUV)
{
  int64_t xDim = x->GetViewShape().GetDimNum();
  op::Shape xShape = x->GetViewShape();
  int64_t m = xShape[xDim - MIN_X_DIM];
  int64_t n = xShape[xDim - 1];
  int64_t k = std::min(m, n);
  op::Shape sigmaShape;
  op::Shape uShape;
  op::Shape vShape;
  // sigma、u、v与x的前xDim-2维格式相同
  for (int64_t i = 0; i < xDim - MIN_X_DIM; i++) {
    sigmaShape.AppendDim(xShape[i]);
    if(computeUV) {
      uShape.AppendDim(xShape[i]);
      vShape.AppendDim(xShape[i]);
    }
  }
  /*
  * x:[..., m, n]  k = min(m, n)
  * fullMatrices == true --> u:[..., m, m]，(sigma):[..., k]，v:[..., n, n]，diag(sigma):[..., m, n]
  * fullMatrices == false --> u:[..., m, k]，(sigma):[..., k]，v:[..., n, k]，diag(sigma):[..., k, k]
  */
  if (computeUV) {
    if (fullMatrices) {
      uShape.AppendDim(m);
      uShape.AppendDim(m);
      vShape.AppendDim(n);
      vShape.AppendDim(n);
    } else {
      uShape.AppendDim(m);
      uShape.AppendDim(k);
      vShape.AppendDim(n);
      vShape.AppendDim(k);
    }
  }
  sigmaShape.AppendDim(k);
  auto u = executor->AllocTensor(uShape, x->GetDataType());
  auto sigma = executor->AllocTensor(sigmaShape, x->GetDataType());
  auto v = executor->AllocTensor(vShape, x->GetDataType());
  return std::tuple<aclTensor*, aclTensor*, aclTensor*>(sigma, u, v);
}


const std::tuple<aclTensor*, aclTensor*, aclTensor*> Svd(
  const aclTensor *x, const bool fullMatrices, const bool computeUV, aclOpExecutor *executor)
{
  std::tuple<aclTensor*, aclTensor*, aclTensor*> outTensor = allocOutTensor(x, fullMatrices, executor, computeUV);
  aclTensor *sigma = get<0>(outTensor);
  aclTensor *u = get<1>(outTensor);
  aclTensor *v = get<2>(outTensor);
  CHECK_RET(sigma != nullptr, (std::tuple<aclTensor*, aclTensor*, aclTensor*>(nullptr, nullptr, nullptr)));
  CHECK_RET(u != nullptr, (std::tuple<aclTensor*, aclTensor*, aclTensor*>(nullptr, nullptr, nullptr)));
  CHECK_RET(v != nullptr, (std::tuple<aclTensor*, aclTensor*, aclTensor*>(nullptr, nullptr, nullptr)));
  L0_DFX(Svd, x, sigma, u, v, fullMatrices, computeUV);
  static internal::AicpuTaskSpace space("Svd", ge::DEPEND_IN_SHAPE, true);
  auto ret = ADD_TO_LAUNCHER_LIST_AICPU(Svd, OP_ATTR_NAMES({"full_matrices", "compute_uv"}),
                                        OP_INPUT(x), OP_OUTPUT(sigma, u, v), OP_ATTR(fullMatrices, computeUV));
  OP_CHECK(ret == ACLNN_SUCCESS, OP_LOGE(ACLNN_ERR_INNER_NULLPTR, "Aicpu op Svd add to launcher list failed."),
           return (std::tuple<aclTensor*, aclTensor*, aclTensor*>(nullptr, nullptr, nullptr)));
  return std::tuple<aclTensor*, aclTensor*, aclTensor*>(sigma, u, v);
}

}  // namespace l0op