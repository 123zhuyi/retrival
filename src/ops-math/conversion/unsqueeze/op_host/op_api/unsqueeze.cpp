/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include "opdev/make_op_executor.h"
#include "opdev/op_dfx.h"
#include "opdev/tensor_view_utils.h"

namespace l0op {

OP_TYPE_REGISTER(UnsqueezeNd);

bool UnsqueezeNdInferShape(const op::Shape &shape, const aclIntArray* dim, op::Shape &newShape) {
  auto dimNum = static_cast<int64_t>(shape.GetDimNum());
  auto totalDimNum = dimNum + static_cast<int64_t>(dim->Size());
  newShape.SetDimNum(totalDimNum);

  op::FVector<int64_t, op::MAX_DIM_NUM> unsqueezeDim(dim->Size());
  for (uint64_t i = 0; i < dim->Size(); i++) {
    int64_t dimValue = (*dim)[i] < 0 ? totalDimNum + (*dim)[i] : (*dim)[i];
    if (dimValue >= totalDimNum || dimValue < 0) {
      OP_LOGE(ACL_ERROR_INVALID_PARAM, "Dimension is out of range. shape: %s, dim: %ld",
                op::ToString(shape).GetString(), dimValue);
      return false;
    }
    unsqueezeDim[i] = dimValue;
  }
  std::sort(unsqueezeDim.begin(), unsqueezeDim.end());
  auto uniqueSize = std::unique(unsqueezeDim.begin(), unsqueezeDim.end()) - unsqueezeDim.begin();
  if (uniqueSize != static_cast<int64_t>(unsqueezeDim.size())) {
    OP_LOGE(ACL_ERROR_INVALID_PARAM, "Dims should not contain duplicate values. shape: %s, dims: %s",
            op::ToString(shape).GetString(), dim->ToString().GetString());
    return false;
  }
  int64_t inIdx = 0;
  int64_t axIdx = 0;
  for (int64_t i = 0; i < totalDimNum; i++) {
    if (axIdx < static_cast<int64_t>(unsqueezeDim.size()) && unsqueezeDim[axIdx] == i) {
      newShape.SetDim(i, 1);
      axIdx++;
    } else {
      newShape.SetDim(i, shape[inIdx]);
      inIdx++;
    }
  }
  return true;
}

const aclTensor *UnsqueezeNd(const aclTensor *x, const aclIntArray* dim, aclOpExecutor *executor) {
  L0_DFX(UnsqueezeNd, x, dim);
  if (!op::IsContiguous(x)) {
    OP_LOGE(ACL_ERROR_FAILURE, "Input tensor must be contiguous.");
    return nullptr;
  }
  if (dim->Size() == 0) {
    return x;
  }
  op::Shape newShape;
  if (!UnsqueezeNdInferShape(x->GetViewShape(), dim, newShape)) {
    OP_LOGE(ACL_ERROR_FAILURE, "InferShape failed for squeeze operation.");
    return nullptr;
  }
  return executor->CreateView(x, newShape, x->GetViewOffset());
}

const aclTensor *UnsqueezeNd(const aclTensor *x, int64_t dim, aclOpExecutor *executor) {
    return UnsqueezeNd(x, executor->AllocIntArray(&dim, 1), executor);
}
} // namespace l0op
