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

OP_TYPE_REGISTER(SqueezeNd);

bool SqueezeNdInferShape(const op::Shape& shape, const aclIntArray* dim, op::Shape& newShape)
{
    auto dimNum = static_cast<int64_t>(shape.GetDimNum());
    if (dim->Size() == 0) {
        for (auto i = 0; i < dimNum; i++) {
            if (shape[i] != 1) {
                newShape.AppendDim(shape[i]);
            }
        }
        return true;
    }
    op::FVector<bool, op::MAX_DIM_NUM> squeezeDim(dimNum, false);
    for (uint64_t i = 0; i < dim->Size(); i++) {
        auto idx = (*dim)[i];
        idx = idx >= 0 ? idx : dimNum + idx;
        if (idx >= dimNum || idx < 0) {
            OP_LOGE(
                ACLNN_ERR_PARAM_INVALID, "Invalid dim value for squeeze operation. shape: %s, dims: %s",
                op::ToString(shape).GetString(), dim->ToString().GetString());
            return false;
        }
        squeezeDim[idx] = shape[idx] == 1;
    }

    for (auto i = 0; i < dimNum; i++) {
        if (!squeezeDim[i]) {
            newShape.AppendDim(shape[i]);
        }
    }
    return true;
}

const aclTensor* SqueezeNd(const aclTensor* x, const aclIntArray* dim, aclOpExecutor* executor)
{
    L0_DFX(SqueezeNd, x, dim);
    if (!op::IsContiguous(x)) {
        OP_LOGE(ACLNN_ERR_PARAM_INVALID, "Input tensor must be contiguous.");
        return nullptr;
    }
    op::Shape newShape;
    if (!SqueezeNdInferShape(x->GetViewShape(), dim, newShape)) {
        OP_LOGE(ACLNN_ERR_PARAM_INVALID, "InferShape failed for squeeze operation.");
        return nullptr;
    }
    return executor->CreateView(x, newShape, x->GetViewOffset());
}

const aclTensor* SqueezeNd(const aclTensor* x, int64_t dim, aclOpExecutor* executor)
{
    return SqueezeNd(x, executor->AllocIntArray(&dim, 1), executor);
}
} // namespace l0op
