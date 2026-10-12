/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include <tuple>
#include <vector>
#include "opdev/op_executor.h"

namespace l0op {

namespace {
op::Shape MakeReduceShape(const aclTensor* self, const aclIntArray* dim, bool keepdim)
{
    op::Shape reduceShape;
    const auto selfShape = self->GetViewShape();
    const size_t dimNum = selfShape.GetDimNum();
    std::vector<bool> reduceMask(dimNum, false);

    if (dim == nullptr || dim->Size() == 0) {
        for (size_t i = 0; i < dimNum; ++i) {
            reduceMask[i] = true;
        }
    } else {
        for (size_t i = 0; i < dim->Size(); ++i) {
            int64_t index = dim->operator[](i);
            if (index < 0) {
                index += static_cast<int64_t>(dimNum);
            }
            if (index >= 0 && index < static_cast<int64_t>(dimNum)) {
                reduceMask[static_cast<size_t>(index)] = true;
            }
        }
    }

    for (size_t i = 0; i < dimNum; ++i) {
        if (reduceMask[i]) {
            if (keepdim) {
                reduceShape.AppendDim(1);
            }
        } else {
            reduceShape.AppendDim(selfShape.GetDim(i));
        }
    }
    return reduceShape;
}

}  // namespace

const aclTensor* ReduceMean(const aclTensor* self, const aclIntArray* dim, bool keepdim, aclOpExecutor* executor)
{
    return executor->AllocTensor(MakeReduceShape(self, dim, keepdim), self->GetDataType(), op::Format::FORMAT_ND);
}

const aclTensor* ReduceStdWithMean(
    const aclTensor* self, const aclTensor*, const aclIntArray* dim, int64_t, bool keepdim, bool, float,
    aclOpExecutor* executor)
{
    return executor->AllocTensor(MakeReduceShape(self, dim, keepdim), self->GetDataType(), op::Format::FORMAT_ND);
}

const aclTensor* ReduceStdV2Update(
    const aclTensor* self, const aclTensor*, const aclIntArray* dim, bool, bool keepdim, aclOpExecutor* executor)
{
    return executor->AllocTensor(MakeReduceShape(self, dim, keepdim), self->GetDataType(), op::Format::FORMAT_ND);
}

const aclTensor* ReduceStdV2UpdateCorrection(
    const aclTensor* self, const aclTensor*, const aclIntArray* dim, int64_t, bool keepdim, aclOpExecutor* executor)
{
    return executor->AllocTensor(MakeReduceShape(self, dim, keepdim), self->GetDataType(), op::Format::FORMAT_ND);
}

const std::tuple<const aclTensor*, const aclTensor*> ReduceStdV2(
    const aclTensor* self, const aclIntArray* dim, int64_t, bool keepdim, bool, aclOpExecutor* executor)
{
    const auto out = executor->AllocTensor(MakeReduceShape(self, dim, keepdim), self->GetDataType(),
        op::Format::FORMAT_ND);
    return std::make_tuple(out, out);
}

}  // namespace l0op
