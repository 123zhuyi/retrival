/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */
#ifndef RANDOM_DROPOUT_COMMON_H
#define RANDOM_DROPOUT_COMMON_H

#include "random/drop_out_do_mask/op_api/dropout_do_mask.h"
#include "math/zero_op/op_api/zero_op.h"
#include "aclnn_kernels/common/op_error_check.h"
#include "opdev/op_executor.h"
#include "conversion/fill/op_api/fill.h"
#include <cmath>
#include <limits>

namespace op {

inline bool IsDoubleEqual(double lhs, double rhs)
{
    return std::abs(lhs - rhs) <= std::numeric_limits<double>::epsilon();
}

// Dropout DoMask公共实现
// inputContiguous: 连续化后的输入tensor
// mask: dropout mask tensor
// p: dropout概率
// executor: OpExecutor
static inline const aclTensor* DoMask(const aclTensor* inputContiguous, const aclTensor* mask, double p,
                                      aclOpExecutor* executor)
{
    if (IsDoubleEqual(p, 0.0)) {
        return inputContiguous;
    } else if (IsDoubleEqual(p, 1.0)) {
        return l0op::ZerosLike(inputContiguous, executor);
    } else {
        FVector<float> probVector = {static_cast<float>(1 - p)};
        auto probTensor = executor->ConvertToTensor(probVector.data(), probVector.size(),
                                                    inputContiguous->GetDataType());
        return l0op::DropoutDoMask(inputContiguous, mask, probTensor, executor);
    }
}

// FillScalar公共实现：用标量值填充mask tensor
// out: 输出tensor（提供shape和存储地址）
// val: 填充值
// executor: OpExecutor
static inline const aclTensor* FillScalar(const aclTensor* out, int8_t val, aclOpExecutor* executor)
{
    auto maskShape = out->GetViewShape();
    FVector<int64_t> maskShapeVector;
    for (size_t i = 0; i < maskShape.GetDimNum(); i++) {
        maskShapeVector.push_back(maskShape.GetDim(i));
    }
    auto dims = executor->ConvertToTensor(maskShapeVector.data(), maskShapeVector.size(), DataType::DT_INT64);
    auto shapeArray = executor->AllocIntArray(maskShapeVector.data(), maskShapeVector.size());

    FVector<int8_t> valVector = {val};
    auto valTensor = executor->ConvertToTensor(valVector.data(), valVector.size(), op::DataType::DT_INT8);
    auto mask = l0op::Fill(dims, valTensor, shapeArray, executor);
    CHECK_RET(mask != nullptr, nullptr);
    mask->SetFromWorkspace(false);
    mask->SetStorageAddr(out->GetStorageAddr());
    return out;
}

} // namespace op

#endif // RANDOM_DROPOUT_COMMON_H
