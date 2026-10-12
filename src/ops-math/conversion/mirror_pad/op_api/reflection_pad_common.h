/**
 * Copyright (c) 2025-2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */
#ifndef REFLECTION_PAD_COMMON_H_
#define REFLECTION_PAD_COMMON_H_

#include "mirrorpad.h"
#include "conversion/pad_v3/op_api/padv3.h"
#include "conversion/unsqueeze/op_host/op_api/unsqueeze.h"
#include "conversion/squeeze/op_host/op_api/squeeze.h"
#include "aclnn_kernels/reshape.h"
#include "aclnn_kernels/common/op_error_check.h"
#include "opdev/op_dfx.h"

namespace op {
const string REFLECTION_MODE = "REFLECT";
inline const aclTensor* GetPaddingTensor(int64_t dim, const aclIntArray* padding, aclOpExecutor* executor)
{
    FVector<int64_t, op::MAX_DIM_NUM> paddingsVector;
    // 2 is the magnification
    for (size_t i = 2 * static_cast<size_t>(dim); i > 0; i -= 2) {
        if (i <= static_cast<size_t>(padding->Size())) {
            // 2 and 1 indicate the element of padding is put into paddingsVector from the back to the front
            paddingsVector.emplace_back((*padding)[i - 2]);
            paddingsVector.emplace_back((*padding)[i - 1]);
        } else {
            paddingsVector.emplace_back(0);
            paddingsVector.emplace_back(0);
        }
    }
    // 2 is the magnification
    auto newpadding = executor->AllocIntArray(paddingsVector.data(), 2 * dim);
    auto paddingsTensor = executor->ConvertToTensor(newpadding, static_cast<op::DataType>(ACL_INT64));
    return paddingsTensor;
}

inline aclnnStatus ProcessPadV3(const aclTensor* selfContiguous, const aclIntArray* padding,
                                const aclTensor*& padResult, int64_t limitDim, aclOpExecutor* executor)
{
    auto dim = static_cast<int64_t>(selfContiguous->GetViewShape().GetDimNum());
    auto dimCp = dim;
    if (dimCp == limitDim) {
        // 0 is index
        const int64_t appendDim[] = {0};
        // 1 is the dim num to be unsqueezed
        aclIntArray* dimArray = executor->AllocIntArray(appendDim, 1);
        selfContiguous = l0op::UnsqueezeNd(selfContiguous, dimArray, executor);
        CHECK_RET(selfContiguous != nullptr, ACLNN_ERR_INNER_NULLPTR);
        dim += 1;
    }
    auto paddingsTensor = GetPaddingTensor(dim, padding, executor);
    aclScalar* constantValueScalar = (executor)->AllocScalar(0);
    auto constantValueTensor = (executor)->ConvertToTensor(constantValueScalar, selfContiguous->GetDataType());
    padResult = l0op::PadV3(selfContiguous, paddingsTensor, constantValueTensor, "reflect", true, executor);
    CHECK_RET(padResult != nullptr, ACLNN_ERR_INNER_NULLPTR);
    // 2 is dim num
    if (dimCp == limitDim) {
        const int64_t appendDim[] = {0};
        aclIntArray* dimArray = (executor)->AllocIntArray(appendDim, 1);
        padResult = l0op::SqueezeNd(padResult, dimArray, executor);
        CHECK_RET(padResult != nullptr, ACLNN_ERR_INNER_NULLPTR);
    }
    return ACLNN_SUCCESS;
}

inline aclnnStatus ProcessMirrorPad(const aclTensor* selfContiguous, const aclIntArray* padding,
                                    const aclTensor*& padResult, aclOpExecutor* executor)
{
    auto dim = static_cast<int64_t>(selfContiguous->GetViewShape().GetDimNum());
    auto paddingsTensor = GetPaddingTensor(dim, padding, executor);
    // 2 is the column number
    op::Shape newShape = {dim, 2};
    paddingsTensor = l0op::Reshape(paddingsTensor, newShape, executor);
    padResult = l0op::MirrorPad(selfContiguous, paddingsTensor, REFLECTION_MODE, executor);
    CHECK_RET(padResult != nullptr, ACLNN_ERR_INNER_NULLPTR);
    return ACLNN_SUCCESS;
}

} // namespace op
#endif // REFLECTION_PAD_COMMON_H_
