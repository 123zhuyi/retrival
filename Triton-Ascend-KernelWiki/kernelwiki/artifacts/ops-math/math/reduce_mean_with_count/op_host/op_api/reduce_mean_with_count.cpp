/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

/*!
 * \file reduce_mean_with_count.cpp
 * \brief L0 op implementation for ReduceMeanWithCount
 *
 * Prototype: INPUT(x, count, count_sum) -> OUTPUT(y), ATTR(axes, keep_dims)
 * Dispatches to AICore. Computation: y = ReduceSum(x * count / count_sum, axes)
 */

#include "reduce_mean_with_count.h"
#include "opdev/data_type_utils.h"
#include "opdev/format_utils.h"
#include "opdev/make_op_executor.h"
#include "opdev/op_def.h"
#include "opdev/op_dfx.h"
#include "opdev/op_executor.h"
#include "opdev/op_log.h"
#include "op_api/aclnn_check.h"
#include "opdev/shape_utils.h"

using namespace op;

namespace l0op {
OP_TYPE_REGISTER(ReduceMeanWithCount);

static op::Shape ComputeReduceOutputShape(const aclTensor* x, const aclIntArray* axes, bool keepDims)
{
    op::Shape yShape;
    auto viewShape = x->GetViewShape();
    auto dimNum = static_cast<int64_t>(viewShape.GetDimNum());

    // Normalize axes into a set
    std::set<int64_t> axisSet;
    if (axes == nullptr || axes->Size() == 0) {
        for (int64_t i = 0; i < dimNum; i++) {
            axisSet.insert(i);
        }
    } else {
        for (size_t i = 0; i < axes->Size(); i++) {
            int64_t ax = (*axes)[i];
            if (ax < 0) {
                ax += dimNum;
            }
            axisSet.insert(ax);
        }
    }

    // Build output shape
    for (int64_t d = 0; d < dimNum; d++) {
        if (axisSet.count(d)) {
            if (keepDims) {
                yShape.AppendDim(1);
            }
        } else {
            yShape.AppendDim(viewShape.GetDim(d));
        }
    }
    return yShape;
}

const aclTensor* ReduceMeanWithCount(const aclTensor* x, const aclTensor* count,
                                     const aclTensor* countSum, const aclIntArray* axes,
                                     bool keepDims, aclOpExecutor* executor)
{
    L0_DFX(ReduceMeanWithCount, x, count, countSum, axes, keepDims);

    const aclTensor* yOut = nullptr;
    if (IsRegBase()) {
        OP_LOGI("ReduceMeanWithCount use manual ComputeReduceOutputShape for RegBase.");
        auto yShape = ComputeReduceOutputShape(x, axes, keepDims);
        yOut = executor->AllocTensor(yShape, x->GetDataType(), x->GetStorageFormat());
    } else {
        OP_LOGI("ReduceMeanWithCount use INFER_SHAPE.");
        yOut = executor->AllocTensor(x->GetDataType(), op::Format::FORMAT_ND, op::Format::FORMAT_ND);
        INFER_SHAPE(ReduceMeanWithCount, OP_INPUT(x, count, countSum), OP_OUTPUT(yOut), OP_ATTR(axes, keepDims));
    }
    CHECK_RET(yOut != nullptr, nullptr);

    auto retAicore = ADD_TO_LAUNCHER_LIST_AICORE(
        ReduceMeanWithCount, OP_INPUT(x, count, countSum), OP_OUTPUT(yOut), OP_ATTR(axes, keepDims));
    OP_CHECK(retAicore == ACLNN_SUCCESS,
             OP_LOGE(ACLNN_ERR_INNER_NULLPTR, "ReduceMeanWithCount ADD_TO_LAUNCHER_LIST_AICORE failed."),
             return nullptr);

    return yOut;
}

}  // namespace l0op
