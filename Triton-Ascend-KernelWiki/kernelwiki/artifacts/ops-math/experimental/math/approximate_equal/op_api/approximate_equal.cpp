/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

/**
 * NOTE: Portions of this code were AI-generated and have been
 * technically reviewed for functional accuracy and security
 */

#include "approximate_equal.h"
#include "opdev/op_log.h"
#include "opdev/op_dfx.h"
#include "opdev/shape_utils.h"
#include "opdev/make_op_executor.h"

using namespace op;

namespace l0op {

OP_TYPE_REGISTER(ApproximateEqual);

static const std::initializer_list<op::DataType> AICORE_DTYPE_SUPPORT_LIST = {
    DataType::DT_FLOAT,
    DataType::DT_FLOAT16,
    DataType::DT_BF16,
};

static bool IsAiCoreSupport(const aclTensor* x1, const aclTensor* x2)
{
    return CheckType(x1->GetDataType(), AICORE_DTYPE_SUPPORT_LIST) &&
           CheckType(x2->GetDataType(), AICORE_DTYPE_SUPPORT_LIST);
}

static const aclTensor* ApproximateEqualAiCore(
    const aclTensor* x1, const aclTensor* x2, const aclTensor* out, float tolerance, aclOpExecutor* executor)
{
    L0_DFX(ApproximateEqualAiCore, x1, x2, out, tolerance);
    auto ret = ADD_TO_LAUNCHER_LIST_AICORE(ApproximateEqual, OP_INPUT(x1, x2), OP_OUTPUT(out), OP_ATTR(tolerance));
    OP_CHECK(
        ret == ACLNN_SUCCESS, OP_LOGE(ACLNN_ERR_INNER_NULLPTR, "ApproximateEqualAiCore launch failed."),
        return nullptr);
    return out;
}

const aclTensor* ApproximateEqual(const aclTensor* x1, const aclTensor* x2, float tolerance, aclOpExecutor* executor)
{
    // y shape equals x1 shape (no broadcast).
    Shape outShape = x1->GetViewShape();

    if (!IsAiCoreSupport(x1, x2)) {
        OP_LOGE(
            ACLNN_ERR_PARAM_INVALID,
            "ApproximateEqual not supported: x1 dtype=%d, x2 dtype=%d (allowed: FLOAT / FLOAT16 / BF16).",
            static_cast<int>(x1->GetDataType()), static_cast<int>(x2->GetDataType()));
        return nullptr;
    }

    const aclTensor* out = executor->AllocTensor(outShape, DataType::DT_BOOL);
    if (out == nullptr) {
        return nullptr;
    }

    return ApproximateEqualAiCore(x1, x2, out, tolerance, executor);
}

} // namespace l0op
