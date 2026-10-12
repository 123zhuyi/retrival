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

/**
 * @file asinh_grad.cpp
 * @brief ACLNN L0 API implementation for AsinhGrad
 *
 * L0 API: shape inference, kernel dispatch
 */

#include "asinh_grad.h"
#include "opdev/op_log.h"
#include "opdev/op_dfx.h"
#include "opdev/shape_utils.h"
#include "opdev/make_op_executor.h"

using namespace op;

namespace l0op {

OP_TYPE_REGISTER(AsinhGrad);

static const std::initializer_list<op::DataType> AICORE_DTYPE_SUPPORT_LIST = {
    DataType::DT_FLOAT, DataType::DT_FLOAT16, DataType::DT_BF16};

static bool IsAiCoreSupport(const aclTensor* y, const aclTensor* dy)
{
    return CheckType(y->GetDataType(), AICORE_DTYPE_SUPPORT_LIST) &&
           CheckType(dy->GetDataType(), AICORE_DTYPE_SUPPORT_LIST);
}

static bool AsinhGradInferShape(const op::Shape& yShape, const op::Shape& /*dyShape*/, op::Shape& outShape)
{
    // AsinhGrad: output shape = input y shape (no broadcast)
    outShape = yShape;
    return true;
}

static const aclTensor* AsinhGradAiCore(
    const aclTensor* y, const aclTensor* dy, const aclTensor* out, aclOpExecutor* executor)
{
    L0_DFX(AsinhGradAiCore, y, dy, out);

    auto ret = ADD_TO_LAUNCHER_LIST_AICORE(AsinhGrad, OP_INPUT(y, dy), OP_OUTPUT(out));
    OP_CHECK(ret == ACLNN_SUCCESS, OP_LOGE(ACLNN_ERR_INNER_NULLPTR, "AsinhGradAiCore failed."), return nullptr);
    return out;
}

const aclTensor* AsinhGrad(const aclTensor* y, const aclTensor* dy, aclOpExecutor* executor)
{
    Shape outShape;
    const aclTensor* out = nullptr;

    if (!AsinhGradInferShape(y->GetViewShape(), dy->GetViewShape(), outShape)) {
        OP_LOGE(ACLNN_ERR_PARAM_INVALID, "AsinhGrad: infer shape failed.");
        return nullptr;
    }

    if (!IsAiCoreSupport(y, dy)) {
        OP_LOGE(
            ACLNN_ERR_PARAM_INVALID, "AsinhGrad: dtype not supported: y=%d, dy=%d.", static_cast<int>(y->GetDataType()),
            static_cast<int>(dy->GetDataType()));
        return nullptr;
    }

    out = executor->AllocTensor(outShape, y->GetDataType());

    return AsinhGradAiCore(y, dy, out, executor);
}

} // namespace l0op
