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
 * @file atan_grad.cpp
 * @brief ACLNN L0 API 实现 - AtanGrad 算子
 *
 * 职责：形状推导、Kernel 调度
 *
 * 流程：
 *   1. InferShape      - 输出 shape = 输入 shape（逐元素算子）
 *   2. IsAiCoreSupport - 确认 dtype 和平台支持
 *   3. AllocTensor     - 分配输出 Tensor
 *   4. AtanGradAiCore  - 调用 Kernel
 */

#include "atan_grad.h"
#include "opdev/op_log.h"
#include "opdev/op_dfx.h"
#include "opdev/shape_utils.h"
#include "opdev/make_op_executor.h"

using namespace op;

namespace l0op {

OP_TYPE_REGISTER(AtanGrad);

static const std::initializer_list<op::DataType> ATAN_GRAD_AICORE_DTYPE_LIST = {
    DataType::DT_FLOAT16, DataType::DT_FLOAT, DataType::DT_BF16};

static bool IsAiCoreSupport(const aclTensor* x, const aclTensor* dy)
{
    return CheckType(x->GetDataType(), ATAN_GRAD_AICORE_DTYPE_LIST) &&
           CheckType(dy->GetDataType(), ATAN_GRAD_AICORE_DTYPE_LIST);
}

static bool AtanGradInferShape(const op::Shape& xShape, op::Shape& outShape)
{
    outShape = xShape;
    return true;
}

static const aclTensor* AtanGradAiCore(
    const aclTensor* x, const aclTensor* dy, const aclTensor* out, aclOpExecutor* executor)
{
    L0_DFX(AtanGradAiCore, x, dy, out);

    auto ret = ADD_TO_LAUNCHER_LIST_AICORE(AtanGrad, OP_INPUT(x, dy), OP_OUTPUT(out));
    OP_CHECK(ret == ACLNN_SUCCESS, OP_LOGE(ACLNN_ERR_INNER_NULLPTR, "AtanGradAiCore failed."), return nullptr);
    return out;
}

const aclTensor* AtanGrad(const aclTensor* x, const aclTensor* dy, aclOpExecutor* executor)
{
    Shape outShape;
    if (!AtanGradInferShape(x->GetViewShape(), outShape)) {
        OP_LOGE(ACLNN_ERR_PARAM_INVALID, "AtanGrad: InferShape failed.");
        return nullptr;
    }

    if (!IsAiCoreSupport(x, dy)) {
        OP_LOGE(
            ACLNN_ERR_PARAM_INVALID, "AtanGrad: unsupported dtype x=%d, dy=%d. Supported: FLOAT16, FLOAT, BF16.",
            static_cast<int>(x->GetDataType()), static_cast<int>(dy->GetDataType()));
        return nullptr;
    }

    const aclTensor* out = executor->AllocTensor(outShape, x->GetDataType());

    return AtanGradAiCore(x, dy, out, executor);
}

} // namespace l0op
