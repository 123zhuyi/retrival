/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

/**
 * NOTE: Portions of this code were AI-generated and have been
 * technically reviewed for functional accuracy and security
 */

/**
 * @file acos_grad.cpp
 * @brief ACLNN L0 API 实现 - AcosGrad 算子
 */

#include "acos_grad.h"
#include "opdev/op_log.h"
#include "opdev/op_dfx.h"
#include "opdev/shape_utils.h"
#include "opdev/make_op_executor.h"
#include "op_api/aclnn_check.h"

using namespace op;

namespace l0op {

OP_TYPE_REGISTER(AcosGrad);

static const std::initializer_list<op::DataType> ASCEND950_AICORE_DTYPE_SUPPORT_LIST = {
    DataType::DT_FLOAT16, DataType::DT_FLOAT, DataType::DT_BF16};

static bool IsAiCoreSupport(const aclTensor* y_grad, const aclTensor* x)
{
    auto npuArch = GetCurrentPlatformInfo().GetCurNpuArch();
    if (npuArch == NpuArch::DAV_2201) {
        return CheckType(y_grad->GetDataType(), ASCEND950_AICORE_DTYPE_SUPPORT_LIST) &&
               CheckType(x->GetDataType(), ASCEND950_AICORE_DTYPE_SUPPORT_LIST);
    }
    if (IsRegBase()) {
        return CheckType(y_grad->GetDataType(), ASCEND950_AICORE_DTYPE_SUPPORT_LIST) &&
               CheckType(x->GetDataType(), ASCEND950_AICORE_DTYPE_SUPPORT_LIST);
    }
    return false;
}

static bool AcosGradInferShape(const op::Shape& yGradShape, const op::Shape& xShape, op::Shape& outShape)
{
    if (yGradShape != xShape) {
        OP_LOGE(
            ACLNN_ERR_PARAM_INVALID, "AcosGrad: Shape mismatch: y_grad=%s, x=%s", op::ToString(yGradShape).GetString(),
            op::ToString(xShape).GetString());
        return false;
    }
    outShape = yGradShape;
    return true;
}

static const aclTensor* AcosGradAiCore(
    const aclTensor* y_grad, const aclTensor* x, const aclTensor* x_grad, aclOpExecutor* executor)
{
    L0_DFX(AcosGradAiCore, y_grad, x, x_grad);

    auto ret = ADD_TO_LAUNCHER_LIST_AICORE(AcosGrad, OP_INPUT(y_grad, x), OP_OUTPUT(x_grad));
    OP_CHECK(ret == ACLNN_SUCCESS, OP_LOGE(ACLNN_ERR_INNER_NULLPTR, "AcosGradAiCore failed."), return nullptr);
    return x_grad;
}

const aclTensor* AcosGrad(const aclTensor* y_grad, const aclTensor* x, aclOpExecutor* executor)
{
    Shape outShape;
    const aclTensor* out = nullptr;

    if (!AcosGradInferShape(y_grad->GetViewShape(), x->GetViewShape(), outShape)) {
        OP_LOGE(ACLNN_ERR_PARAM_INVALID, "AcosGrad: Infer shape failed.");
        return nullptr;
    }

    if (!IsAiCoreSupport(y_grad, x)) {
        OP_LOGE(
            ACLNN_ERR_PARAM_INVALID,
            "AcosGrad not supported: dtype y_grad=%d, x=%d. "
            "Supported dtypes: FLOAT16, FLOAT32, BF16.",
            static_cast<int>(y_grad->GetDataType()), static_cast<int>(x->GetDataType()));
        return nullptr;
    }

    out = executor->AllocTensor(outShape, y_grad->GetDataType());

    return AcosGradAiCore(y_grad, x, out, executor);
}

} // namespace l0op
