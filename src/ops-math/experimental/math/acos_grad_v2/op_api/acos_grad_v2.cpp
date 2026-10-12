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
 * @file acos_grad_v2.cpp
 * @brief ACLNN L0 API 实现 - AcosGradV2 算子 (A2 / Ascend910B)
 *
 * z = -dy / sqrt(1 - y^2)
 */

#include "acos_grad_v2.h"
#include "opdev/op_log.h"
#include "opdev/op_dfx.h"
#include "opdev/shape_utils.h"
#include "opdev/make_op_executor.h"
#include "op_api/aclnn_check.h"

using namespace op;

namespace l0op {

OP_TYPE_REGISTER(AcosGradV2);

static const std::initializer_list<op::DataType> ACOS_GRAD_V2_DTYPE_SUPPORT_LIST = {
    DataType::DT_FLOAT16, DataType::DT_FLOAT, DataType::DT_BF16};

static bool IsAiCoreSupport(const aclTensor* y, const aclTensor* dy)
{
    auto npuArch = GetCurrentPlatformInfo().GetCurNpuArch();
    // A2 (Atlas A2 训练/推理系列产品) -> DAV_2201
    if (npuArch == NpuArch::DAV_2201) {
        return CheckType(y->GetDataType(), ACOS_GRAD_V2_DTYPE_SUPPORT_LIST) &&
               CheckType(dy->GetDataType(), ACOS_GRAD_V2_DTYPE_SUPPORT_LIST);
    }
    return false;
}

static bool AcosGradV2InferShape(const op::Shape& yShape, const op::Shape& dyShape, op::Shape& outShape)
{
    if (yShape != dyShape) {
        OP_LOGE(ACLNN_ERR_PARAM_INVALID, "AcosGradV2: Shape mismatch: y=%s, dy=%s", op::ToString(yShape).GetString(),
                op::ToString(dyShape).GetString());
        return false;
    }
    outShape = yShape;
    return true;
}

static const aclTensor* AcosGradV2AiCore(const aclTensor* y, const aclTensor* dy, const aclTensor* z,
                                         aclOpExecutor* executor)
{
    L0_DFX(AcosGradV2AiCore, y, dy, z);

    auto ret = ADD_TO_LAUNCHER_LIST_AICORE(AcosGradV2, OP_INPUT(y, dy), OP_OUTPUT(z));
    OP_CHECK(ret == ACLNN_SUCCESS, OP_LOGE(ACLNN_ERR_INNER_NULLPTR, "AcosGradV2AiCore failed."), return nullptr);
    return z;
}

const aclTensor* AcosGradV2(const aclTensor* y, const aclTensor* dy, aclOpExecutor* executor)
{
    Shape outShape;
    const aclTensor* out = nullptr;

    if (!AcosGradV2InferShape(y->GetViewShape(), dy->GetViewShape(), outShape)) {
        OP_LOGE(ACLNN_ERR_PARAM_INVALID, "AcosGradV2: Infer shape failed.");
        return nullptr;
    }

    if (!IsAiCoreSupport(y, dy)) {
        OP_LOGE(ACLNN_ERR_PARAM_INVALID,
                "AcosGradV2 not supported: dtype y=%d, dy=%d. Supported dtypes: FLOAT16, FLOAT32, BF16.",
                static_cast<int>(y->GetDataType()), static_cast<int>(dy->GetDataType()));
        return nullptr;
    }

    out = executor->AllocTensor(outShape, y->GetDataType());
    OP_CHECK(out != nullptr, OP_LOGE(ACLNN_ERR_INNER_NULLPTR, "AcosGradV2: AllocTensor failed."), return nullptr);

    return AcosGradV2AiCore(y, dy, out, executor);
}

} // namespace l0op
