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
 *
 * NOTE: Portions of this code were AI-generated and have been
 * technically reviewed for functional accuracy and security
 */

/**
 * @file mul_no_nan.cpp
 * @brief ACLNN L0 API implementation for MulNoNan
 */

#include "mul_no_nan.h"
#include "opdev/op_log.h"
#include "opdev/op_dfx.h"
#include "opdev/shape_utils.h"
#include "opdev/make_op_executor.h"
#include "opdev/platform.h"

using namespace op;

namespace l0op {

OP_TYPE_REGISTER(MulNoNan);

static const std::initializer_list<op::DataType> AICORE_DTYPE_SUPPORT_LIST = {
    DataType::DT_FLOAT16, DataType::DT_FLOAT, DataType::DT_BF16};

static bool IsAiCoreSupport(const aclTensor* x, const aclTensor* y)
{
    auto npuArch = GetCurrentPlatformInfo().GetCurNpuArch();
    OP_CHECK(
        npuArch == NpuArch::DAV_3510,
        OP_LOGE(
            ACLNN_ERR_PARAM_INVALID, "MulNoNan not supported on this platform: npuArch=%d.", static_cast<int>(npuArch)),
        return false);
    OP_CHECK(
        CheckType(x->GetDataType(), AICORE_DTYPE_SUPPORT_LIST) &&
            CheckType(y->GetDataType(), AICORE_DTYPE_SUPPORT_LIST),
        OP_LOGE(
            ACLNN_ERR_PARAM_INVALID,
            "MulNoNan not supported: dtype x=%d, y=%d. Supported dtypes: FLOAT16, FLOAT, BFLOAT16.",
            static_cast<int>(x->GetDataType()), static_cast<int>(y->GetDataType())),
        return false);
    return true;
}

static bool MulNoNanInferShape(const op::Shape& xShape, const op::Shape& yShape, op::Shape& outShape)
{
    OP_CHECK(
        BroadcastInferShape(xShape, yShape, outShape), OP_LOGE(ACLNN_ERR_PARAM_INVALID, "Shape broadcast failed."),
        return false);
    return true;
}

static const aclTensor* MulNoNanAiCore(
    const aclTensor* x, const aclTensor* y, const aclTensor* out, aclOpExecutor* executor)
{
    L0_DFX(MulNoNanAiCore, x, y, out);

    auto ret = ADD_TO_LAUNCHER_LIST_AICORE(MulNoNan, OP_INPUT(x, y), OP_OUTPUT(out));
    OP_CHECK(ret == ACLNN_SUCCESS, OP_LOGE(ACLNN_ERR_INNER_NULLPTR, "MulNoNanAiCore failed."), return nullptr);
    return out;
}

const aclTensor* MulNoNan(const aclTensor* x, const aclTensor* y, aclOpExecutor* executor)
{
    Shape outShape;
    const aclTensor* out = nullptr;

    OP_CHECK(
        MulNoNanInferShape(x->GetViewShape(), y->GetViewShape(), outShape),
        OP_LOGE(ACLNN_ERR_PARAM_INVALID, "Infer shape failed."), return nullptr);

    OP_CHECK(IsAiCoreSupport(x, y), OP_LOGE(ACLNN_ERR_PARAM_INVALID, "IsAiCoreSupport check failed."), return nullptr);

    out = executor->AllocTensor(outShape, x->GetDataType());
    OP_CHECK(out != nullptr, OP_LOGE(ACLNN_ERR_INNER_NULLPTR, "MulNoNan: AllocTensor failed."), return nullptr);

    return MulNoNanAiCore(x, y, out, executor);
}

} // namespace l0op
