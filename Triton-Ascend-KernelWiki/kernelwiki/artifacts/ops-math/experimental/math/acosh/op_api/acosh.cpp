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
 * @file acosh.cpp
 * @brief ACLNN L0 API 实现 - Acosh 算子
 *
 * 职责：形状推导、Kernel 调度
 */

#include "acosh.h"
#include "opdev/op_log.h"
#include "opdev/op_dfx.h"
#include "opdev/shape_utils.h"
#include "opdev/make_op_executor.h"

using namespace op;

namespace l0op {

OP_TYPE_REGISTER(Acosh);

static const std::initializer_list<op::DataType> AICORE_DTYPE_SUPPORT_LIST = {
    DataType::DT_FLOAT16, DataType::DT_FLOAT, DataType::DT_BF16};

static bool IsAiCoreSupport(const aclTensor* self)
{
    auto npuArch = GetCurrentPlatformInfo().GetCurNpuArch();
    if (npuArch == NpuArch::DAV_2201 || npuArch == NpuArch::DAV_3510) {
        return CheckType(self->GetDataType(), AICORE_DTYPE_SUPPORT_LIST);
    }
    return false;
}

// 逐元素算子：输出 shape = 输入 shape
static bool AcoshInferShape(const op::Shape& selfShape, op::Shape& outShape)
{
    outShape = selfShape;
    return true;
}

static const aclTensor* AcoshAiCore(const aclTensor* self, const aclTensor* out, aclOpExecutor* executor)
{
    L0_DFX(AcoshAiCore, self, out);

    auto ret = ADD_TO_LAUNCHER_LIST_AICORE(Acosh, OP_INPUT(self), OP_OUTPUT(out));
    OP_CHECK(ret == ACLNN_SUCCESS, OP_LOGE(ACLNN_ERR_INNER_NULLPTR, "AcoshAiCore failed."), return nullptr);
    return out;
}

/**
 * @brief L0 API 入口
 */
const aclTensor* Acosh(const aclTensor* self, aclOpExecutor* executor)
{
    Shape outShape;

    if (!AcoshInferShape(self->GetViewShape(), outShape)) {
        OP_LOGE(ACLNN_ERR_PARAM_INVALID, "Acosh: InferShape failed.");
        return nullptr;
    }

    if (!IsAiCoreSupport(self)) {
        OP_LOGE(
            ACLNN_ERR_PARAM_INVALID,
            "Acosh not supported: dtype=%d. "
            "Supported dtypes: FLOAT16, FLOAT, BF16 (Ascend950).",
            static_cast<int>(self->GetDataType()));
        return nullptr;
    }

    const aclTensor* out = executor->AllocTensor(outShape, self->GetDataType());

    return AcoshAiCore(self, out, executor);
}

} // namespace l0op
