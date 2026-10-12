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
 * @file ndtri.cpp
 * @brief ACLNN L0 API 实现 - Ndtri
 */

#include "ndtri.h"
#include "opdev/op_log.h"
#include "opdev/op_dfx.h"
#include "opdev/shape_utils.h"
#include "opdev/make_op_executor.h"
#include "opdev/platform.h"

using namespace op;

namespace l0op {

OP_TYPE_REGISTER(Ndtri);

static const std::initializer_list<op::DataType> AICORE_DTYPE_SUPPORT_LIST = {
    DataType::DT_FLOAT, DataType::DT_FLOAT16, DataType::DT_BF16};

static bool IsAiCoreSupport(const aclTensor* self)
{
    auto npuArch = GetCurrentPlatformInfo().GetCurNpuArch();
    // 本期仅支持 Ascend950 (DAV_3510 / arch35)
    OP_CHECK(
        npuArch == NpuArch::DAV_3510,
        OP_LOGE(
            ACLNN_ERR_PARAM_INVALID, "Ndtri only supported on Ascend950 (DAV_3510): npuArch=%d.",
            static_cast<int>(npuArch)),
        return false);
    OP_CHECK(
        CheckType(self->GetDataType(), AICORE_DTYPE_SUPPORT_LIST),
        OP_LOGE(
            ACLNN_ERR_PARAM_INVALID, "Ndtri unsupported dtype: %d. Supported: FLOAT, FLOAT16, BF16.",
            static_cast<int>(self->GetDataType())),
        return false);
    return true;
}

const aclTensor* Ndtri(const aclTensor* self, aclOpExecutor* executor)
{
    L0_DFX(Ndtri, self);

    if (!IsAiCoreSupport(self)) {
        return nullptr;
    }

    const aclTensor* out = executor->AllocTensor(self->GetViewShape(), self->GetDataType());
    if (out == nullptr) {
        OP_LOGE(ACLNN_ERR_INNER_NULLPTR, "Ndtri: AllocTensor for output failed.");
        return nullptr;
    }

    auto ret = ADD_TO_LAUNCHER_LIST_AICORE(Ndtri, OP_INPUT(self), OP_OUTPUT(out));
    if (ret != ACLNN_SUCCESS) {
        OP_LOGE(ACLNN_ERR_INNER_NULLPTR, "Ndtri AiCore launch failed.");
        return nullptr;
    }

    return out;
}

} // namespace l0op
