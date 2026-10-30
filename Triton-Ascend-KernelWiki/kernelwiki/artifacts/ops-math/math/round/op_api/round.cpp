/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */
#include "round.h"
#include "opdev/op_log.h"
#include "opdev/shape_utils.h"
#include "opdev/op_def.h"
#include "opdev/op_dfx.h"
#include "opdev/aicpu/aicpu_task.h"
#include "aclnn_kernels/common/op_error_check.h"
#include "op_api/aclnn_check.h"

using namespace op;

namespace l0op {
OP_TYPE_REGISTER(Round);

static const std::initializer_list<op::DataType> AICORE_DTYPE_SUPPORT_LIST = {
    op::DataType::DT_FLOAT16, op::DataType::DT_FLOAT, op::DataType::DT_INT32};

static const std::initializer_list<op::DataType> ASCEND910B_AICORE_DTYPE_SUPPORT_LIST = {
    op::DataType::DT_FLOAT16, op::DataType::DT_FLOAT, op::DataType::DT_INT32, op::DataType::DT_BF16};

inline static bool IsAiCoreSupport(const aclTensor* self)
{
    if (GetCurrentPlatformInfo().GetSocVersion() == SocVersion::ASCEND910B ||
        GetCurrentPlatformInfo().GetSocVersion() == SocVersion::ASCEND910_93 ||
        IsRegBase()) {
        return CheckType(self->GetDataType(), ASCEND910B_AICORE_DTYPE_SUPPORT_LIST);
    }
    return CheckType(self->GetDataType(), AICORE_DTYPE_SUPPORT_LIST);
}

inline const aclTensor* RoundAiCore(const aclTensor* self, aclTensor* out, aclOpExecutor* executor)
{
    L0_DFX(RoundAiCore, self, out);
    auto retAicore = ADD_TO_LAUNCHER_LIST_AICORE(Round, OP_INPUT(self), OP_OUTPUT(out));
    OP_CHECK_ADD_TO_LAUNCHER_LIST_AICORE(
        retAicore != ACLNN_SUCCESS, return nullptr, "Round ADD_TO_LAUNCHER_LIST_AICORE failed.");
    return out;
}

inline const aclTensor* RoundAiCpu(const aclTensor* self, aclTensor* out, aclOpExecutor* executor)
{
    L0_DFX(RoundAiCpu, self, out);
    static internal::AicpuTaskSpace space("Round");
    auto ret = ADD_TO_LAUNCHER_LIST_AICPU(Round, OP_ATTR_NAMES(), OP_INPUT(self), OP_OUTPUT(out));
    CHECK_RET(ret == ACLNN_SUCCESS, nullptr);
    return out;
}

const aclTensor* Round(const aclTensor* self, aclOpExecutor* executor)
{
    auto out = executor->AllocTensor(self->GetViewShape(), self->GetDataType(), self->GetViewFormat());
    if (IsAiCoreSupport(self)) {
        return RoundAiCore(self, out, executor);
    } else {
        return RoundAiCpu(self, out, executor);
    }
}

// AICORE算子kernel
static inline const aclTensor* RoundDecimalsAiCore(
    const aclTensor* self, int64_t decimals, aclTensor* output, aclOpExecutor* executor)
{
    L0_DFX(RoundDecimalsAiCore, self, decimals, output);
    // 使用框架宏ADD_TO_LAUNCHER_LIST_AICORE，将AiCore Round算子加入任务队列
    ADD_TO_LAUNCHER_LIST_AICORE(Round, OP_INPUT(self), OP_ATTR(decimals), OP_OUTPUT(output));
    return output;
}

// AICPU算子kernel
static inline const aclTensor* RoundDecimalsAiCpu(
    const aclTensor* self, int64_t decimals, aclTensor* output, aclOpExecutor* executor)
{
    L0_DFX(RoundDecimalsAiCpu, self, decimals, output);
    // 使用框架宏ADD_TO_LAUNCHER_LIST_AICPU，将AiCpu Round算子加入任务队列
    static internal::AicpuTaskSpace space("Round");
    auto ret = ADD_TO_LAUNCHER_LIST_AICPU(
        Round, OP_ATTR_NAMES({"decimals"}), OP_INPUT(self), OP_ATTR(decimals), OP_OUTPUT(output));
    CHECK_RET(ret == ACLNN_SUCCESS, nullptr);
    return output;
}

const aclTensor* RoundDecimals(const aclTensor* self, int64_t decimals, aclOpExecutor* executor)
{
    auto out = executor->AllocTensor(self->GetViewShape(), self->GetDataType(), self->GetViewFormat());
    if (IsAiCoreSupport(self)) {
        return RoundDecimalsAiCore(self, decimals, out, executor);
    } else {
        return RoundDecimalsAiCpu(self, decimals, out, executor);
    }
}
} // namespace l0op
