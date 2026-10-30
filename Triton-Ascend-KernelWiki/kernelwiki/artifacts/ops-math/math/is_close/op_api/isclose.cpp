/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include "opdev/op_dfx.h"
#include "opdev/shape_utils.h"
#include "opdev/make_op_executor.h"
#include "opdev/aicpu/aicpu_task.h"
#include "isclose.h"

using namespace op;
namespace l0op {
OP_TYPE_REGISTER(IsClose);

static const std::initializer_list<op::DataType> AICORE_DTYPE_SUPPORT_LIST = {
    op::DataType::DT_FLOAT, op::DataType::DT_FLOAT16, op::DataType::DT_INT32};

static const std::initializer_list<op::DataType> AICORE_910B_DTYPE_SUPPORT_LIST = {
    op::DataType::DT_FLOAT, op::DataType::DT_FLOAT16, op::DataType::DT_INT32, op::DataType::DT_BF16};

static bool IsAiCoreSupport(const aclTensor* self)
{
    if (GetCurrentPlatformInfo().GetSocVersion() >= SocVersion::ASCEND910B ||
        GetCurrentPlatformInfo().GetSocVersion() <= SocVersion::ASCEND910E) {
        return CheckType(self->GetDataType(), AICORE_910B_DTYPE_SUPPORT_LIST);
    }
    return CheckType(self->GetDataType(), AICORE_DTYPE_SUPPORT_LIST);
}

static const aclTensor* IsCloseAiCore(
    const aclTensor* self, const aclTensor* other, float rtol, float atol, bool equal_nan, aclTensor* out,
    aclOpExecutor* executor)
{
    L0_DFX(IsCloseAiCore, self, other, rtol, atol, equal_nan, out);

    auto ret =
        ADD_TO_LAUNCHER_LIST_AICORE(IsClose, OP_INPUT(self, other), OP_OUTPUT(out), OP_ATTR(rtol, atol, equal_nan));
    OP_CHECK(
        ret == ACLNN_SUCCESS, OP_LOGE(ACLNN_ERR_INNER_NULLPTR, "IsCloseAiCore ADD_TO_LAUNCHER_LIST_AICORE failed."),
        return nullptr);
    return out;
}

static const aclTensor* IsCloseAiCpu(
    const aclTensor* self, const aclTensor* other, float rtol, float atol, bool equal_nan, aclTensor* out,
    aclOpExecutor* executor)
{
    L0_DFX(IsCloseAiCpu, self, other, rtol, atol, equal_nan, out);

    static internal::AicpuTaskSpace space("IsClose");
    auto ret = ADD_TO_LAUNCHER_LIST_AICPU(
        IsClose, OP_ATTR_NAMES({"rtol", "atol", "equal_nan"}), OP_INPUT(self, other), OP_OUTPUT(out),
        OP_ATTR(rtol, atol, equal_nan));
    OP_CHECK(
        ret == ACLNN_SUCCESS, OP_LOGE(ACLNN_ERR_INNER_NULLPTR, "IsCloseAiCpu ADD_TO_LAUNCHER_LIST_AICPU failed."),
        return nullptr);
    return out;
}

const aclTensor* IsClose(
    const aclTensor* self, const aclTensor* other, float rtol, float atol, bool equal_nan, aclTensor* out,
    aclOpExecutor* executor)
{
    auto outTensor = executor->AllocTensor(out->GetViewShape(), op::DataType::DT_BOOL, op::Format::FORMAT_ND);
    if (IsAiCoreSupport(self)) {
        return IsCloseAiCore(self, other, rtol, atol, equal_nan, outTensor, executor);
    } else {
        return IsCloseAiCpu(self, other, rtol, atol, equal_nan, outTensor, executor);
    }
}
} // namespace l0op
