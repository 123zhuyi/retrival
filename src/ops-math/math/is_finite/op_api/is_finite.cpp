/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include "is_finite.h"
#include "opdev/make_op_executor.h"
#include "opdev/aicpu/aicpu_task.h"
#include "opdev/op_dfx.h"
#include "opdev/platform.h"
#include "op_api/aclnn_check.h"

using namespace op;

namespace l0op {
OP_TYPE_REGISTER(IsFinite);

static const std::initializer_list<op::DataType> AICORE_DTYPE_SUPPORT_LIST = {
    op::DataType::DT_FLOAT, op::DataType::DT_FLOAT16, op::DataType::DT_BF16};

static const std::initializer_list<op::DataType> AICORE_310P_DTYPE_SUPPORT_LIST = {
    op::DataType::DT_FLOAT, op::DataType::DT_FLOAT16};

const aclTensor* IsFiniteAiCore(const aclTensor* self, aclTensor* out, aclOpExecutor* executor)
{
    L0_DFX(IsFiniteAiCore, self, out);

    auto ret = ADD_TO_LAUNCHER_LIST_AICORE(IsFinite, OP_INPUT(self), OP_OUTPUT(out));
    OP_CHECK(
        ret == ACLNN_SUCCESS, OP_LOGE(ACLNN_ERR_INNER_NULLPTR, "IsFiniteAiCore ADD_TO_LAUNCHER_LIST_AICORE failed."),
        return nullptr);
    return out;
}

const aclTensor* IsFiniteAiCpu(const aclTensor* self, aclTensor* out, aclOpExecutor* executor)
{
    L0_DFX(IsFiniteAiCpu, self, out);

    static internal::AicpuTaskSpace space("IsFinite", ge::DEPEND_IN_SHAPE, true);
    auto ret = ADD_TO_LAUNCHER_LIST_AICPU(
        IsFinite, OP_ATTR_NAMES({"T"}), OP_INPUT(self), OP_OUTPUT(out), OP_ATTR(self->GetDataType()));
    OP_CHECK(
        ret == ACLNN_SUCCESS, OP_LOGE(ACLNN_ERR_INNER_NULLPTR, "IsFiniteAiCpu ADD_TO_LAUNCHER_LIST_AICPU failed."),
        return nullptr);
    return out;
}

const aclTensor* IsFinite(const aclTensor* self, aclOpExecutor* executor)
{
    auto out = executor->AllocTensor(self->GetViewShape(), op::DataType::DT_BOOL, op::Format::FORMAT_ND);
    auto npuArch = op::GetCurrentPlatformInfo().GetCurNpuArch();
    bool socSupported =
        (npuArch == NpuArch::DAV_2201 || IsRegBase(npuArch));
    if (socSupported && CheckType(self->GetDataType(), AICORE_DTYPE_SUPPORT_LIST)) {
        return IsFiniteAiCore(self, out, executor);
    } else if ((npuArch == NpuArch::DAV_2002) && CheckType(self->GetDataType(), AICORE_310P_DTYPE_SUPPORT_LIST)) {
        return IsFiniteAiCore(self, out, executor);
    } else {
        return IsFiniteAiCpu(self, out, executor);
    }
}
} // namespace l0op
