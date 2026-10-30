/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include "sign.h"

#include "opdev/aicpu/aicpu_task.h"
#include "opdev/make_op_executor.h"
#include "opdev/op_def.h"
#include "opdev/op_dfx.h"
#include "opdev/op_executor.h"
#include "opdev/op_log.h"
#include "opdev/shape_utils.h"
#include "aclnn_kernels/common/op_error_check.h"
#include "opdev/platform.h"
#include "op_api/aclnn_check.h"

using namespace op;
namespace l0op {
OP_TYPE_REGISTER(Sign);
static const std::initializer_list<op::DataType> AICORE_DTYPE_SUPPORT_LIST = {
    op::DataType::DT_FLOAT, op::DataType::DT_FLOAT16, op::DataType::DT_INT32, op::DataType::DT_BF16};

static const std::initializer_list<op::DataType> AICORE_DTYPE_SUPPORT_LIST_ARCH3510 = {
    op::DataType::DT_FLOAT, op::DataType::DT_FLOAT16, op::DataType::DT_INT32, op::DataType::DT_BF16,
    op::DataType::DT_INT64};

static bool IsAiCoreSupport(const aclTensor* self)
{
    if (IsRegBase()) {
        return CheckType(self->GetDataType(), AICORE_DTYPE_SUPPORT_LIST_ARCH3510);
    }
    return CheckType(self->GetDataType(), AICORE_DTYPE_SUPPORT_LIST);
}

static const aclTensor* SignAiCore(const aclTensor* self, aclTensor* result, aclOpExecutor* executor)
{
    L0_DFX(SignAiCore, self, result);

    auto retAicore = ADD_TO_LAUNCHER_LIST_AICORE(Sign, OP_INPUT(self), OP_OUTPUT(result));
    OP_CHECK_ADD_TO_LAUNCHER_LIST_AICORE(
        retAicore != ACLNN_SUCCESS, return nullptr, "Sign ADD_TO_LAUNCHER_LIST_AICORE failed.");
    return result;
}

static const aclTensor* SignAiCpu(const aclTensor* self, aclTensor* result, aclOpExecutor* executor)
{
    L0_DFX(SignAiCpu, self, result);

    static internal::AicpuTaskSpace space("Sign", ge::DEPEND_IN_SHAPE, true);
    auto ret = ADD_TO_LAUNCHER_LIST_AICPU(Sign, OP_ATTR_NAMES(), OP_INPUT(self), OP_OUTPUT(result));
    CHECK_RET(ret == ACLNN_SUCCESS, nullptr);
    return result;
}

const aclTensor* Sign(const aclTensor* self, aclOpExecutor* executor)
{
    auto result = executor->AllocTensor(self->GetViewShape(), self->GetDataType());
    if (IsAiCoreSupport(self)) {
        return SignAiCore(self, result, executor);
    } else {
        return SignAiCpu(self, result, executor);
    }
}
} // namespace l0op
