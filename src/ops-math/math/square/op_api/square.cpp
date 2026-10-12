/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include "square.h"
#include "opdev/aicpu/aicpu_task.h"
#include "opdev/make_op_executor.h"
#include "opdev/op_def.h"
#include "opdev/op_dfx.h"
#include "opdev/op_executor.h"
#include "opdev/op_log.h"
#include "opdev/shape_utils.h"
#include "opdev/platform.h"
#include "aclnn_kernels/common/op_error_check.h"
#include "opdev/make_op_executor.h"
#include "op_api/aclnn_check.h"

using namespace op;

namespace l0op {

OP_TYPE_REGISTER(Square);

static const std::initializer_list<op::DataType> AICORE_DTYPE_SUPPORT_LIST = {
    op::DataType::DT_FLOAT, op::DataType::DT_FLOAT16, op::DataType::DT_INT32, op::DataType::DT_BF16};

static const std::initializer_list<op::DataType> AICORE_DTYPE_SUPPORT_LIST_AFTER_910B = {
    op::DataType::DT_FLOAT, op::DataType::DT_FLOAT16, op::DataType::DT_INT32, op::DataType::DT_BF16,
    op::DataType::DT_INT64};

// 根据芯片类型、dtype判断算子是否支持走aicore
static inline bool IsAiCoreSupport(DataType inputDtype)
{
    auto socVersion = GetCurrentPlatformInfo().GetSocVersion();
    if (socVersion == SocVersion::ASCEND910B || socVersion == SocVersion::ASCEND910_93 ||
        IsRegBase()) {
        return CheckType(inputDtype, AICORE_DTYPE_SUPPORT_LIST_AFTER_910B);
    }
    return CheckType(inputDtype, AICORE_DTYPE_SUPPORT_LIST);
}

const aclTensor* SquareAiCpu(const aclTensor* self, aclTensor* squareOut, aclOpExecutor* executor)
{
    L0_DFX(SquareAiCpu, self, squareOut);
    static internal::AicpuTaskSpace space("Square");
    auto ret = ADD_TO_LAUNCHER_LIST_AICPU(Square, OP_ATTR_NAMES(), OP_INPUT(self), OP_OUTPUT(squareOut));
    CHECK_RET(ret == ACLNN_SUCCESS, nullptr);
    return squareOut;
}

const aclTensor* SquareAiCore(const aclTensor* self, aclTensor* squareOut, aclOpExecutor* executor)
{
    L0_DFX(SquareAiCore, self);
    auto retAicore = ADD_TO_LAUNCHER_LIST_AICORE(Square, OP_INPUT(self), OP_OUTPUT(squareOut));
    OP_CHECK_ADD_TO_LAUNCHER_LIST_AICORE(
        retAicore != ACLNN_SUCCESS, return nullptr, "Square ADD_TO_LAUNCHER_LIST_AICORE failed.");
    return squareOut;
}

const aclTensor* Square(const aclTensor* self, aclOpExecutor* executor)
{
    auto squareOut = executor->AllocTensor(self->GetStorageShape(), self->GetDataType());
    CHECK_RET(squareOut != nullptr, nullptr);
    if (IsAiCoreSupport(self->GetDataType())) {
        return SquareAiCore(self, squareOut, executor);
    } else {
        return SquareAiCpu(self, squareOut, executor);
    }
}

} // namespace l0op
