/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include "logical_and.h"
#include "opdev/aicpu/aicpu_task.h"
#include "opdev/make_op_executor.h"
#include "opdev/op_def.h"
#include "opdev/op_dfx.h"
#include "opdev/op_executor.h"
#include "opdev/op_log.h"
#include "opdev/shape_utils.h"

using namespace op;

namespace l0op {

OP_TYPE_REGISTER(LogicalAnd);

static const std::initializer_list<op::DataType> AICORE_DTYPE_SUPPORT_LIST = {op::DataType::DT_BOOL};

// 根据芯片类型、dtype判断算子是否支持走AiCore
static bool IsAiCoreSupport(const aclTensor* self)
{
    // LogicalAnd只需要判断dtype
    return CheckType(self->GetDataType(), AICORE_DTYPE_SUPPORT_LIST);
}

// AICORE算子kernel
static const aclTensor* LogicalAndAiCore(
    const aclTensor* self, const aclTensor* other, aclTensor* logical_andOut, aclOpExecutor* executor)
{
    L0_DFX(LogicalAndAiCore, self, other, logical_andOut);
    // 使用框架宏ADD_TO_LAUNCHER_LIST_AICORE，将AiCore LogicalAnd算子加入任务队列
    // LogicalAnd是算子的OpType，self、other是算子的输入，logical_andOut是算子的输出
    auto ret = ADD_TO_LAUNCHER_LIST_AICORE(LogicalAnd, OP_INPUT(self, other), OP_OUTPUT(logical_andOut));
    OP_CHECK(
        ret == ACLNN_SUCCESS, OP_LOGE(ACLNN_ERR_INNER_NULLPTR, "LogicalAndAiCore ADD_TO_LAUNCHER_LIST_AICORE failed."),
        return nullptr);
    return logical_andOut;
}

// AICPU算子kernel
static const aclTensor* LogicalAndAiCpu(
    const aclTensor* self, const aclTensor* other, aclTensor* logical_andOut, aclOpExecutor* executor)
{
    // 使用框架宏ADD_TO_LAUNCHER_LIST_AICPU，将AiCpu LogicalAnd算子加入任务队列
    // LogicalAnd是算子的OpType，self、other是算子的输入，logical_andOut是算子的输出
    L0_DFX(LogicalAndAiCpu, self, other);

    static internal::AicpuTaskSpace space("LogicalAnd");
    auto ret =
        ADD_TO_LAUNCHER_LIST_AICPU(LogicalAnd, OP_ATTR_NAMES(), OP_INPUT(self, other), OP_OUTPUT(logical_andOut));
    OP_CHECK(
        ret == ACLNN_SUCCESS, OP_LOGE(ACLNN_ERR_INNER_NULLPTR, "LogicalAndAiCpu ADD_TO_LAUNCHER_LIST_AICPU failed."),
        return nullptr);
    return logical_andOut;
}

const aclTensor* LogicalAnd(const aclTensor* self, const aclTensor* other, aclOpExecutor* executor)
{
    op::Shape broadcastShape;
    if (!BroadcastInferShape(self->GetViewShape(), other->GetViewShape(), broadcastShape)) {
        OP_LOGE(
            ACLNN_ERR_PARAM_INVALID, "Broadcast %s and %s failed.", op::ToString(self->GetViewShape()).GetString(),
            op::ToString(other->GetViewShape()).GetString());
        return nullptr;
    }

    auto logical_andOut = executor->AllocTensor(broadcastShape, self->GetDataType());
    if (IsAiCoreSupport(self)) {
        return LogicalAndAiCore(self, other, logical_andOut, executor);
    } else {
        return LogicalAndAiCpu(self, other, logical_andOut, executor);
    }
}

} // namespace l0op
