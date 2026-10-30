/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include "log1p.h"
#include "opdev/aicpu/aicpu_task.h"
#include "opdev/op_log.h"
#include "opdev/op_executor.h"
#include "opdev/make_op_executor.h"
#include "opdev/shape_utils.h"
#include "opdev/op_def.h"
#include "opdev/op_dfx.h"

using namespace op;

namespace l0op {
OP_TYPE_REGISTER(Log1p);

static const std::initializer_list<op::DataType> AICORE_DTYPE_SUPPORT_LIST = {
    op::DataType::DT_FLOAT, op::DataType::DT_FLOAT16, op::DataType::DT_BF16};

// 根据芯片类型、dtype判断算子是否支持走aicore
inline static bool IsAiCoreSupport(const aclTensor *self)
{
    // Log1p只需要判断dtype
    return CheckType(self->GetDataType(), AICORE_DTYPE_SUPPORT_LIST);
}

// AICORE算子kernel
inline static const aclTensor *Log1pAiCore(const aclTensor *self, aclTensor *log1pOut, aclOpExecutor *executor)
{
    L0_DFX(Log1pAiCore, self, log1pOut);
    // 使用框架宏ADD_TO_LAUNCHER_LIST_AICORE，将AiCore Log1p算子加入任务队列
    // Log1p是算子的OpType，self是算子的输入，log1pOut是算子的输出
    auto ret = ADD_TO_LAUNCHER_LIST_AICORE(Log1p, OP_INPUT(self), OP_OUTPUT(log1pOut));
    OP_CHECK(ret == ACLNN_SUCCESS, OP_LOGE(ACLNN_ERR_INNER_NULLPTR, "Log1pAiCore ADD_TO_LAUNCHER_LIST_AICORE failed."), return nullptr);
    return log1pOut;
}

// AICPU算子kernel
inline static const aclTensor *Log1pAiCpu(const aclTensor *self, aclTensor *log1pOut, aclOpExecutor *executor)
{
    // 使用框架宏ADD_TO_CPU_LAUNCHER_LIST，将AiCpu Log1p算子加入任务队列
    // Log1p是算子的OpType，self是算子的输入，log1pOut是算子的输出
    L0_DFX(Log1pAiCpu, self, log1pOut);

    static internal::AicpuTaskSpace space("Log1p", ge::DEPEND_IN_SHAPE, true);
    auto ret = ADD_TO_LAUNCHER_LIST_AICPU(Log1p, OP_ATTR_NAMES(), OP_INPUT(self), OP_OUTPUT(log1pOut));
    OP_CHECK(ret == ACLNN_SUCCESS, OP_LOGE(ACLNN_ERR_INNER_NULLPTR, "Log1pAiCpu ADD_TO_LAUNCHER_LIST_AICPU failed."), return nullptr);
    return log1pOut;
}

const aclTensor *Log1p(const aclTensor *self, aclOpExecutor *executor)
{
    auto log1pOut = executor->AllocTensor(self->GetViewShape(), self->GetDataType(), op::Format::FORMAT_ND);

    if (IsAiCoreSupport(self)) {
        return Log1pAiCore(self, log1pOut, executor);
    } else {
        return Log1pAiCpu(self, log1pOut, executor);
    }
}
}
