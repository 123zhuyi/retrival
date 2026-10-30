/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */
#include "rsqrt.h"
#include "opdev/aicpu/aicpu_task.h"
#include "opdev/make_op_executor.h"
#include "opdev/op_dfx.h"
#include "aclnn_kernels/common/op_error_check.h"

using namespace op;

namespace l0op {
OP_TYPE_REGISTER(Rsqrt);

static const std::initializer_list<op::DataType> AICORE_DTYPE_SUPPORT_LIST = {
    op::DataType::DT_FLOAT, op::DataType::DT_FLOAT16, op::DataType::DT_BF16};

// 根据芯片类型、dtype判断算子是否支持走aicore
inline static bool IsAiCoreSupport(const aclTensor *self)
{
    // Rsqrt只需要判断dtype
    return CheckType(self->GetDataType(), AICORE_DTYPE_SUPPORT_LIST);
}

// AICORE算子kernel
inline static const aclTensor *RsqrtAiCore(const aclTensor *self, aclTensor *rsqrtOut, aclOpExecutor *executor)
{
    L0_DFX(RsqrtAiCore, self, rsqrtOut);
    // 使用框架宏ADD_TO_LAUNCHER_LIST_AICORE，将AiCore Rsqrt算子加入任务队列
    // Rsqrt是算子的OpType，self是算子的输入，rsqrtOut是算子的输出
    auto retAicore = ADD_TO_LAUNCHER_LIST_AICORE(Rsqrt, OP_INPUT(self), OP_OUTPUT(rsqrtOut));
    OP_CHECK_ADD_TO_LAUNCHER_LIST_AICORE(retAicore != ACLNN_SUCCESS, return nullptr,
                                         "Rsqrt ADD_TO_LAUNCHER_LIST_AICORE failed.");
    return rsqrtOut;
}

// AICPU算子kernel
inline static const aclTensor *RsqrtAiCpu(const aclTensor *self, aclTensor *rsqrtOut, aclOpExecutor *executor)
{
    // 使用框架宏ADD_TO_CPU_LAUNCHER_LIST，将AiCpu Rsqrt算子加入任务队列
    // Rsqrt是算子的OpType，self是算子的输入，rsqrtOut是算子的输出
    L0_DFX(RsqrtAiCpu, self, rsqrtOut);

    static internal::AicpuTaskSpace space("Rsqrt");
    auto ret = ADD_TO_LAUNCHER_LIST_AICPU(Rsqrt, OP_ATTR_NAMES(), OP_INPUT(self), OP_OUTPUT(rsqrtOut));
    CHECK_RET(ret == ACLNN_SUCCESS, nullptr);

    return rsqrtOut;
}

const aclTensor *Rsqrt(const aclTensor *self, aclOpExecutor *executor)
{
    auto rsqrtOut = executor->AllocTensor(self->GetViewShape(), self->GetDataType(), op::Format::FORMAT_ND);

    if (IsAiCoreSupport(self)) {
        return RsqrtAiCore(self, rsqrtOut, executor);
    } else {
        return RsqrtAiCpu(self, rsqrtOut, executor);
    }
}
}
