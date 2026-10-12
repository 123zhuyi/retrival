/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include "neg.h"
#include "opdev/aicpu/aicpu_task.h"
#include "opdev/op_log.h"
#include "opdev/op_executor.h"
#include "opdev/make_op_executor.h"
#include "opdev/op_def.h"
#include "opdev/op_dfx.h"
#include "aclnn_kernels/common/op_error_check.h"
#include "op_api/aclnn_check.h"

using namespace op;

namespace l0op {

OP_TYPE_REGISTER(Neg);

static const std::initializer_list<op::DataType> AICORE_DTYPE_SUPPORT_LIST = {
    op::DataType::DT_FLOAT, op::DataType::DT_FLOAT16, op::DataType::DT_INT32, op::DataType::DT_INT8,
    op::DataType::DT_BF16};

static const std::initializer_list<op::DataType> AICORE_DTYPE_SUPPORT_LIST_AFTER_910B = {
    op::DataType::DT_FLOAT, op::DataType::DT_FLOAT16, op::DataType::DT_INT32,
    op::DataType::DT_INT8,  op::DataType::DT_BF16,    op::DataType::DT_INT64};

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

// AICORE算子kernel
static const aclTensor* NegAiCore(const aclTensor* self, aclTensor* negOut, aclOpExecutor* executor)
{
    L0_DFX(NegAiCore, self, negOut);
    // 使用框架宏ADD_TO_LAUNCHER_LIST_AICORE，将AiCore Neg算子加入任务队列
    // Neg是算子的OpType，self是算子的输入，negOut是算子的输出
    auto retAicore = ADD_TO_LAUNCHER_LIST_AICORE(Neg, OP_INPUT(self), OP_OUTPUT(negOut));
    OP_CHECK_ADD_TO_LAUNCHER_LIST_AICORE(
        retAicore != ACLNN_SUCCESS, return nullptr, "Neg ADD_TO_LAUNCHER_LIST_AICORE failed.");
    return negOut;
}

// AICPU算子kernel
static const aclTensor* NegAiCpu(const aclTensor* self, aclTensor* negOut, aclOpExecutor* executor)
{
    // 使用框架宏ADD_TO_AICPU__LAUNCHER_LIST，将AiCpu Neg算子加入任务队列
    // Neg是算子的OpType，self是算子的输入，negOut是算子的输出
    L0_DFX(NegAiCpu, self, negOut);

    static internal::AicpuTaskSpace space("Neg");
    auto ret = ADD_TO_LAUNCHER_LIST_AICPU(Neg, OP_ATTR_NAMES(), OP_INPUT(self), OP_OUTPUT(negOut));
    CHECK_RET(ret == ACLNN_SUCCESS, nullptr);
    return negOut;
}

const aclTensor* Neg(const aclTensor* self, aclOpExecutor* executor)
{
    auto negOut = executor->AllocTensor(self->GetViewShape(), self->GetDataType());
    if (IsAiCoreSupport(self->GetDataType())) {
        return NegAiCore(self, negOut, executor);
    }

    return NegAiCpu(self, negOut, executor);
} // namespace l0op

} // namespace l0op