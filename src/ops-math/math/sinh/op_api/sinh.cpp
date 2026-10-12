/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include "sinh.h"
#include "opdev/aicpu/aicpu_task.h"
#include "opdev/make_op_executor.h"
#include "opdev/op_dfx.h"
#include "aclnn_kernels/common/op_error_check.h"
#include "op_api/aclnn_check.h"

using namespace op;
namespace l0op {
OP_TYPE_REGISTER(Sinh);

static const std::initializer_list<op::DataType> AICORE_DTYPE_SUPPORT_LIST = {
    op::DataType::DT_FLOAT, op::DataType::DT_FLOAT16};

static const std::initializer_list<op::DataType> ASCEND910B_AICORE_DTYPE_SUPPORT_LIST = {
    op::DataType::DT_FLOAT, op::DataType::DT_FLOAT16, op::DataType::DT_BF16};

// 根据芯片类型、dtype判断算子是否支持走aicore
static inline bool IsAiCoreSupport(const aclTensor *self)
{
    if (GetCurrentPlatformInfo().GetSocVersion() == SocVersion::ASCEND910B ||
        GetCurrentPlatformInfo().GetSocVersion() == SocVersion::ASCEND910_93 ||
        IsRegBase()) {
        return CheckType(self->GetDataType(), ASCEND910B_AICORE_DTYPE_SUPPORT_LIST);
    }
    return CheckType(self->GetDataType(), AICORE_DTYPE_SUPPORT_LIST);
}

// AICORE算子kernel
static inline const aclTensor *SinhAiCore(const aclTensor *self, aclTensor *sinhOut, aclOpExecutor *executor)
{
    L0_DFX(SinhAiCore, self, sinhOut);
    // 使用框架宏ADD_TO_LAUNCHER_LIST_AICORE，将AiCore Sinh算子加入任务队列
    // Sinh是算子的OpType，self是算子的输入，sinhOut是算子的输出
    auto retAicore = ADD_TO_LAUNCHER_LIST_AICORE(Sinh, OP_INPUT(self), OP_OUTPUT(sinhOut));
    OP_CHECK_ADD_TO_LAUNCHER_LIST_AICORE(retAicore != ACLNN_SUCCESS, return nullptr,
                                         "Sinh ADD_TO_LAUNCHER_LIST_AICORE failed.");
    return sinhOut;
}

// AICPU算子kernel
static inline const aclTensor *SinhAiCpu(const aclTensor *self, aclTensor *sinhOut, aclOpExecutor *executor)
{
    // 使用框架宏ADD_TO_CPU_LAUNCHER_LIST，将AiCpu Sinh算子加入任务队列
    // Sinh是算子的OpType，self是算子的输入，sinhOut是算子的输出
    L0_DFX(SinhAiCpu, self, sinhOut);

    static internal::AicpuTaskSpace space("Sinh", ge::DEPEND_IN_SHAPE, true);
    auto ret = ADD_TO_LAUNCHER_LIST_AICPU(Sinh, OP_ATTR_NAMES(), OP_INPUT(self), OP_OUTPUT(sinhOut));
    CHECK_RET(ret == ACLNN_SUCCESS, nullptr);

    return sinhOut;
}

const aclTensor *Sinh(const aclTensor *self, aclOpExecutor *executor)
{
    auto sinhOut = executor->AllocTensor(self->GetViewShape(), self->GetDataType(), op::Format::FORMAT_ND);
    if (sinhOut == nullptr) {
        OP_LOGE(ACLNN_ERR_INNER_NULLPTR, "alloc out tensor failed.");
        return nullptr;
    }
    
    if (IsAiCoreSupport(self)) {
        return SinhAiCore(self, sinhOut, executor);
    } else {
        return SinhAiCpu(self, sinhOut, executor);
    }
}
}
