/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include "div.h"
#include "opdev/aicpu/aicpu_task.h"
#include "opdev/make_op_executor.h"
#include "opdev/op_def.h"
#include "opdev/op_dfx.h"
#include "opdev/op_executor.h"
#include "opdev/op_log.h"
#include "opdev/platform.h"
#include "opdev/shape_utils.h"

using namespace op;

namespace l0op {

OP_TYPE_REGISTER(Div);

static const std::initializer_list<op::DataType> ASCEND910_AICORE_DTYPE_SUPPORT_LIST = {
    op::DataType::DT_FLOAT, op::DataType::DT_FLOAT16, op::DataType::DT_INT32, op::DataType::DT_INT8,
    op::DataType::DT_UINT8};

static const std::initializer_list<op::DataType> ASCEND910B_AICORE_DTYPE_SUPPORT_LIST = {
    op::DataType::DT_FLOAT, op::DataType::DT_FLOAT16, op::DataType::DT_INT32,
    op::DataType::DT_INT8,  op::DataType::DT_UINT8,   op::DataType::DT_BF16};

static inline const std::initializer_list<op::DataType>& GetAiCoreDtypeSupportListBySocVersion()
{
    auto npuArch = op::GetCurrentPlatformInfo().GetCurNpuArch();
    switch (npuArch) {
        case NpuArch::DAV_2201:
        case NpuArch::DAV_3510: {
            return ASCEND910B_AICORE_DTYPE_SUPPORT_LIST;
        }
        case NpuArch::DAV_1001:
        default: {
            return ASCEND910_AICORE_DTYPE_SUPPORT_LIST;
        }
    }
}

// 根据芯片类型、dtype判断算子是否支持走aicore
static bool IsAiCoreSupport(const aclTensor* self)
{
    // Div只需要判断dtype
    return CheckType(self->GetDataType(), GetAiCoreDtypeSupportListBySocVersion());
}

// AICORE算子kernel
static const aclTensor* DivAiCore(
    const aclTensor* self, const aclTensor* other, aclTensor* divOut, aclOpExecutor* executor)
{
    L0_DFX(DivAiCore, self, other, divOut);
    // 使用框架宏ADD_TO_LAUNCHER_LIST_AICORE，将AiCore Div算子加入任务队列
    // Div是算子的OpType，self、other是算子的输入，divOut是算子的输出
    auto ret = ADD_TO_LAUNCHER_LIST_AICORE(Div, OP_INPUT(self, other), OP_OUTPUT(divOut));
    OP_CHECK(
        ret == ACL_SUCCESS, OP_LOGE(ACLNN_ERR_INNER_NULLPTR, "DivAiCore ADD_TO_LAUNCHER_LIST_AICORE failed."),
        return nullptr);
    return divOut;
}

// AICPU算子kernel
static const aclTensor* DivAiCpu(
    const aclTensor* self, const aclTensor* other, aclTensor* divOut, aclOpExecutor* executor)
{
    // 使用框架宏ADD_TO_LAUNCHER_LIST_AICPU，将AiCpu Div算子加入任务队列
    // Div是算子的OpType，self、other是算子的输入，divOut是算子的输出
    L0_DFX(DivAiCpu, self, other);

    static internal::AicpuTaskSpace space("Div");
    auto ret = ADD_TO_LAUNCHER_LIST_AICPU(Div, OP_ATTR_NAMES(), OP_INPUT(self, other), OP_OUTPUT(divOut));
    OP_CHECK(
        ret == ACL_SUCCESS, OP_LOGE(ACLNN_ERR_INNER_NULLPTR, "DivAiCpu ADD_TO_LAUNCHER_LIST_AICPU failed."),
        return nullptr);
    return divOut;
}

const aclTensor* Div(const aclTensor* self, const aclTensor* other, aclOpExecutor* executor)
{
    op::Shape broadcastShape;
    if (!BroadcastInferShape(self->GetViewShape(), other->GetViewShape(), broadcastShape)) {
        OP_LOGE(
            ACLNN_ERR_PARAM_INVALID, "Broadcast %s and %s failed.", op::ToString(self->GetViewShape()).GetString(),
            op::ToString(other->GetViewShape()).GetString());
        return nullptr;
    }

    auto divOut = executor->AllocTensor(broadcastShape, self->GetDataType());
    if (IsAiCoreSupport(self)) {
        return DivAiCore(self, other, divOut, executor);
    }
    return DivAiCpu(self, other, divOut, executor);
}

} // namespace l0op
