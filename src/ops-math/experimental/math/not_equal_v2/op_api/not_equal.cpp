/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include "not_equal_v2.h"
#include "opdev/aicpu/aicpu_task.h"
#include "opdev/make_op_executor.h"
#include "opdev/op_def.h"
#include "opdev/op_dfx.h"
#include "opdev/op_executor.h"
#include "opdev/op_log.h"
#include "opdev/shape_utils.h"
#include "op_api/aclnn_check.h"

using namespace op;

namespace l0op {

OP_TYPE_REGISTER(NotEqualV2);

// AICORE
static const std::initializer_list<op::DataType> AICORE910_DTYPE_SUPPORT_LIST = {
    op::DataType::DT_FLOAT, op::DataType::DT_INT32, op::DataType::DT_INT64, op::DataType::DT_FLOAT16,
    op::DataType::DT_INT8,  op::DataType::DT_UINT8, op::DataType::DT_BOOL,  op::DataType::DT_BF16};

static const std::initializer_list<op::DataType> ASCEND950_DTYPE_SUPPORT_LIST = {
    op::DataType::DT_FLOAT,   op::DataType::DT_INT32, op::DataType::DT_INT64,
    op::DataType::DT_FLOAT16, op::DataType::DT_INT8,  op::DataType::DT_UINT8,
    op::DataType::DT_BOOL,    op::DataType::DT_BF16,  op::DataType::DT_UINT64};

// AICPU TF
static const std::initializer_list<op::DataType> AICPU_TF_DTYPE_SUPPORT_LIST = {
    op::DataType::DT_COMPLEX64, op::DataType::DT_COMPLEX128};

// 根据芯片类型、dtype判断算子是否支持走aicore
static bool IsAiCoreSupport(const aclTensor* self)
{
    if (IsRegBase()) {
        return CheckType(self->GetDataType(), ASCEND950_DTYPE_SUPPORT_LIST);
    }
    return CheckType(self->GetDataType(), AICORE910_DTYPE_SUPPORT_LIST);
}

// AICORE算子kernel
static const aclTensor* NotEqualV2AiCore(
    const aclTensor* self, const aclTensor* other, aclTensor* neOut, aclOpExecutor* executor)
{
    L0_DFX(NotEqualV2AiCore, self, other, neOut);
    // 使用框架宏 ADD_TO_LAUNCHER_LIST_AICORE，将NotEqualV2算子加入任务队列
    auto ret = ADD_TO_LAUNCHER_LIST_AICORE(NotEqualV2, OP_INPUT(self, other), OP_OUTPUT(neOut));
    OP_CHECK(
        ret == ACLNN_SUCCESS, OP_LOGE(ACLNN_ERR_INNER_NULLPTR, "NotEqualV2AiCore ADD_TO_LAUNCHER_LIST_AICORE failed."),
        return nullptr);
    return neOut;
}

// AICPU算子kernel
static const aclTensor* NotEqualV2AiCpu(
    const aclTensor* self, const aclTensor* other, aclTensor* neOut, aclOpExecutor* executor)
{
    L0_DFX(NotEqualV2AiCpu, self, other);

    static internal::AicpuTaskSpace space("NotEqualV2");
    // 使用框架宏ADD_TO_LAUNCHER_LIST_AICPU，将NotEqualV2算子加入任务队列
    auto ret = ADD_TO_LAUNCHER_LIST_AICPU(NotEqualV2, OP_ATTR_NAMES(), OP_INPUT(self, other), OP_OUTPUT(neOut));
    CHECK_RET(ret == ACLNN_SUCCESS, nullptr);

    return neOut;
}

// AICPU算子 TF
static const aclTensor* NotEqualV2AiCpuTf(
    const aclTensor* self, const aclTensor* other, aclTensor* neOut, aclOpExecutor* executor)
{
    L0_DFX(NotEqualV2AiCpuTf, self, other);

    // 走TF
    static internal::AicpuTaskSpace space("NotEqualV2", ge::DEPEND_IN_SHAPE, true);

    // 使用框架宏ADD_TO_LAUNCHER_LIST_AICPU，将NotEqualV2算子加入任务队列
    auto ret = ADD_TO_LAUNCHER_LIST_AICPU(
        NotEqualV2, OP_ATTR_NAMES({"T", "incompatible_shape_error"}), OP_INPUT(self, other), OP_OUTPUT(neOut),
        OP_ATTR(self->GetDataType(), true));
    CHECK_RET(ret == ACLNN_SUCCESS, nullptr);

    return neOut;
}

const aclTensor* NotEqualV2(const aclTensor* self, const aclTensor* other, aclOpExecutor* executor)
{
    op::Shape outShape;
    if (!BroadcastInferShape(self->GetViewShape(), other->GetViewShape(), outShape)) {
        OP_LOGE(
            ACLNN_ERR_PARAM_INVALID, "Broadcast %s and %s failed.", op::ToString(self->GetViewShape()).GetString(),
            op::ToString(other->GetViewShape()).GetString());
        return nullptr;
    }

    auto neOut = executor->AllocTensor(outShape, op::DataType::DT_BOOL, op::Format::FORMAT_ND);

    if (IsAiCoreSupport(self) && IsAiCoreSupport(other)) {
        return NotEqualV2AiCore(self, other, neOut, executor);
    } else {
        if (CheckType(self->GetDataType(), AICPU_TF_DTYPE_SUPPORT_LIST) ||
            CheckType(other->GetDataType(), AICPU_TF_DTYPE_SUPPORT_LIST)) {
            return NotEqualV2AiCpuTf(self, other, neOut, executor);
        } else {
            return NotEqualV2AiCpu(self, other, neOut, executor);
        }
    }
}
} // namespace l0op
