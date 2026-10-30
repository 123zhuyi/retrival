/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include "clip_by_value.h"
#include "opdev/data_type_utils.h"
#include "opdev/make_op_executor.h"
#include "opdev/op_dfx.h"
#include "opdev/op_log.h"
#include "opdev/shape_utils.h"
#include "opdev/aicpu/aicpu_task.h"

using namespace op;

namespace l0op {

OP_TYPE_REGISTER(ClipByValueV2);

static const std::initializer_list<op::DataType> AICORE_DTYPE_SUPPORT_LIST = {
    op::DataType::DT_FLOAT16, op::DataType::DT_FLOAT, op::DataType::DT_INT32, op::DataType::DT_INT64,
    DataType::DT_BF16};

// 根据芯片类型、dtype判断算子是否支持走aicore
static bool IsAiCoreSupport(const aclTensor* self)
{
    // Add只需要判断dtype
    return op::CheckType(self->GetDataType(), AICORE_DTYPE_SUPPORT_LIST);
}

// AICORE算子kernel
static const aclTensor* ClipByValueV2AiCore(
    const aclTensor* self, const aclTensor* clipValueMin, const aclTensor* clipValueMax, aclOpExecutor* executor)
{
    L0_DFX(ClipByValueV2AiCore);
    auto out = executor->AllocTensor(
        self->GetDataType(), static_cast<op::Format>(self->GetStorageFormat()), self->GetOriginalFormat());
    auto ret = INFER_SHAPE(ClipByValueV2, OP_INPUT(self, clipValueMin, clipValueMax), OP_OUTPUT(out));
    if (ret != ACLNN_SUCCESS) {
        OP_LOGE(ACLNN_ERR_PARAM_INVALID, "infershape failed.");
        return nullptr;
    }

    ret = ADD_TO_LAUNCHER_LIST_AICORE(ClipByValueV2, OP_INPUT(self, clipValueMin, clipValueMax), OP_OUTPUT(out));
    OP_CHECK(
        ret == ACL_SUCCESS, OP_LOGE(ACLNN_ERR_INNER_NULLPTR, "ClipByValueV2AiCore ADD_TO_LAUNCHER_LIST_AICORE failed."),
        return nullptr);
    return out;
}

// AICPU算子kernel
static const aclTensor* ClipByValueV2AiCpu(
    const aclTensor* self, const aclTensor* clipValueMin, const aclTensor* clipValueMax, aclOpExecutor* executor)
{
    L0_DFX(ClipByValueV2AiCpu);
    auto out = executor->AllocTensor(self->GetViewShape(), self->GetDataType());

    static internal::AicpuTaskSpace space("ClipByValueV2");
    auto ret = ADD_TO_LAUNCHER_LIST_AICPU(
        ClipByValueV2, OP_ATTR_NAMES(), OP_INPUT(self, clipValueMin, clipValueMax), OP_OUTPUT(out));
    OP_CHECK(
        ret == ACL_SUCCESS, OP_LOGE(ACLNN_ERR_INNER_NULLPTR, "ClipByValueV2AiCpu ADD_TO_LAUNCHER_LIST_AICPU failed."),
        return nullptr);
    return out;
}

const aclTensor* ClipByValueV2(
    const aclTensor* self, const aclTensor* clipValueMin, const aclTensor* clipValueMax, aclOpExecutor* executor)
{
    if (IsAiCoreSupport(self)) {
        return ClipByValueV2AiCore(self, clipValueMin, clipValueMax, executor);
    } else {
        return ClipByValueV2AiCpu(self, clipValueMin, clipValueMax, executor);
    }
}
} // namespace l0op
