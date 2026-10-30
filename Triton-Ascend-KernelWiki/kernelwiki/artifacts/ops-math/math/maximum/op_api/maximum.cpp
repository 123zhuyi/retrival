/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include "maximum.h"
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

OP_TYPE_REGISTER(Maximum);

static const std::initializer_list<op::DataType> ASCEND610LITE_DTYPE_SUPPORT_LIST = {
    op::DataType::DT_FLOAT, op::DataType::DT_FLOAT16, op::DataType::DT_INT32, op::DataType::DT_INT8};

static const std::initializer_list<op::DataType> AICORE_DTYPE_SUPPORT_LIST = {
    op::DataType::DT_FLOAT, op::DataType::DT_FLOAT16, op::DataType::DT_INT32,
    op::DataType::DT_INT8,  op::DataType::DT_INT64,   op::DataType::DT_BF16};

static const std::initializer_list<op::DataType> ASCEND950_DTYPE_SUPPORT_LIST = {
    op::DataType::DT_FLOAT, op::DataType::DT_FLOAT16, op::DataType::DT_INT32, op::DataType::DT_INT8,
    op::DataType::DT_UINT8, op::DataType::DT_INT64,   op::DataType::DT_BF16};

// 根据芯片类型、dtype判断算子是否支持走AiCore
static bool IsAiCoreSupport(const aclTensor* self)
{
    auto socVersion = GetCurrentPlatformInfo().GetSocVersion();
    if (IsRegBase()) {
        return CheckType(self->GetDataType(), ASCEND950_DTYPE_SUPPORT_LIST);
    }
    if (socVersion == SocVersion::ASCEND610LITE) {
        return CheckType(self->GetDataType(), ASCEND610LITE_DTYPE_SUPPORT_LIST);
    }
    // Maximum只需要判断dtype
    return CheckType(self->GetDataType(), AICORE_DTYPE_SUPPORT_LIST);
}

// AICORE算子kernel
static const aclTensor* MaximumAiCore(
    const aclTensor* self, const aclTensor* other, const aclTensor* maximumOut, aclOpExecutor* executor)
{
    L0_DFX(MaximumAiCore, self, other, maximumOut)
    // 使用框架宏ADD_TO_LAUNCHER_LIST_AICORE，将Maximum算子加入任务队列
    // Maximum是算子的OpType，self、other是算子的输入，maximumOut是算子的输出
    ADD_TO_LAUNCHER_LIST_AICORE(Maximum, OP_INPUT(self, other), OP_OUTPUT(maximumOut));
    return maximumOut;
}

// AICPU算子kernel
static const aclTensor* MaximumAiCpu(
    const aclTensor* self, const aclTensor* other, aclTensor* maximumOut, aclOpExecutor* executor)
{
    L0_DFX(MaximumAiCpu, self, other, maximumOut)
    // 使用框架宏ADD_TO_LAUNCHER_LIST_AICORE，将AiCpu Maximum算子加入任务队列
    // Maximum是算子的OpType，self、other是算子的输入，maximumOut是算子的输出
    static internal::AicpuTaskSpace space("Maximum", ge::DEPEND_IN_SHAPE, true);
    auto ret = ADD_TO_LAUNCHER_LIST_AICPU(Maximum, OP_ATTR_NAMES(), OP_INPUT(self, other), OP_OUTPUT(maximumOut));
    CHECK_RET(ret == ACLNN_SUCCESS, nullptr);
    return maximumOut;
}

const aclTensor* Maximum(const aclTensor* self, const aclTensor* other, aclOpExecutor* executor)
{
    op::Shape broadcastShape;
    if (!BroadcastInferShape(self->GetViewShape(), other->GetViewShape(), broadcastShape)) {
        OP_LOGE(
            ACLNN_ERR_PARAM_INVALID, "Broadcast %s and %s failed.", op::ToString(self->GetViewShape()).GetString(),
            op::ToString(self->GetViewShape()).GetString());
        return nullptr;
    }

    auto maximumOut = executor->AllocTensor(broadcastShape, self->GetDataType());
    if (IsAiCoreSupport(self)) {
        return MaximumAiCore(self, other, maximumOut, executor);
    } else {
        return MaximumAiCpu(self, other, maximumOut, executor);
    }
}
} // namespace l0op
