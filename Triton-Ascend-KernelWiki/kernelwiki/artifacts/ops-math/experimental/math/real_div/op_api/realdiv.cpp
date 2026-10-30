/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include "realdiv.h"
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

OP_TYPE_REGISTER(RealDiv);

static const std::initializer_list<op::DataType> AICORE_DTYPE_SUPPORT_LIST = {
    op::DataType::DT_FLOAT, op::DataType::DT_FLOAT16, op::DataType::DT_BF16, op::DataType::DT_BOOL};

static const std::initializer_list<op::DataType> ASCEND610LITE_DTYPE_SUPPORT_LIST = {
    op::DataType::DT_FLOAT, op::DataType::DT_FLOAT16, op::DataType::DT_INT8, op::DataType::DT_UINT8,
    op::DataType::DT_INT32};

static const std::initializer_list<op::DataType> ASCEND950_DTYPE_SUPPORT_LIST = {
    op::DataType::DT_FLOAT, op::DataType::DT_FLOAT16, op::DataType::DT_BF16, op::DataType::DT_BOOL,
    op::DataType::DT_INT32};

// 根据芯片类型、dtype判断算子是否支持走aicore
static bool IsAiCoreSupport(const aclTensor* self)
{
    // 根据dtype返回决定是否走aicore：true则走aicore
    auto socVersion = GetCurrentPlatformInfo().GetSocVersion();
    if (IsRegBase()) {
        return CheckType(self->GetDataType(), ASCEND950_DTYPE_SUPPORT_LIST);
    }
    if (socVersion == SocVersion::ASCEND610LITE) {
        return CheckType(self->GetDataType(), ASCEND610LITE_DTYPE_SUPPORT_LIST);
    }
    return CheckType(self->GetDataType(), AICORE_DTYPE_SUPPORT_LIST);
}

// AICORE算子kernel
static const aclTensor* RealDivAiCore(
    const aclTensor* self, const aclTensor* other, aclTensor* divOut, aclOpExecutor* executor)
{
    L0_DFX(RealDivAiCore, self, other, divOut);
    // 使用框架宏ADD_TO_LAUNCHER_LIST_AICORE，将AiCore RealDiv算子加入任务队列
    // RealDiv是算子的OpType，self、other是算子的输入，divOut是算子的输出
    ADD_TO_LAUNCHER_LIST_AICORE(RealDiv, OP_INPUT(self, other), OP_OUTPUT(divOut));
    return divOut;
}

// AICPU算子kernel
static const aclTensor* RealDivAiCpu(
    const aclTensor* self, const aclTensor* other, aclTensor* divOut, aclOpExecutor* executor)
{
    L0_DFX(RealDivAiCpu);
    static internal::AicpuTaskSpace space("RealDiv");
    auto ret = ADD_TO_LAUNCHER_LIST_AICPU(RealDiv, OP_ATTR_NAMES(), OP_INPUT(self, other), OP_OUTPUT(divOut));
    CHECK_RET(ret == ACLNN_SUCCESS, nullptr);
    return divOut;
}

const aclTensor* RealDiv(const aclTensor* self, const aclTensor* other, aclOpExecutor* executor)
{
    op::Shape broadcastShape;
    if (!BroadcastInferShape(self->GetViewShape(), other->GetViewShape(), broadcastShape)) {
        OP_LOGE(
            ACLNN_ERR_PARAM_INVALID, "Broadcast %s and %s failed.", op::ToString(self->GetViewShape()).GetString(),
            op::ToString(other->GetViewShape()).GetString());
        return nullptr;
    }

    aclTensor* divOut;
    if (self->GetDataType() != op::DataType::DT_BOOL) {
        divOut = executor->AllocTensor(broadcastShape, self->GetDataType());
    } else {
        divOut = executor->AllocTensor(broadcastShape, op::DataType::DT_FLOAT);
    }

    if (IsAiCoreSupport(self)) {
        return RealDivAiCore(self, other, divOut, executor);
    } else {
        return RealDivAiCpu(self, other, divOut, executor);
    }
    return divOut;
}

} // namespace l0op
