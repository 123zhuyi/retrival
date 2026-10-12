/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include "select_v2.h"
#include "opdev/op_log.h"
#include "opdev/op_executor.h"
#include "opdev/make_op_executor.h"
#include "opdev/shape_utils.h"
#include "opdev/aicpu/aicpu_task.h"
#include "opdev/shape_utils.h"
#include "opdev/op_def.h"
#include "opdev/op_dfx.h"

using namespace op;

namespace l0op {
OP_TYPE_REGISTER(SelectV2);

static const std::initializer_list<op::DataType> AICORE_DTYPE_SUPPORT_LIST = {
    op::DataType::DT_FLOAT, op::DataType::DT_FLOAT16, op::DataType::DT_INT32, op::DataType::DT_INT8,
    op::DataType::DT_UINT8, op::DataType::DT_BF16,    op::DataType::DT_INT64};

static const std::initializer_list<op::DataType> ASCEND610LITE_DTYPE_SUPPORT_LIST = {
    op::DataType::DT_FLOAT, op::DataType::DT_FLOAT16, op::DataType::DT_INT32, op::DataType::DT_INT8,
    op::DataType::DT_UINT8};

// 根据芯片类型、dtype判断算子是否支持走aicore
static bool IsAiCoreSupport(const aclTensor* self)
{
    // Select只需要判断dtype
    if (GetCurrentPlatformInfo().GetSocVersion() == SocVersion::ASCEND610LITE) {
        return CheckType(self->GetDataType(), ASCEND610LITE_DTYPE_SUPPORT_LIST);
    }
    return CheckType(self->GetDataType(), AICORE_DTYPE_SUPPORT_LIST);
}

static bool CanBroadcast(const aclTensor* t1, const aclTensor* t2, const aclTensor* t3, op::Shape& broadcastShape)
{
    op::Shape tempShape;
    if (BroadcastInferShape(t1->GetViewShape(), t2->GetViewShape(), tempShape)) {
        return BroadcastInferShape(t3->GetViewShape(), tempShape, broadcastShape);
    }
    return false;
}

// AICORE算子kernel
static const aclTensor* SelectV2AiCore(
    const aclTensor* condition, const aclTensor* self, const aclTensor* other, aclTensor* selectOut,
    aclOpExecutor* executor)
{
    L0_DFX(SelectV2AiCore, condition, self, other);
    // 使用框架宏ADD_TO_LAUNCHER_LIST_AICORE，将AiCore Select算子加入任务队列
    // Select是算子的OpType，self、other是算子的输入，selectOut是算子的输出
    ADD_TO_LAUNCHER_LIST_AICORE(SelectV2, OP_INPUT(condition, self, other), OP_OUTPUT(selectOut));
    return selectOut;
}

// AICPU算子kernel
static const aclTensor* SelectV2AiCpu(
    const aclTensor* condition, const aclTensor* self, const aclTensor* other, aclTensor* selectOut,
    aclOpExecutor* executor)
{
    // 使用框架宏ADD_TO_LAUNCHER_LIST_AICPU，将AiCpu Select算子加入任务队列
    // SelectV2是算子的OpType，self、other是算子的输入，selectOut是算子的输出
    L0_DFX(SelectV2AiCpu, condition, self, other);
    static internal::AicpuTaskSpace space("SelectV2");
    auto ret =
        ADD_TO_LAUNCHER_LIST_AICPU(SelectV2, OP_ATTR_NAMES(), OP_INPUT(condition, self, other), OP_OUTPUT(selectOut));
    CHECK_RET(ret == ACLNN_SUCCESS, nullptr);
    return selectOut;
}

const aclTensor* SelectV2(const aclTensor* condition, const aclTensor* x1, const aclTensor* x2, aclOpExecutor* executor)
{
    op::Shape broadcastShape;
    if (!CanBroadcast(condition, x1, x2, broadcastShape)) {
        OP_LOGE(
            ACL_ERROR_INVALID_PARAM, "broadcast condition: %s, x1: %s, x2: %s failed.",
            op::ToString(condition->GetViewShape()).GetString(), op::ToString(x1->GetViewShape()).GetString(),
            op::ToString(x2->GetViewShape()).GetString());
        return nullptr;
    }
    auto selectOut = executor->AllocTensor(broadcastShape, x1->GetDataType());
    CHECK_RET(selectOut != nullptr, nullptr);
    if (IsAiCoreSupport(x1)) {
        return SelectV2AiCore(condition, x1, x2, selectOut, executor);
    } else {
        return SelectV2AiCpu(condition, x1, x2, selectOut, executor);
    }
    return selectOut;
}
} // namespace l0op