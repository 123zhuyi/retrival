/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include "sub.h"
#include "opdev/aicpu/aicpu_task.h"
#include "opdev/data_type_utils.h"
#include "opdev/format_utils.h"
#include "opdev/make_op_executor.h"
#include "opdev/op_def.h"
#include "opdev/op_dfx.h"
#include "opdev/op_executor.h"
#include "opdev/op_log.h"
#include "opdev/shape_utils.h"
#include "opdev/platform.h"

using namespace op;

namespace l0op {

OP_TYPE_REGISTER(Sub);

static const std::initializer_list<op::DataType> ASCEND910_AICORE_DTYPE_SUPPORT_LIST = {
    op::DataType::DT_FLOAT, op::DataType::DT_FLOAT16, op::DataType::DT_INT32, op::DataType::DT_INT64,
    op::DataType::DT_INT8,  op::DataType::DT_UINT8,   op::DataType::DT_BOOL};

static const std::initializer_list<op::DataType> ASCEND910B_AICORE_DTYPE_SUPPORT_LIST = {
    op::DataType::DT_FLOAT, op::DataType::DT_FLOAT16, op::DataType::DT_BF16,
    op::DataType::DT_INT64, op::DataType::DT_INT32,   op::DataType::DT_INT8,
    op::DataType::DT_UINT8, op::DataType::DT_BOOL,    op::DataType::DT_COMPLEX64};

static const std::initializer_list<op::DataType> ASCEND610LITE_DTYPE_SUPPORT_LIST = {
    op::DataType::DT_FLOAT, op::DataType::DT_FLOAT16, op::DataType::DT_INT32};

// 根据芯片类型、dtype判断算子是否支持走aicore
static bool IsAiCoreSupport(const aclTensor* self)
{
    // 获取芯片类型,判断是1971还是1980
    if (GetCurrentPlatformInfo().GetSocVersion() >= SocVersion::ASCEND910B &&
        GetCurrentPlatformInfo().GetSocVersion() <= SocVersion::ASCEND910E) {
        return CheckType(self->GetDataType(), ASCEND910B_AICORE_DTYPE_SUPPORT_LIST);
    }

    if (GetCurrentPlatformInfo().GetSocVersion() == SocVersion::ASCEND610LITE) {
        return CheckType(self->GetDataType(), ASCEND610LITE_DTYPE_SUPPORT_LIST);
    }
    // 1980 & other
    return CheckType(self->GetDataType(), ASCEND910_AICORE_DTYPE_SUPPORT_LIST);
}

// AICORE算子kernel
static const aclTensor* SubAiCore(
    const aclTensor* self, const aclTensor* other, aclTensor* subOut, aclOpExecutor* executor)
{
    L0_DFX(SubAiCore, self, other, subOut);
    // 使用框架宏ADD_TO_LAUNCHER_LIST_AICORE，将AiCore Sub算子加入任务队列
    // Sub是算子的OpType，self、other是算子的输入，subOut是算子的输出
    ADD_TO_LAUNCHER_LIST_AICORE(Sub, OP_INPUT(self, other), OP_OUTPUT(subOut));
    return subOut;
}

// AICPU算子kernel
static const aclTensor* SubAiCpu(
    const aclTensor* self, const aclTensor* other, aclTensor* subOut, aclOpExecutor* executor)
{
    // 使用框架宏ADD_TO_LAUNCHER_LIST_AICPU，将AiCpu Sub算子加入任务队列
    // Sub是算子的OpType，self、other是算子的输入，subOut是算子的输出
    L0_DFX(SubAiCpu, self, other);

    static internal::AicpuTaskSpace space("Sub");
    auto ret = ADD_TO_LAUNCHER_LIST_AICPU(Sub, OP_ATTR_NAMES(), OP_INPUT(self, other), OP_OUTPUT(subOut));
    CHECK_RET(ret == ACLNN_SUCCESS, nullptr);
    return subOut;
}

const aclTensor* Sub(const aclTensor* self, const aclTensor* other, aclOpExecutor* executor)
{
    // 通过输入shape推导算子输出shape
    op::Shape broadcastShape;
    if (!BroadcastInferShape(self->GetViewShape(), other->GetViewShape(), broadcastShape)) {
        OP_LOGE(
            ACLNN_ERR_PARAM_INVALID, "Broadcast %s and %s failed.", op::ToString(self->GetViewShape()).GetString(),
            op::ToString(other->GetViewShape()).GetString());
        return nullptr;
    }

    auto subOut = executor->AllocTensor(broadcastShape, self->GetDataType());

    // 根据推导出的输出shape申请输出tensor
    if (IsAiCoreSupport(self)) {
        return SubAiCore(self, other, subOut, executor);
    } else {
        return SubAiCpu(self, other, subOut, executor);
    }
}
} // namespace l0op