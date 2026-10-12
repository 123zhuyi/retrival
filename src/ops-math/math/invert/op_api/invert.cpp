/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include "invert.h"
#include "opdev/data_type_utils.h"
#include "opdev/make_op_executor.h"
#include "opdev/op_def.h"
#include "opdev/op_dfx.h"
#include "opdev/op_executor.h"
#include "opdev/op_log.h"
#include "opdev/shape_utils.h"
#include "opdev/aicpu/aicpu_task.h"
#include "op_api/aclnn_check.h"

using namespace op;

namespace l0op {
OP_TYPE_REGISTER(Invert);

static const std::initializer_list<op::DataType> AICORE_DTYPE_SUPPORT_LIST = {
    op::DataType::DT_INT16, op::DataType::DT_UINT16};

static const std::initializer_list<op::DataType> REGBASE_DTYPE_SUPPORT_LIST = {
    op::DataType::DT_INT8,  op::DataType::DT_INT16,  op::DataType::DT_INT32,  op::DataType::DT_INT64,
    op::DataType::DT_UINT8, op::DataType::DT_UINT16, op::DataType::DT_UINT32, op::DataType::DT_UINT64};

// 根据芯片类型、dtype判断算子是否支持走aicore
static inline bool IsAiCoreSupport(const aclTensor* self)
{
    // Invert只需要判断dtype
    auto npuArch = op::GetCurrentPlatformInfo().GetCurNpuArch();
    if (IsRegBase(npuArch)) {
        return CheckType(self->GetDataType(), REGBASE_DTYPE_SUPPORT_LIST);
    }

    return CheckType(self->GetDataType(), AICORE_DTYPE_SUPPORT_LIST);
}

// AICORE算子kernel
static const aclTensor* InvertAiCore(const aclTensor* self, const aclTensor* out, aclOpExecutor* executor)
{
    L0_DFX(InvertAiCore);
    // 使用框架宏ADD_TO_LAUNCHER_LIST，将Aicore Invert算子加入任务队列
    // Invert是算子的Optype，self是算子的输入，out是算子的输出
    auto ret = ADD_TO_LAUNCHER_LIST_AICORE(Invert, OP_INPUT(self), OP_OUTPUT(out));
    OP_CHECK(
        ret == ACLNN_SUCCESS, OP_LOGE(ACLNN_ERR_INNER_NULLPTR, "Invert ADD_TO_LAUNCHER_LIST_AICORE failed."),
        return nullptr);
    return out;
}

// AICPU算子kernel
static const aclTensor* InvertAiCpu(const aclTensor* self, aclTensor* out, aclOpExecutor* executor)
{
    // 使用框架宏ADD_TO_LAUNCHER_LIST，将AiCpu Invert算子加入任务队列
    // Invert是算子的Optype，self是算子的输入，out是算子的输出，走tf_kernel
    L0_DFX(InvertAiCpu);
    static internal::AicpuTaskSpace space("Invert", ge::DEPEND_IN_SHAPE, true);
    auto ret = ADD_TO_LAUNCHER_LIST_AICPU(Invert, OP_ATTR_NAMES(), OP_INPUT(self), OP_OUTPUT(out));
    OP_CHECK(
        ret == ACLNN_SUCCESS, OP_LOGE(ACLNN_ERR_INNER_NULLPTR, "InvertAiCpu ADD_TO_LAUNCHER_LIST_AICPU failed."),
        return nullptr);
    return out;
}

const aclTensor* Invert(const aclTensor* self, aclOpExecutor* executor)
{
    auto out = executor->AllocTensor(self->GetViewShape(), self->GetDataType());
    if (IsAiCoreSupport(self)) {
        return InvertAiCore(self, out, executor);
    } else {
        return InvertAiCpu(self, out, executor);
    }
    return out;
}
} // namespace l0op