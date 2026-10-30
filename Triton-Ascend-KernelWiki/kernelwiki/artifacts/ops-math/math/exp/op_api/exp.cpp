/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include "exp.h"
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
#include "op_api/aclnn_check.h"

using namespace op;

namespace l0op {

OP_TYPE_REGISTER(Exp);

static const std::initializer_list<DataType> AICORE910_DTYPE_SUPPORT_LIST = {DataType::DT_FLOAT, DataType::DT_FLOAT16};

static const std::initializer_list<DataType> AICORE910B_DTYPE_SUPPORT_LIST = {
    DataType::DT_FLOAT, DataType::DT_FLOAT16, op::DataType::DT_BF16};

// 根据芯片类型、dtype判断算子是否支持走aicore
static inline bool IsAiCoreSupport(const aclTensor* self)
{
    auto npuArch = op::GetCurrentPlatformInfo().GetCurNpuArch();
    if (npuArch == NpuArch::DAV_2201 || IsRegBase(npuArch)) {
        return CheckType(self->GetDataType(), AICORE910B_DTYPE_SUPPORT_LIST);
    }

    // 1980
    return CheckType(self->GetDataType(), AICORE910_DTYPE_SUPPORT_LIST);
}

// AICORE算子kernel
static inline const aclTensor* ExpAiCore(const aclTensor* self, aclTensor* out, aclOpExecutor* executor)
{
    L0_DFX(ExpAiCore, self, out);
    // 使用框架宏ADD_TO_LAUNCHER_LIST_AICORE，将AiCore Exp算子加入任务队列
    // Exp是算子的OpType，self是算子的输入，out是算子的输出
    const float base = -1.0f;
    const float scale = 1.0f;
    const float shift = 0.0f;
    auto retAicore = ADD_TO_LAUNCHER_LIST_AICORE(Exp, OP_INPUT(self), OP_OUTPUT(out), OP_ATTR(base, scale, shift));
    OP_CHECK(
        retAicore == ACLNN_SUCCESS, OP_LOGE(ACLNN_ERR_INNER_NULLPTR, "Exp ADD_TO_LAUNCHER_LIST_AICORE failed."),
        return nullptr);
    return out;
}

// AICPU算子kernel
static inline const aclTensor* ExpAiCpu(const aclTensor* self, aclTensor* out, aclOpExecutor* executor)
{
    L0_DFX(ExpAiCpu, self, out);

    // 使用框架宏ADD_TO_LAUNCHER_LIST_AICPU，将AiCpu Exp算子加入任务队列
    // Exp是算子的OpType，self是算子的输入，out是算子的输出
    aclnnStatus ret = ACL_ERROR_INVALID_PARAM;
    if (self->GetDataType() != ge::DataType::DT_COMPLEX64) {
      static internal::AicpuTaskSpace space("Exp", ge::DEPEND_IN_SHAPE, true);
      ret = ADD_TO_LAUNCHER_LIST_AICPU(Exp, OP_ATTR_NAMES(), OP_INPUT(self), OP_OUTPUT(out)); 
    } else {
      static internal::AicpuTaskSpace space("Exp", ge::DEPEND_IN_SHAPE, false);
      ret = ADD_TO_LAUNCHER_LIST_AICPU(Exp, OP_ATTR_NAMES(), OP_INPUT(self), OP_OUTPUT(out));
    }
    OP_CHECK(
        ret == ACL_SUCCESS, OP_LOGE(ACLNN_ERR_INNER_NULLPTR, "ExpAiCpu ADD_TO_LAUNCHER_LIST_AICPU failed."),
        return nullptr);
    return out;
}

const aclTensor* Exp(const aclTensor* self, aclOpExecutor* executor)
{
    L0_DFX(Exp, self);
    // 根据输入shape申请输出tensor
    auto out = executor->AllocTensor(self->GetViewShape(), self->GetDataType(), Format::FORMAT_ND);
    if (out == nullptr) {
        OP_LOGE(ACLNN_ERR_INNER_NULLPTR, "alloc out tensor failed.");
        return nullptr;
    }

    if (IsAiCoreSupport(self)) {
        return ExpAiCore(self, out, executor);
    } else {
        return ExpAiCpu(self, out, executor);
    }
}

} // namespace l0op